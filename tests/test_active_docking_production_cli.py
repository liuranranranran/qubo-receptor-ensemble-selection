from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        workflow="active_ligand_receptor_docking",
        fingerprint="a" * 64,
        prepared_run_directory=Path("prepared"),
        active_run_directory=Path("active"),
    )


def test_validate_does_not_construct_runner_or_require_matrix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from scripts import run_active_experiment as cli

    monkeypatch.setattr(cli, "load_active_production_config", lambda *args, **kwargs: _fake_config())

    class ForbiddenRunner:
        def __init__(self, config: object) -> None:
            raise AssertionError("validate must not construct a docking runner")

    monkeypatch.setattr(cli, "ActiveProductionRunner", ForbiddenRunner)
    assert cli.main(["validate", "--config", str(tmp_path / "active.json")]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "valid"
    assert output["real_docking_executed"] is False


def test_prepared_run_directory_override_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from scripts import run_active_experiment as cli

    calls: list[tuple[object, object, object]] = []

    def fake_load(path: object, *, data_root: object, prepared_run_directory: object) -> SimpleNamespace:
        calls.append((path, data_root, prepared_run_directory))
        return _fake_config()

    monkeypatch.setattr(cli, "load_active_production_config", fake_load)
    class FakeRunner:
        def __init__(self, config: object, *, progress: object) -> None:
            del config
            del progress

        def prepare(self, **kwargs: object) -> dict[str, object]:
            assert kwargs == {"overwrite": False}
            return {"status": "prepared"}

    monkeypatch.setattr(cli, "ActiveProductionRunner", FakeRunner)
    prepared = tmp_path / "history"
    assert cli.main([
        "prepare",
        "--config",
        str(tmp_path / "active.json"),
        "--data-root",
        str(tmp_path / "data"),
        "--prepared-run-directory",
        str(prepared),
    ]) == 0
    assert calls == [(
        tmp_path / "active.json",
        tmp_path / "data",
        prepared,
    )]
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_empty_prepared_run_directory_argument_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_active_experiment as cli

    monkeypatch.setattr(cli, "load_active_production_config", lambda *args, **kwargs: _fake_config())

    with pytest.raises(SystemExit) as error:
        cli.main([
            "validate",
            "--config",
            str(tmp_path / "active.json"),
            "--prepared-run-directory",
            "",
        ])

    assert error.value.code == 2


def test_run_resume_and_finalize_route_to_separate_runner_methods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import run_active_experiment as cli

    monkeypatch.setattr(cli, "load_active_production_config", lambda *args, **kwargs: _fake_config())
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeRunner:
        def __init__(self, config: object, *, progress: object) -> None:
            del config
            del progress

        def run(self, **kwargs: object) -> dict[str, object]:
            calls.append(("run", kwargs))
            return {"status": "completed", "real_docking_executed": True}

        def finalize(self) -> dict[str, object]:
            calls.append(("finalize", {}))
            return {"status": "finalized"}

    monkeypatch.setattr(cli, "ActiveProductionRunner", FakeRunner)
    assert cli.main(["run", "--config", str(tmp_path / "active.json"), "--max-rounds", "2"]) == 0
    assert cli.main(["resume", "--config", str(tmp_path / "active.json")]) == 0
    assert cli.main(["finalize", "--config", str(tmp_path / "active.json")]) == 0

    assert calls[0] == ("run", {"resume": False, "overwrite": False, "max_rounds": 2})
    assert calls[1] == ("run", {"resume": True, "overwrite": False, "max_rounds": None})
    assert calls[2] == ("finalize", {})


def test_production_cli_writes_progress_to_stderr_without_polluting_json_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from scripts import run_active_experiment as cli

    monkeypatch.setattr(cli, "load_active_production_config", lambda *args, **kwargs: _fake_config())

    class FakeRunner:
        def __init__(self, config: object, *, progress: object) -> None:
            del config
            self.progress = progress

        def run(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            assert callable(self.progress)
            self.progress("[active][round=002][stage=solver] done selected=3")
            return {"status": "completed"}

    monkeypatch.setattr(cli, "ActiveProductionRunner", FakeRunner)
    assert cli.main(["run", "--config", str(tmp_path / "active.json")]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "completed"
    assert "[active][round=002][stage=solver] done selected=3" in captured.err
