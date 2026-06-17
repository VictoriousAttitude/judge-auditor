from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judge_auditor.cli import app, build_backend, load_examples, load_judge_config

runner = CliRunner()


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def scalar_config(path: Path) -> Path:
    return write(
        path / "judge.toml",
        'model = "mock-judge"\n'
        'mode = "scalar"\n'
        'prompt_template = "Q: {prompt}\\nA: {response}\\nScore 1-10."\n'
        "score_min = 1.0\nscore_max = 10.0\n",
    )


def pairwise_config(path: Path) -> Path:
    return write(
        path / "judge.json",
        json.dumps(
            {
                "model": "mock-judge",
                "mode": "pairwise",
                "prompt_template": "Q: {prompt}\nA: {response_a}\nB: {response_b}\nWhich?",
            }
        ),
    )


def scalar_examples(path: Path) -> Path:
    rows = [{"id": f"ex{i}", "prompt": f"q{i}", "response_a": f"answer {i} text"} for i in range(6)]
    return write(path / "ex.jsonl", "\n".join(json.dumps(r) for r in rows))


def pairwise_examples(path: Path) -> Path:
    rows = [
        {"id": f"ex{i}", "prompt": f"q{i}", "response_a": f"a{i}", "response_b": f"b{i}"}
        for i in range(6)
    ]
    return write(path / "ex.json", json.dumps(rows))


# --- loaders --------------------------------------------------------------------


def test_load_examples_jsonl(tmp_path):
    exs = load_examples(scalar_examples(tmp_path))
    assert len(exs) == 6
    assert exs[0].id == "ex0" and exs[0].response_b is None


def test_load_examples_json_array(tmp_path):
    exs = load_examples(pairwise_examples(tmp_path))
    assert len(exs) == 6
    assert exs[0].response_b == "b0"


def test_load_judge_config_toml(tmp_path):
    cfg = load_judge_config(scalar_config(tmp_path))
    assert cfg.model == "mock-judge"
    assert cfg.mode.value == "scalar"


def test_load_judge_config_json(tmp_path):
    cfg = load_judge_config(pairwise_config(tmp_path))
    assert cfg.mode.value == "pairwise"


def test_build_backend_unknown_raises():
    with pytest.raises(Exception, match="unknown backend"):
        build_backend("bogus", None)


# --- run command (mock backend, no network) -------------------------------------


def test_run_scalar_terminal(tmp_path):
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    result = runner.invoke(
        app, ["run", "-c", str(cfg), "-e", str(exs), "--backend", "mock", "-k", "6"]
    )
    assert result.exit_code == 0, result.output
    assert "JUDGE RELIABILITY" in result.output


def test_run_scalar_json_is_valid(tmp_path):
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    result = runner.invoke(
        app, ["run", "-c", str(cfg), "-e", str(exs), "-b", "mock", "-k", "5", "-f", "json"]
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["mode"] == "scalar"
    assert "recommendations" in parsed


def test_run_pairwise_html_to_file(tmp_path):
    cfg, exs = pairwise_config(tmp_path), pairwise_examples(tmp_path)
    out = tmp_path / "report.html"
    args = ["run", "-c", str(cfg), "-e", str(exs), "-b", "mock", "-k", "6", "-f", "html"]
    result = runner.invoke(app, [*args, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_run_save_judgments_then_report_roundtrip(tmp_path):
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    saved = tmp_path / "judgments.json"
    args = ["run", "-c", str(cfg), "-e", str(exs), "-b", "mock", "-k", "5"]
    r1 = runner.invoke(app, [*args, "--save-judgments", str(saved)])
    assert r1.exit_code == 0, r1.output
    assert saved.exists()

    r2 = runner.invoke(
        app, ["report", "-j", str(saved), "-e", str(exs), "-f", "json"]
    )
    assert r2.exit_code == 0, r2.output
    assert json.loads(r2.output)["mode"] == "scalar"


def test_run_cost_prompt_abort(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")  # lets the backend construct
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    result = runner.invoke(
        app,
        ["run", "-c", str(cfg), "-e", str(exs), "-b", "openai", "-k", "3"],
        input="n\n",
    )
    assert result.exit_code == 1
    assert "Aborted" in result.output
