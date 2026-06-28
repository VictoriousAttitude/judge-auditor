from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from judge_auditor.cli import (
    _parse_labeled_path,
    app,
    build_backend,
    load_examples,
    load_judge_config,
)

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


def test_run_with_probe_flags_collects_probe_records(tmp_path):
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    saved = tmp_path / "judgments.json"
    result = runner.invoke(
        app,
        [
            "run", "-c", str(cfg), "-e", str(exs), "-b", "mock", "-k", "4",
            "--probe-sycophancy", "--probe-anchoring", "--save-judgments", str(saved),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(saved.read_text(encoding="utf-8"))
    probes = {r["probe"] for r in data["records"]}
    assert {"neutral", "sycophancy_up", "sycophancy_down", "anchor_up", "anchor_down"} == probes


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


# --- error paths ----------------------------------------------------------------


def test_run_missing_config_file_is_usage_error(tmp_path):
    exs = scalar_examples(tmp_path)
    result = runner.invoke(
        app, ["run", "-c", str(tmp_path / "nope.toml"), "-e", str(exs), "-b", "mock"]
    )
    assert result.exit_code == 2  # click validates exists=True before the body runs


def test_run_missing_examples_file_is_usage_error(tmp_path):
    cfg = scalar_config(tmp_path)
    result = runner.invoke(
        app, ["run", "-c", str(cfg), "-e", str(tmp_path / "nope.jsonl"), "-b", "mock"]
    )
    assert result.exit_code == 2


def test_run_unknown_format_is_usage_error(tmp_path):
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    result = runner.invoke(
        app,
        ["run", "-c", str(cfg), "-e", str(exs), "-b", "mock", "-k", "5", "-f", "bogus"],
    )
    assert result.exit_code == 2
    assert "unknown format" in result.output


def test_run_unknown_backend_is_usage_error(tmp_path):
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    result = runner.invoke(
        app, ["run", "-c", str(cfg), "-e", str(exs), "-b", "bogus", "-k", "5"]
    )
    assert result.exit_code == 2
    assert "unknown backend" in result.output


def test_run_malformed_toml_config_fails(tmp_path):
    bad = write(tmp_path / "judge.toml", "model = \nmode = ")  # not valid TOML
    exs = scalar_examples(tmp_path)
    result = runner.invoke(app, ["run", "-c", str(bad), "-e", str(exs), "-b", "mock"])
    assert result.exit_code != 0


def test_run_example_missing_required_field_fails(tmp_path):
    cfg = scalar_config(tmp_path)
    # Row without the required "prompt" key.
    bad = write(tmp_path / "ex.jsonl", json.dumps({"id": "ex0", "response_a": "x"}))
    result = runner.invoke(app, ["run", "-c", str(cfg), "-e", str(bad), "-b", "mock"])
    assert result.exit_code != 0
    assert isinstance(result.exception, KeyError)


# --- compare command ------------------------------------------------------------


def _save_scalar_judgments(tmp_path: Path, name: str) -> Path:
    """Run the mock scalar judge once and persist its judgments to ``name``."""
    cfg, exs = scalar_config(tmp_path), scalar_examples(tmp_path)
    saved = tmp_path / name
    result = runner.invoke(
        app,
        ["run", "-c", str(cfg), "-e", str(exs), "-b", "mock", "-k", "5",
         "--save-judgments", str(saved)],
    )
    assert result.exit_code == 0, result.output
    return saved


def test_parse_labeled_path_explicit_label(tmp_path):
    p = write(tmp_path / "j.json", "{}")
    label, path = _parse_labeled_path(f"GPT-4o={p}")
    assert label == "GPT-4o" and path == p


def test_parse_labeled_path_defaults_to_stem(tmp_path):
    p = write(tmp_path / "claude.json", "{}")
    label, path = _parse_labeled_path(str(p))
    assert label == "claude" and path == p


def test_parse_labeled_path_missing_raises(tmp_path):
    with pytest.raises(Exception, match="not found"):
        _parse_labeled_path(f"x={tmp_path / 'nope.json'}")


def test_compare_renders_table_with_both_judges(tmp_path):
    a = _save_scalar_judgments(tmp_path, "a.json")
    b = _save_scalar_judgments(tmp_path, "b.json")
    exs = scalar_examples(tmp_path)
    result = runner.invoke(
        app,
        ["compare", "-e", str(exs), "-j", f"Alpha={a}", "-j", f"Beta={b}"],
    )
    assert result.exit_code == 0, result.output
    assert "| Judge |" in result.output
    assert "| --- |" in result.output
    assert "Alpha" in result.output and "Beta" in result.output


def test_compare_label_defaults_to_filename(tmp_path):
    saved = _save_scalar_judgments(tmp_path, "myjudge.json")
    exs = scalar_examples(tmp_path)
    result = runner.invoke(app, ["compare", "-e", str(exs), "-j", str(saved)])
    assert result.exit_code == 0, result.output
    assert "myjudge" in result.output


def test_compare_writes_to_file(tmp_path):
    saved = _save_scalar_judgments(tmp_path, "j.json")
    exs = scalar_examples(tmp_path)
    out = tmp_path / "table.md"
    result = runner.invoke(
        app, ["compare", "-e", str(exs), "-j", f"J={saved}", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("| Judge |")


def test_compare_missing_judgments_file_is_usage_error(tmp_path):
    exs = scalar_examples(tmp_path)
    result = runner.invoke(
        app, ["compare", "-e", str(exs), "-j", f"J={tmp_path / 'nope.json'}"]
    )
    assert result.exit_code == 2
    assert "not found" in result.output


# --- diff command ---------------------------------------------------------------


def test_diff_terminal_renders(tmp_path):
    a = _save_scalar_judgments(tmp_path, "a.json")
    b = _save_scalar_judgments(tmp_path, "b.json")
    exs = scalar_examples(tmp_path)
    result = runner.invoke(
        app, ["diff", "-e", str(exs), "-j", f"V1={a}", "-j", f"V2={b}"]
    )
    assert result.exit_code == 0, result.output
    assert "REPORT DIFF: V1 -> V2" in result.output


def test_diff_markdown_to_file(tmp_path):
    a = _save_scalar_judgments(tmp_path, "a.json")
    b = _save_scalar_judgments(tmp_path, "b.json")
    exs = scalar_examples(tmp_path)
    out = tmp_path / "diff.md"
    result = runner.invoke(
        app,
        ["diff", "-e", str(exs), "-j", f"V1={a}", "-j", f"V2={b}",
         "-f", "markdown", "-o", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("### Report diff: V1 -> V2")


def test_diff_requires_exactly_two(tmp_path):
    a = _save_scalar_judgments(tmp_path, "a.json")
    exs = scalar_examples(tmp_path)
    result = runner.invoke(app, ["diff", "-e", str(exs), "-j", f"V1={a}"])
    assert result.exit_code == 2
    assert "exactly two" in result.output


def test_diff_unknown_format_is_usage_error(tmp_path):
    a = _save_scalar_judgments(tmp_path, "a.json")
    b = _save_scalar_judgments(tmp_path, "b.json")
    exs = scalar_examples(tmp_path)
    result = runner.invoke(
        app, ["diff", "-e", str(exs), "-j", f"V1={a}", "-j", f"V2={b}", "-f", "bogus"]
    )
    assert result.exit_code == 2
    assert "unknown format" in result.output
