import os

import pandas as pd

from src.report import _format_seconds, generate_report


def test_format_seconds_handles_missing_p95():
    assert _format_seconds(None) == "n/a"
    assert _format_seconds(float("nan")) == "n/a"
    assert _format_seconds(pd.NA) == "n/a"
    assert _format_seconds(0.12345) == "0.1235s"


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_report_ignores_oom_rows(tmp_path, capsys):
    csv_path = tmp_path / "results.csv"
    _write_csv(
        csv_path,
        [
            {
                "batch_size": 1,
                "tokens_per_second": 10.0,
                "latency_seconds": 0.20,
                "p50_latency_seconds": 0.20,
                "p95_latency_seconds": None,
                "status": "ok",
            },
            {
                "batch_size": 8,
                "tokens_per_second": None,
                "latency_seconds": None,
                "p50_latency_seconds": None,
                "p95_latency_seconds": None,
                "status": "oom",
            },
            {
                "batch_size": 2,
                "tokens_per_second": 18.0,
                "latency_seconds": 0.15,
                "p50_latency_seconds": 0.15,
                "p95_latency_seconds": None,
                "status": "ok",
            },
        ],
    )

    generate_report(str(csv_path))
    output = capsys.readouterr().out

    assert "Batch 2: 18.00 tokens/sec" in output
    assert "Best throughput: Batch 2" in output
    assert "Throughput improvement: Batch 1 → Batch 2: 1.80x" in output


def test_generate_report_legacy_csv_without_status(tmp_path, capsys):
    csv_path = tmp_path / "legacy.csv"
    _write_csv(
        csv_path,
        [
            {
                "batch_size": 1,
                "tokens_per_second": 12.5,
                "latency_seconds": 0.08,
            }
        ],
    )

    generate_report(str(csv_path))
    output = capsys.readouterr().out

    assert "Best throughput: Batch 1" in output
    assert "Best latency: Batch 1" in output


def test_generate_report_all_oom(tmp_path, capsys):
    csv_path = tmp_path / "oom.csv"
    _write_csv(
        csv_path,
        [
            {
                "batch_size": 16,
                "tokens_per_second": None,
                "status": "oom",
            }
        ],
    )

    generate_report(str(csv_path))
    output = capsys.readouterr().out

    assert "No successful benchmark rows to summarize." in output
