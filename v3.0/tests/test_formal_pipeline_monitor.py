from pathlib import Path

import scripts.formal_pipeline_monitor as monitor


def test_active_job_parser_ignores_header():
    output = "JOB_ID JOB_NAME ST\n6645209 cache RUN\nnot-a-job x y\n"
    assert monitor.active_job_ids(output) == {"6645209"}


def test_cache_evidence_requires_exact_unique_cluster_set(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    dataset.mkdir()
    cache.mkdir()
    (dataset / "train.jsonl").write_text('{"cluster_id":"a"}\n')
    (dataset / "validation.jsonl").write_text('{"cluster_id":"b"}\n')
    (cache / "index.jsonl").write_text(
        '{"cluster_id":"a"}\n{"cluster_id":"b"}\n'
    )
    (cache / "failures.json").write_text("[]\n")
    monkeypatch.setattr(monitor, "DATASET", dataset)
    monkeypatch.setattr(monitor, "CACHE", cache)
    evidence = monitor.cache_evidence()
    assert evidence["complete"] is True
    (cache / "index.jsonl").write_text(
        '{"cluster_id":"a"}\n{"cluster_id":"a"}\n{"cluster_id":"b"}\n'
    )
    evidence = monitor.cache_evidence()
    assert evidence["complete"] is False
    assert evidence["duplicates"] == 1


def test_parse_submission_requires_real_job_id():
    import subprocess
    result = subprocess.CompletedProcess([], 0, "[INFO] PJM Job 12345 submitted.\n", "")
    assert monitor.parse_submitted_job(result) == "12345"
