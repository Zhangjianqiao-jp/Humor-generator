import json

from scripts.build_dpo_scaling_dashboard import build


def test_dashboard_builds_without_optional_plotting_dependencies(tmp_path) -> None:
    run_dir, output_dir = tmp_path / "run", tmp_path / "dashboard"
    run_dir.mkdir()
    (run_dir / "run_status.json").write_text(
        json.dumps({"state": "running", "step": 10, "total_steps": 100}), encoding="utf-8"
    )
    row = {
        "split": "validation",
        "step": 10,
        "eval_loss": 0.69,
        "eval_image_mean_loss": 0.69,
        "eval_image_mean_reward_accuracy": 0.6,
        "eval_image_mean_policy_accuracy": 0.6,
        "eval_image_mean_reward_margin": 0.01,
        "eval_image_mean_chosen_logp_per_token": -2.9,
        "eval_image_mean_rejected_logp_per_token": -3.1,
    }
    (run_dir / "train_metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert build(run_dir, output_dir) == "running"
    assert (output_dir / "index.html").is_file()
    assert (output_dir / "monitor.png").is_file()
