from src.preference.cluster_metrics import summarize_image_clusters


def test_cluster_summary_weights_images_equally() -> None:
    rows = [
        {"image_id": "a", "loss": 0.0},
        {"image_id": "a", "loss": 0.0},
        {"image_id": "a", "loss": 0.0},
        {"image_id": "b", "loss": 1.0},
    ]
    report = summarize_image_clusters(rows, ["loss"], bootstrap_samples=100, seed=7)
    assert report["pair_count"] == 4
    assert report["image_count"] == 2
    assert report["metrics"]["loss"]["pair_mean"] == 0.25
    assert report["metrics"]["loss"]["image_mean"] == 0.5
    low, high = report["metrics"]["loss"]["image_cluster_bootstrap_95ci"]
    assert 0.0 <= low <= high <= 1.0
