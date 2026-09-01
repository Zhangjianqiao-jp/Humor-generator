from humor_generator_v3.evaluation.formal import aggregate_group3, build_group3_packets


def generations():
    rows = []
    for condition in ("text_homer", "statebridge", "learned_latent"):
        for cluster in ("nycc_1", "nycc_2"):
            for seed in (11, 22, 33):
                rows.append({
                    "receiver": "base",
                    "condition": condition,
                    "cluster_id": cluster,
                    "image": f"{cluster}.jpg",
                    "standard_description": "a test cartoon",
                    "generation_seed": seed,
                    "caption": f"{condition}-{cluster}-{seed}",
                })
    return rows


def test_group3_is_complete_and_blinded():
    packets, mapping = build_group3_packets(generations(), seed=7)
    assert len(packets) == 4
    assert len(mapping) == 4
    assert all("condition" not in str(packet) for packet in packets)
    assert all("standard_description" not in packet for packet in packets)
    assert all(len(packet["group_A"]) == len(packet["group_B"]) == 3 for packet in packets)


def test_description_is_only_an_explicit_fallback():
    packets, _ = build_group3_packets(generations(), include_standard_description=True)
    assert all(packet["standard_description"] == "a test cartoon" for packet in packets)


def test_group3_rejects_missing_seed():
    rows = generations()
    rows.pop()
    try:
        build_group3_packets(rows)
    except ValueError as exc:
        assert "exactly three unique seeds" in str(exc)
    else:
        raise AssertionError("missing seed was accepted")


def test_aggregation_is_image_clustered_and_seed_aware():
    packets, mapping = build_group3_packets(generations(), seed=7)
    decisions = {}
    for packet, hidden in zip(packets, mapping):
        challenger_side = "A" if hidden["condition_A"] != "text_homer" else "B"
        decisions[packet["blind_id"]] = {
            "overall": challenger_side,
            "absolute_A": "weak",
            "absolute_B": "good",
            "candidate_labels_A": ["bad", "weak", "good"],
            "candidate_labels_B": ["weak", "good", "good"],
        }
    report = aggregate_group3(mapping, [{"rater_id": "judge", "decisions": decisions}])
    assert len(report["comparisons"]) == 2
    assert all(item["image_clusters"] == 2 for item in report["comparisons"])
    assert all(item["rater_averaged_win_rate_ties_half"] == 1.0 for item in report["comparisons"])
    assert all(
        item["absolute_quality"][item["challenger"]]["seed_variance_available"]
        for item in report["comparisons"]
    )
