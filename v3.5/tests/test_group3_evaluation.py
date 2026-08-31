from humor_generator_v35.evaluation.formal import aggregate_group3, build_group3_packets


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
    assert all(
        hidden["seeds_A"] != [11, 22, 33] or hidden["seeds_B"] != [11, 22, 33]
        for hidden in mapping
    )


def test_description_is_only_an_explicit_fallback():
    packets, _ = build_group3_packets(generations(), include_standard_description=True)
    assert all(packet["standard_description"] == "a test cartoon" for packet in packets)


def test_group3_rejects_missing_seed():
    rows = generations()
    rows.pop()
    try:
        build_group3_packets(rows)
    except ValueError as exc:
        assert "exactly 3 unique seeds" in str(exc)
    else:
        raise AssertionError("missing seed was accepted")


def test_aggregation_is_image_clustered_and_seed_aware():
    packets, mapping = build_group3_packets(generations(), seed=7)
    decisions = {}
    for packet, hidden in zip(packets, mapping):
        challenger_side = "A" if hidden["condition_A"] != "text_homer" else "B"
        decisions[packet["blind_id"]] = {
            "overall": challenger_side,
            "best_pick": challenger_side,
            "absolute_A": "weak",
            "absolute_B": "good",
            "best_A_index": 1,
            "best_B_index": 1,
            "candidate_labels_A": ["bad", "weak", "good"],
            "candidate_labels_B": ["weak", "good", "good"],
        }
    report = aggregate_group3(mapping, [{
        "rater_id": "judge",
        "judge_metadata": {
            "provider": "test", "model": "test", "version_or_date": "1",
            "temperature": 0, "prompt_sha256": "0" * 64,
        },
        "decisions": decisions,
    }])
    assert len(report["comparisons"]) == 2
    assert all(item["image_clusters"] == 2 for item in report["comparisons"])
    assert all(
        item["relative_metrics"]["overall"]["rater_averaged_win_rate_ties_half"] == 1.0
        for item in report["comparisons"]
    )
    assert all(
        item["absolute_quality"][item["challenger"]]["generation_seed_sample_variance"] is not None
        for item in report["comparisons"]
    )


def test_explicit_comparisons_do_not_create_unplanned_pairs():
    packets, mapping = build_group3_packets(
        generations(), comparisons=[("statebridge", "learned_latent")], seed=9
    )
    assert len(packets) == 2
    assert {(item["reference"], item["challenger"]) for item in mapping} == {
        ("statebridge", "learned_latent")
    }


def test_paper_aligned_group10_preserves_size_and_randomized_seed_map():
    rows = []
    for condition in ("full_plan_text", "typed_learned_latent"):
        for seed in range(10):
            rows.append({
                "receiver": "sft", "condition": condition, "cluster_id": "nycc_9",
                "image": "9.jpg", "generation_seed": seed,
                "caption": f"{condition}-{seed}",
            })
    packets, mapping = build_group3_packets(
        rows, comparisons=[("full_plan_text", "typed_learned_latent")],
        group_size=10, seed=5,
    )
    assert len(packets) == len(mapping) == 1
    assert len(packets[0]["group_A"]) == len(packets[0]["group_B"]) == 10
    assert mapping[0]["group_size"] == 10
    assert mapping[0]["seeds_A"] != list(range(10))


def test_group_size_family_and_mirror_orientation_have_distinct_ids():
    rows = []
    for condition in ("full_plan_text", "typed_learned_latent"):
        for seed in range(10):
            rows.append({
                "receiver": "sft", "condition": condition, "cluster_id": "nycc_10",
                "image": "10.jpg", "generation_seed": seed, "caption": f"{condition}-{seed}",
            })
    _, primary = build_group3_packets(
        rows, comparisons=[("full_plan_text", "typed_learned_latent")],
        group_size=10, comparison_family="primary", mirror_sides=True,
    )
    _, secondary = build_group3_packets(
        rows, comparisons=[("full_plan_text", "typed_learned_latent")],
        group_size=10, comparison_family="secondary", mirror_sides=True,
    )
    assert len(primary) == 2
    assert primary[0]["condition_A"] == primary[1]["condition_B"]
    assert primary[0]["mirror_pair_id"] == primary[1]["mirror_pair_id"]
    assert len({item["blind_id"] for item in primary + secondary}) == 4


def test_five_shot_calibration_is_validated_and_hashed() -> None:
    examples = [
        {"image": f"cal-{index}.jpg", "caption_A": "A", "caption_B": "B", "answer": "A"}
        for index in range(5)
    ]
    packets, mapping = build_group3_packets(
        generations(), comparisons=[("text_homer", "learned_latent")],
        calibration_examples=examples,
    )
    assert packets[0]["calibration_examples"] == examples
    assert len(mapping[0]["calibration_sha256"]) == 64
