from src.evaluation.ranker import rank_candidates


def test_ranker_top5_sorted() -> None:
    cands = [f"caption {i} with chaos" for i in range(8)]
    out = rank_candidates("x.jpg", "img1", cands)
    assert len(out["top5"]) == 5
    scores = [x["total_score"] for x in out["top5"]]
    assert scores == sorted(scores, reverse=True)
