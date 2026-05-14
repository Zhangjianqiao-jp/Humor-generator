from __future__ import annotations

from src.evaluation.judge import HeuristicJudge

WEIGHTS = {"IR": 0.25, "HU": 0.30, "SP": 0.15, "RA": 0.10, "CR": 0.10, "SA": 0.10}


def score_total(scores: dict[str, float]) -> float:
    return sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)


def rank_candidates(image: str, image_id: str, candidates: list[str], judge: HeuristicJudge | None = None) -> dict:
    judge = judge or HeuristicJudge()
    ranked = []
    for caption in candidates:
        scores = judge.score_caption(caption)
        ranked.append({"caption": caption, "scores": scores, "total_score": score_total(scores)})
    ranked = sorted(ranked, key=lambda x: x["total_score"], reverse=True)
    return {"image": image, "image_id": image_id, "top5": ranked[:5], "all_ranked": ranked}
