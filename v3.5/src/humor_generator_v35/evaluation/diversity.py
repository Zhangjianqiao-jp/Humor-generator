"""Quality-aware lexical and semantic diversity for caption candidate sets."""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN = re.compile(r"\b[\w']+\b", re.UNICODE)
OFFICIAL_EAD_VOCAB_SIZE = 32_000


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def _distinct(captions: list[str], n: int) -> float:
    grams = []
    for caption in captions:
        tokens = _tokens(caption)
        grams.extend(tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1))
    return len(set(grams)) / len(grams) if grams else 0.0


def _semantic(captions: list[str]) -> tuple[float, float]:
    if len(captions) < 2:
        return 0.0, 1.0
    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(captions)
    except ValueError:
        return 0.0, 1.0
    kernel = cosine_similarity(matrix)
    upper = kernel[np.triu_indices(len(captions), k=1)]
    distance = float(np.mean(1.0 - upper))
    # Vendi score: exp(entropy) of normalized eigenvalues of a PSD similarity kernel.
    eigen = np.linalg.eigvalsh((kernel + kernel.T) * 0.5)
    eigen = np.clip(eigen, 0.0, None)
    probabilities = eigen / max(float(eigen.sum()), 1e-12)
    probabilities = probabilities[probabilities > 1e-12]
    vendi = float(math.exp(-float(np.sum(probabilities * np.log(probabilities)))))
    return distance, vendi


def _self_bleu(captions: list[str]) -> float | None:
    if len(captions) < 2:
        return None
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    tokenized = [_tokens(value) for value in captions]
    scores = []
    for index, hypothesis in enumerate(tokenized):
        references = [value for offset, value in enumerate(tokenized) if offset != index]
        scores.append(sentence_bleu(
            references, hypothesis, weights=(0.5, 0.5),
            smoothing_function=SmoothingFunction().method1,
        ))
    return sum(scores) / len(scores)


def official_expectation_adjusted_distinct(
    captions: list[str], *, n_min: int = 1, n_max: int = 5,
    vocabulary_size: int = OFFICIAL_EAD_VOCAB_SIZE,
) -> float:
    """Reproduce the NeurIPS 2024 Humor-in-AI EAD implementation."""
    if n_min < 1 or n_max < n_min or vocabulary_size < 2:
        raise ValueError("invalid EAD configuration")
    by_order = []
    for n in range(n_min, n_max + 1):
        ngrams = []
        for caption in captions:
            words = [part for part in str(caption).replace(".", "").replace("\n", "").split(" ") if part]
            ngrams.extend(tuple(words[index:index + n]) for index in range(len(words) - n + 1))
        count, unique = len(ngrams), len(set(ngrams))
        denominator = vocabulary_size * (1 - ((vocabulary_size - 1) / vocabulary_size) ** count)
        by_order.append(unique / denominator if denominator else 0.0)
    return sum(by_order) / len(by_order)


def official_sbert_diversity(captions: list[str], model: Any) -> float:
    """Paper-compatible 1 - mean cosine similarity, including the diagonal."""
    if not captions:
        return 0.0
    embeddings = np.asarray(model.encode(captions, normalize_embeddings=True))
    return float(1.0 - np.mean(embeddings @ embeddings.T))


def candidate_set_metrics(
    captions: list[str], *, sbert_model: Any | None = None,
) -> dict[str, float | int | None]:
    normalized = [" ".join(value.split()) for value in captions if value.strip()]
    semantic_distance, vendi = _semantic(normalized)
    return {
        "candidates": len(normalized),
        "unique_caption_rate": len({value.casefold() for value in normalized}) / len(normalized) if normalized else 0.0,
        "distinct_1": _distinct(normalized, 1),
        "distinct_2": _distinct(normalized, 2),
        "self_bleu_2": _self_bleu(normalized),
        "mean_pairwise_tfidf_distance": semantic_distance,
        "vendi_score_tfidf": vendi,
        "official_average_ead_n1_n5": official_expectation_adjusted_distinct(normalized),
        "official_sbert_all_mpnet_base_v2_diversity": (
            official_sbert_diversity(normalized, sbert_model) if sbert_model is not None else None
        ),
    }


def summarize_diversity(
    rows: Iterable[dict[str, Any]], *, min_candidates: int = 2,
    sbert_model: Any | None = None,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        key = (str(row["receiver"]), str(row["condition"]), str(row["cluster_id"]))
        grouped.setdefault(key, []).append(row)
    records = []
    for (receiver, condition, cluster), values in sorted(grouped.items()):
        if len(values) < min_candidates:
            raise ValueError(f"{receiver}/{condition}/{cluster} has only {len(values)} candidates")
        captions = [str(item["caption"]) for item in values]
        record = {
            "receiver": receiver, "condition": condition, "cluster_id": cluster,
            **candidate_set_metrics(captions, sbert_model=sbert_model),
        }
        quality = [item for item in values if item.get("absolute_label") in {"good", "weak", "bad"}]
        good = [str(item["caption"]) for item in quality if item["absolute_label"] == "good"]
        record["good_only"] = (
            candidate_set_metrics(good, sbert_model=sbert_model) if len(good) >= 2 else None
        )
        angles = [str(item["angle_label"]) for item in values if str(item.get("angle_label") or "").strip()]
        record["human_angle_coverage"] = len(set(angles)) if angles else None
        records.append(record)
    by_system: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        by_system.setdefault((record["receiver"], record["condition"]), []).append(record)
    summaries = []
    numeric = (
        "unique_caption_rate", "distinct_1", "distinct_2", "self_bleu_2",
        "mean_pairwise_tfidf_distance", "vendi_score_tfidf",
        "official_average_ead_n1_n5", "official_sbert_all_mpnet_base_v2_diversity",
    )
    for (receiver, condition), values in sorted(by_system.items()):
        summary: dict[str, Any] = {"receiver": receiver, "condition": condition, "image_clusters": len(values)}
        for name in numeric:
            present = [float(item[name]) for item in values if item[name] is not None]
            summary[f"mean_{name}"] = sum(present) / len(present) if present else None
        summaries.append(summary)
    return {
        "schema_version": 1,
        "statistical_unit": "image_cluster",
        "warning": (
            "EAD and all-mpnet-base-v2 follow Humor in AI; lexical/TF-IDF/Vendi are "
            "descriptive supplements, and human angle coverage remains a semantic-angle endpoint"
        ),
        "per_cluster": records,
        "system_summary": summaries,
    }
