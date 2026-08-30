"""HOMER joke retrieval and paper-defined humor-relevance pruning.

The scoring equations are from HOMER Section 2.2.  The paper permits a
statistical or LM embedding but does not disclose the exact implementation;
this module therefore exposes the retrieval backend and records it as a
reproduction variable rather than hiding a choice.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable, Protocol, Sequence

from .contracts import HomerPlan, HumorLeaf, ImaginationTree, NodeExpansion


_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def words(text: str) -> list[str]:
    return [token.casefold() for token in _WORD.findall(text)]


class LexicalGraph(Protocol):
    def senses(self, word: str) -> Sequence[object]: ...
    def wup(self, left: object, right: object) -> float | None: ...
    def concept_set(self, sense: object) -> set[str]: ...
    def pos_inventory(self, word: str) -> set[str]: ...


class NltkWordNetGraph:
    """Lazy WordNet adapter; raises a clear error when the corpus is absent."""

    def __init__(self) -> None:
        try:
            from nltk.corpus import wordnet as wn
            wn.ensure_loaded()
        except Exception as exc:
            raise RuntimeError("NLTK WordNet data is required for paper-faithful HOMER pruning") from exc
        self.wn = wn

    def senses(self, word: str) -> Sequence[object]:
        return self.wn.synsets(word)

    def wup(self, left: object, right: object) -> float | None:
        return left.wup_similarity(right)

    def concept_set(self, sense: object) -> set[str]:
        neighbours = [sense]
        for name in (
            "hypernyms", "hyponyms", "part_meronyms", "substance_meronyms", "member_meronyms",
            "part_holonyms", "substance_holonyms", "member_holonyms",
        ):
            neighbours.extend(getattr(sense, name)())
        result: set[str] = set()
        for node in neighbours:
            result.update(lemma.name().casefold() for lemma in node.lemmas())
        return result

    def pos_inventory(self, word: str) -> set[str]:
        return {sense.pos() for sense in self.wn.synsets(word)}


def relevance_opposition(query: str, candidate: str, graph: LexicalGraph) -> float:
    left, right = graph.senses(query), graph.senses(candidate)
    if not left or not right:
        return 0.0
    tss = max((graph.wup(a, b) or 0.0) for a in left for b in right)
    max_jaccard = 0.0
    for a in left:
        set_a = graph.concept_set(a)
        for b in right:
            set_b = graph.concept_set(b)
            union = set_a | set_b
            overlap = len(set_a & set_b) / len(union) if union else 0.0
            max_jaccard = max(max_jaccard, overlap)
    opposition = 1.0 - max_jaccard
    return tss + (tss * math.exp(-tss)) * opposition


def humor_frequency(candidate: str, retrieved_token_lists: Sequence[Sequence[str]]) -> float:
    if not retrieved_token_lists:
        return 0.0
    token_total = sum(len(tokens) for tokens in retrieved_token_lists)
    if token_total == 0:
        return 0.0
    count = sum(tokens.count(candidate) for tokens in retrieved_token_lists)
    joke_count = sum(candidate in tokens for tokens in retrieved_token_lists)
    return math.sqrt((count / token_total) * (joke_count / len(retrieved_token_lists)))


def pos_diversity(candidate: str, graph: LexicalGraph) -> float:
    # WordNet's inventory is noun, verb, adjective, adverb.
    return len(graph.pos_inventory(candidate)) / 4.0


@dataclass(frozen=True)
class RankedEntity:
    entity: str
    total: float
    relevance: float
    frequency: float
    diversity: float


def rank_entities(
    query_entity: str,
    retrieved_jokes: Sequence[str],
    graph: LexicalGraph,
    *,
    delta: int = 5,
) -> list[RankedEntity]:
    token_lists = [words(joke) for joke in retrieved_jokes]
    candidates = sorted({token for tokens in token_lists for token in tokens})
    ranked: list[RankedEntity] = []
    for candidate in candidates:
        rel = relevance_opposition(query_entity, candidate, graph)
        freq = humor_frequency(candidate, token_lists)
        div = pos_diversity(candidate, graph)
        ranked.append(RankedEntity(candidate, rel + freq + div, rel, freq, div))
    ranked.sort(key=lambda item: (-item.total, item.entity))
    return ranked[:delta]


class SparseTfidfIndex:
    """Deterministic statistical retrieval backend with no external dependency."""

    def __init__(self, documents: Sequence[str]) -> None:
        if not documents:
            raise ValueError("joke corpus is empty")
        self.documents = tuple(documents)
        tokenized = [words(document) for document in documents]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        self.idf = {
            token: math.log((1 + len(documents)) / (1 + count)) + 1.0
            for token, count in document_frequency.items()
        }
        self.vectors = [self._vector(tokens) for tokens in tokenized]

    def _vector(self, tokens: Iterable[str]) -> dict[str, float]:
        counts = Counter(tokens)
        weighted = {token: count * self.idf.get(token, 0.0) for token, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
        return {token: value / norm for token, value in weighted.items() if value}

    def search(self, query: str, *, k: int = 5) -> list[str]:
        if k < 1:
            raise ValueError("k must be positive")
        vector = self._vector(words(query))
        scored = []
        for index, document in enumerate(self.vectors):
            score = sum(value * document.get(token, 0.0) for token, value in vector.items())
            scored.append((score, -index, self.documents[index]))
        scored.sort(reverse=True)
        return [item[2] for item in scored[:k]]


@dataclass(frozen=True)
class HomerRetrievalConfig:
    top_k: int = 5
    delta: int = 5
    merge_mode: str = "exact_path"


class HomerRetrievalAugmenter:
    """Grow every backbone node with pruned joke-token leaves.

    The paper discloses the tree operation but not its exact tokenizer,
    lemmatizer, embedding backend, or fuzzy entity merger.  Those choices stay
    explicit; the default uses the deterministic sparse index and exact-path
    duplicate removal.
    """

    def __init__(
        self,
        index: SparseTfidfIndex,
        graph: LexicalGraph,
        *,
        config: HomerRetrievalConfig = HomerRetrievalConfig(),
    ) -> None:
        if config.top_k < 1 or config.delta < 1:
            raise ValueError("top_k and delta must be positive")
        if config.merge_mode != "exact_path":
            raise ValueError("only the auditable exact_path merge is implemented")
        self.index = index
        self.graph = graph
        self.config = config

    def augment(self, plan: HomerPlan) -> HomerPlan:
        unique_chains = []
        seen: set[tuple[str, ...]] = set()
        for chain in plan.local_chains + plan.global_chains:
            key = tuple(value.casefold() for value in chain.path)
            if key in seen:
                continue
            seen.add(key)
            unique_chains.append(chain)

        conflict_context = plan.conflicts_text()
        trees: list[ImaginationTree] = []
        for chain in unique_chains:
            expansions: list[NodeExpansion] = []
            for node_index, node in enumerate(chain.path):
                query = f"{plan.description} {conflict_context} {node}"
                jokes = tuple(self.index.search(query, k=self.config.top_k))
                ranked = rank_entities(node, jokes, self.graph, delta=self.config.delta)
                leaves = tuple(
                    HumorLeaf(
                        item.entity,
                        item.total,
                        item.relevance,
                        item.frequency,
                        item.diversity,
                    )
                    for item in ranked
                    if item.entity.casefold() != node.casefold()
                )
                expansions.append(NodeExpansion(node_index, node, jokes, leaves))
            trees.append(ImaginationTree(chain, tuple(expansions)))
        if not trees:
            raise RuntimeError("retrieval augmentation produced no imagination trees")
        return HomerPlan(
            description=plan.description,
            conflicts=plan.conflicts,
            local_chains=plan.local_chains,
            global_chains=plan.global_chains,
            imagination_trees=tuple(trees),
        )
