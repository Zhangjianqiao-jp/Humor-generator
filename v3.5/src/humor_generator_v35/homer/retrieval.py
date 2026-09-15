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


class OfficialQueryFittedTfidfIndex:
    """Retrieval behavior in HOMER official commit d1334f2.

    The released implementation fits a fresh sklearn vectorizer on the single
    repeated-entity/context query, then transforms the complete joke corpus.
    This unusual choice is preserved for the strict Text-HOMER baseline rather
    than silently replacing it with a globally fitted index.
    """

    def __init__(self, documents: Sequence[str]) -> None:
        if not documents:
            raise ValueError("joke corpus is empty")
        self.documents = tuple(str(value) for value in documents)

    def search_context(
        self,
        entity: str,
        *,
        description: str,
        conflicts: str,
        k: int = 5,
    ) -> list[str]:
        if k < 1:
            raise ValueError("k must be positive")
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # pandas str.contains in the official code is regex-based.  Avoid a
        # heavyweight Series while preserving case-insensitive regex search.
        pattern = re.compile(entity, flags=re.I)
        exact = [document for document in self.documents if pattern.search(document)]
        if len(exact) >= k:
            return exact[:k]
        query = f"{entity} {entity} {entity} {entity} {description} {conflicts}"
        vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 3)
        )
        query_vector = vectorizer.fit_transform([query])
        document_vectors = vectorizer.transform(self.documents)
        similarities = cosine_similarity(query_vector, document_vectors).ravel()
        indices = np.argsort(similarities)[::-1]
        result: list[str] = []
        seen: set[str] = set()
        for index in indices:
            document = self.documents[int(index)]
            if document in seen:
                continue
            seen.add(document)
            result.append(document)
            if len(result) == k:
                break
        return result


class OfficialCodeRetrieval:
    """Compatibility implementation of ``imaginator.py`` retrieval.

    This deliberately mirrors the pinned public repository's observable
    behavior instead of reusing the more principled v3.5 ``HomerRetrievalAugmenter``:

    * regex entity matches are returned before semantic search;
    * TF-IDF is fitted on the repeated entity/context query only;
    * noun/verb entities are extracted from the five retrieved jokes;
    * the official relevance, frequency and POS-diversity formulas are used;
    * the root/value/retrieved-entity edge construction follows
      ``build_entity_tree`` (including its duplicate-preserving behavior).

    The official code does not publish a frozen vectorizer or WordNet version,
    so this class records the Python/NLTK/sklearn environment in the caller's
    provenance and should be called ``official-code-compatible``, not a byte
    identical execution of the remote API process.
    """

    def __init__(self, documents: Sequence[str]) -> None:
        if not documents:
            raise ValueError("joke corpus is empty")
        self.documents = tuple(str(value) for value in documents)
        import nltk
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import WordNetLemmatizer

        try:
            self.stop_words = set(stopwords.words("english"))
            wordnet.ensure_loaded()
        except Exception as exc:
            raise RuntimeError("NLTK stopwords and WordNet are required for official-code retrieval") from exc
        self.wordnet = wordnet
        self.lemmatizer = WordNetLemmatizer()
        self.humor_stop_words = self.stop_words.union({
            "said", "says", "tell", "tells", "joke", "jokes", "funny", "laugh",
            "haha", "lol", "rofl", "hehe", "joking", "kidding",
        })
        self._nltk = nltk

    def search_context(
        self,
        entity: str,
        *,
        description: str,
        conflicts: str,
        k: int = 5,
    ) -> list[str]:
        if k < 1:
            raise ValueError("k must be positive")
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            pattern = re.compile(entity, flags=re.I)
        except re.error:
            pattern = re.compile(re.escape(entity), flags=re.I)
        exact = [document for document in self.documents if pattern.search(document)]
        if len(exact) >= k:
            return exact[:k]

        query = f"{entity} {entity} {entity} {entity}"
        if description:
            query += f" {description}"
        if conflicts:
            query += f" {conflicts}"
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english",
            ngram_range=(1, 3),
        )
        query_vector = vectorizer.fit_transform([query])
        document_vectors = vectorizer.transform(self.documents)
        similarities = cosine_similarity(query_vector, document_vectors).flatten()
        # np.argsort(...)[::-1] is intentionally retained from the official
        # implementation, including its tie behavior.
        result: list[str] = []
        seen: set[str] = set()
        for index in np.argsort(similarities)[::-1]:
            document = self.documents[int(index)]
            if document in seen:
                continue
            seen.add(document)
            result.append(document)
            if len(result) == k:
                break
        return result

    def _word_similarity(self, left: str, right: str) -> float:
        left_senses = self.wordnet.synsets(left.lower())
        right_senses = self.wordnet.synsets(right.lower())
        if not left_senses or not right_senses:
            return 0.0
        best = 0.0
        for left_sense in left_senses:
            for right_sense in right_senses:
                try:
                    score = left_sense.wup_similarity(right_sense)
                except Exception:
                    score = None
                if score is not None and score > best:
                    best = score
        return min(1.0, max(0.0, best))

    def _relation_similarity(self, left: str, right: str) -> float:
        left_senses = self.wordnet.synsets(left.lower())
        right_senses = self.wordnet.synsets(right.lower())
        if not left_senses or not right_senses:
            return 0.0
        best = 0.0

        def related(sense: object) -> set[str]:
            values: set[str] = set()
            for lemma in sense.lemmas():
                values.add(lemma.name().lower())
            for parent in sense.hypernyms() + sense.hyponyms() + sense.part_meronyms() + sense.part_holonyms():
                for lemma in parent.lemmas():
                    values.add(lemma.name().lower())
            return values

        for left_sense in left_senses:
            for right_sense in right_senses:
                left_values, right_values = related(left_sense), related(right_sense)
                union = left_values | right_values
                if union:
                    best = max(best, len(left_values & right_values) / len(union))
        return min(1.0, max(0.0, best))

    def _entity_relevance(self, candidate: str, target: str) -> float:
        if candidate == target.lower():
            return 0.0
        ta = self._word_similarity(candidate, target)
        so = max(1.0 - self._relation_similarity(candidate, target), 0.0)
        return ta + (ta * math.exp(-ta)) * so

    @staticmethod
    def _frequency_score(token_frequency: int, total_tokens: int, entity_joke_count: int, total_jokes: int) -> float:
        if token_frequency <= 0 or total_tokens <= 0 or total_jokes <= 0:
            return 0.0
        tf = float(token_frequency) / float(total_tokens)
        idf = math.log((float(total_jokes) + 1.0) / (float(entity_joke_count) + 1.0)) + 1.0
        return min(1.0, max(0.0, tf * idf))

    @staticmethod
    def _pos_diversity(info: Sequence[dict[str, str]]) -> float:
        # The official code normalizes over eight noun/verb POS tags.
        return float(len({item["pos"] for item in info})) / 8.0 if info else 0.0

    def extract_entities_from_jokes(
        self,
        jokes: Sequence[str],
        target_entity: str,
        entity_keys: Sequence[str],
    ) -> list[str]:
        del entity_keys  # accepted because the pinned implementation does not use it in scoring
        from nltk import pos_tag, word_tokenize

        entity_categories = {"nouns": ["NN", "NNS", "NNP", "NNPS"], "verbs": ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]}
        relevant: dict[str, list[dict[str, str]]] = {}
        entity_joke_ids: dict[str, set[int]] = {}
        total_tokens = 0
        for joke_idx, joke in enumerate(jokes):
            if not isinstance(joke, str) or not joke.strip():
                continue
            try:
                tagged = pos_tag(word_tokenize(joke.lower()))
            except Exception:
                continue
            for token, pos in tagged:
                if pos not in entity_categories["nouns"] + entity_categories["verbs"]:
                    continue
                clean = re.sub(r"[^\w\s]", "", token).strip()
                if len(clean) <= 2 or clean in self.humor_stop_words:
                    continue
                lemma = self.lemmatizer.lemmatize(clean)
                total_tokens += 1
                score = self._entity_relevance(lemma, target_entity)
                if score <= 0:
                    continue
                relevant.setdefault(lemma, []).append({"pos": pos, "score": score})
                entity_joke_ids.setdefault(lemma, set()).add(joke_idx)

        ranked: list[tuple[str, float]] = []
        total_jokes = len(jokes)
        for entity, info in relevant.items():
            avg_relevance = sum(float(item["score"]) for item in info) / len(info)
            frequency = self._frequency_score(
                len(info), total_tokens or 1, len(entity_joke_ids.get(entity, set())), total_jokes
            )
            score = avg_relevance + frequency + self._pos_diversity(info)
            ranked.append((entity, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [entity for entity, _ in ranked[:5]]

    def _build_entity_tree(
        self,
        entity_key: str,
        entity_values: Sequence[str],
        extracted: Sequence[tuple[str, str]],
    ) -> list[list[str]]:
        if not entity_values:
            return []
        edges = [[entity_key, entity_values[0]]]
        edges.extend([[entity_key, value] for value in entity_values])
        edges.extend([[source, entity] for source, entity in extracted])
        return edges

    def retrieve(
        self,
        summary: Mapping[str, Any],
        *,
        description: str,
        conflicts: str,
    ) -> dict[str, list[list[str]]]:
        output: dict[str, list[list[str]]] = {}
        for entity_key, raw_values in summary.items():
            if not isinstance(entity_key, str) or not isinstance(raw_values, list):
                raise ValueError("official summary must map string entities to lists")
            if not raw_values or not all(isinstance(value, str) for value in raw_values):
                raise ValueError("official summary values must be non-empty lists of strings")
            values = [str(value) for value in raw_values]
            extracted: list[tuple[str, str]] = []
            for source in [entity_key, *values]:
                jokes = self.search_context(
                    source, description=description, conflicts=conflicts, k=5
                )
                for candidate in self.extract_entities_from_jokes(jokes, source, list(summary)):
                    extracted.append((source, candidate))
            edges = self._build_entity_tree(entity_key, values, extracted)
            if not edges:
                raise ValueError(f"official retrieval produced no edges for {entity_key!r}")
            output[entity_key] = edges
        if not output:
            raise ValueError("official retrieval produced no entity trees")
        return output


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
                search_context = getattr(self.index, "search_context", None)
                if search_context is None:
                    jokes = tuple(self.index.search(query, k=self.config.top_k))
                else:
                    jokes = tuple(search_context(
                        node,
                        description=plan.description,
                        conflicts=conflict_context,
                        k=self.config.top_k,
                    ))
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
