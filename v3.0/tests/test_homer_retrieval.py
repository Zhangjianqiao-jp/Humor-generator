from __future__ import annotations

from dataclasses import dataclass

from humor_generator_v3.homer.contracts import validate_plan
from humor_generator_v3.homer.curation import JokeRecord, curate_jokes, english_word_overlap
from humor_generator_v3.homer.retrieval import (
    HomerRetrievalAugmenter,
    OfficialQueryFittedTfidfIndex,
    SparseTfidfIndex,
    humor_frequency,
    rank_entities,
)


@dataclass(frozen=True)
class Sense:
    word: str


class FakeGraph:
    def senses(self, word: str):
        return [Sense(word)]

    def wup(self, left: Sense, right: Sense):
        return 1.0 if left.word == right.word else 0.5

    def concept_set(self, sense: Sense):
        return {sense.word, "entity"}

    def pos_inventory(self, word: str):
        return {"n", "v"} if word == "duck" else {"n"}


def test_curation_keeps_longer_near_duplicate_and_rating_gate() -> None:
    rows, report = curate_jokes([
        JokeRecord("A duck walks into a bar", "a", 4),
        JokeRecord("A duck walks into a very old bar", "b", 4),
        JokeRecord("not funny", "c", 2),
    ])
    assert rows == [JokeRecord("A duck walks into a very old bar", "b", 4)]
    assert report.rating_removed == 1 and report.near_duplicates_removed == 1
    assert english_word_overlap("a duck walks into a bar", "a duck walks into a very old bar") > 0.8


def test_sparse_tfidf_and_homer_scores() -> None:
    index = SparseTfidfIndex(["duck walks into a bar", "spreadsheet office meeting"])
    assert index.search("duck bar", k=1) == ["duck walks into a bar"]
    assert humor_frequency("duck", [["duck", "bar"], ["duck", "pond"]]) > 0
    ranked = rank_entities("duck", ["duck bar", "duck pond"], FakeGraph(), delta=2)
    assert ranked[0].entity == "duck"


def test_official_query_fitted_retrieval_preserves_exact_match_priority() -> None:
    index = OfficialQueryFittedTfidfIndex([
        "duck one", "duck two", "duck three", "duck four", "duck five", "office only"
    ])
    assert index.search_context(
        "duck", description="a board meeting", conflicts="animal vs office", k=5
    ) == ["duck one", "duck two", "duck three", "duck four", "duck five"]


def test_retrieval_grows_backbone_nodes_and_enumerates_paths() -> None:
    plan = validate_plan(
        "A duck stands behind a lectern while addressing a formal board meeting.",
        "1. animal behavior vs. office behavior 2. pond life vs. corporate life",
        '{"duck": ["pond", "water", "boat"]}',
        '{"lectern": ["speech", "meeting", "spreadsheet"]}',
    )
    index = SparseTfidfIndex([
        "duck walks into a bar with a spreadsheet",
        "pond meeting runs afowl",
        "office boat budget joke",
        "formal speech quacks up the board",
        "water cooler comedy",
    ])
    augmented = HomerRetrievalAugmenter(index, FakeGraph()).augment(plan)
    assert len(augmented.imagination_trees) == 2
    first = augmented.imagination_trees[0]
    assert len(first.expansions) == 4
    assert all(len(expansion.retrieved_jokes) == 5 for expansion in first.expansions)
    assert first.paths()
    assert all(path[0] == "duck" for path in first.paths())
