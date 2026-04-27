"""
eval_harness.py — automated retrieval quality evaluation.

For each test case, retrieves top-5 songs and checks whether the
expected genres and moods appear in the top-3 results.

Scoring per test:
  genre_hit  : at least one top-3 result matches an expected genre
  mood_hit   : at least one top-3 result matches an expected mood
  passed     : genre_hit AND mood_hit
  confidence : cosine similarity of the #1 retrieved result (0.0–1.0)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass, field
from typing import List
from tabulate import tabulate
from src.recommender import load_songs
from src.rag import embed_catalog, retrieve


@dataclass
class TestCase:
    description: str
    query: str
    expected_genres: List[str]
    expected_moods: List[str]


TEST_CASES = [
    TestCase(
        description="Late-night study session",
        query="something chill and calm to study to late at night",
        expected_genres=["lofi", "ambient", "jazz"],
        expected_moods=["chill", "focused", "relaxed", "calm"],
    ),
    TestCase(
        description="High-energy workout",
        query="high energy intense workout music to push through the gym",
        expected_genres=["pop", "rock", "electronic", "hip-hop"],
        expected_moods=["intense", "energetic", "euphoric", "angry"],
    ),
    TestCase(
        description="Sad indie folk",
        query="sad slow indie folk acoustic songs for a rainy day",
        expected_genres=["indie folk", "folk pop", "dream pop"],
        expected_moods=["melancholic", "calm", "nostalgic", "chill"],
    ),
    TestCase(
        description="Upbeat party pop",
        query="upbeat fun pop songs for a party or celebration",
        expected_genres=["pop", "indie pop", "soul"],
        expected_moods=["happy", "playful", "euphoric", "energetic"],
    ),
    TestCase(
        description="Dark electronic",
        query="dark moody electronic music with a brooding atmosphere",
        expected_genres=["electronic", "synthwave"],
        expected_moods=["dark", "moody", "intense"],
    ),
    TestCase(
        description="Jazz cafe morning",
        query="relaxed jazz for a quiet coffee shop morning",
        expected_genres=["jazz", "lofi", "ambient"],
        expected_moods=["relaxed", "chill", "calm", "focused"],
    ),
    TestCase(
        description="Romantic date night",
        query="romantic slow songs for a date night",
        expected_genres=["pop", "r&b", "soul", "indie pop", "dream pop"],
        expected_moods=["romantic", "nostalgic", "moody"],
    ),
    TestCase(
        description="Aggressive rap",
        query="aggressive rap with attitude and hard-hitting beats",
        expected_genres=["hip-hop", "electronic", "rock"],
        expected_moods=["angry", "intense", "dark"],
    ),
]


def evaluate(test: TestCase, songs: list, embeddings) -> dict:
    results = retrieve(test.query, songs, embeddings, k=5)
    top3_genres = {r[0]["genre"] for r in results[:3]}
    top3_moods  = {r[0]["mood"]  for r in results[:3]}
    top_sim     = results[0][1] if results else 0.0

    genre_hit = bool(top3_genres & set(test.expected_genres))
    mood_hit  = bool(top3_moods  & set(test.expected_moods))

    return {
        "description": test.description,
        "passed":      genre_hit and mood_hit,
        "genre_hit":   genre_hit,
        "mood_hit":    mood_hit,
        "confidence":  top_sim,
        "top_result":  results[0][0]["title"] if results else "-",
        "top_genre":   results[0][0]["genre"]  if results else "-",
        "top_mood":    results[0][0]["mood"]   if results else "-",
    }


def main() -> None:
    print("Loading catalog ...", end=" ", flush=True)
    songs = load_songs("data/songs.csv")
    embeddings = embed_catalog(songs)
    print(f"{len(songs)} songs ready.\n")

    outcomes = [evaluate(tc, songs, embeddings) for tc in TEST_CASES]

    rows = []
    for r in outcomes:
        rows.append([
            r["description"],
            "PASS" if r["passed"] else "FAIL",
            f"{r['confidence']:.3f}",
            "Y" if r["genre_hit"] else "N",
            "Y" if r["mood_hit"]  else "N",
            f"{r['top_result']} ({r['top_genre']}, {r['top_mood']})",
        ])

    print(tabulate(
        rows,
        headers=["Test", "Status", "Confidence", "Genre", "Mood", "Top Result"],
        tablefmt="grid",
    ))

    passed   = sum(1 for r in outcomes if r["passed"])
    total    = len(outcomes)
    avg_conf = sum(r["confidence"] for r in outcomes) / total

    print(f"\n  {passed}/{total} passed   avg confidence: {avg_conf:.3f}")
    if passed < total:
        print("  Failed tests:")
        for r in outcomes:
            if not r["passed"]:
                miss = []
                if not r["genre_hit"]: miss.append("genre")
                if not r["mood_hit"]:  miss.append("mood")
                print(f"    - {r['description']}: missed {' and '.join(miss)}")
    print()


if __name__ == "__main__":
    main()
