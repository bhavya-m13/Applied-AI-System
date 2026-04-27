"""
chat.py — natural-language CLI for the music recommender.

Flow per turn:
  1. User types a plain-English request
  2. RAG retrieves the closest songs from the catalog
  3. Claude turns the raw results into a friendly 2-3 sentence explanation
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import anthropic
from tabulate import tabulate
from recommender import load_songs
from rag import embed_catalog, retrieve

_SYSTEM = (
    "You are a music concierge. "
    "Given a user's listening request and a shortlist of retrieved songs, "
    "write 2-3 sentences explaining why these tracks match the vibe. "
    "Be specific: reference genre, mood, and energy. Keep it conversational."
)


def _explain(client: anthropic.Anthropic, query: str, results: list) -> str:
    lines = [
        f'- "{s["title"]}" by {s["artist"]} '
        f'({s["genre"]}, {s["mood"]}, energy={s["energy"]})'
        for s, _ in results
    ]
    user_msg = f'Request: "{query}"\n\nRetrieved songs:\n' + "\n".join(lines)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text


def _print_results(results: list) -> None:
    rows = [
        [i + 1, s["title"], s["artist"], s["genre"], s["mood"], f"{sim:.3f}"]
        for i, (s, sim) in enumerate(results)
    ]
    print(tabulate(
        rows,
        headers=["#", "Title", "Artist", "Genre", "Mood", "Similarity"],
        tablefmt="grid",
    ))


def main() -> None:
    print("Loading catalog ...", end=" ", flush=True)
    songs = load_songs("data/songs.csv")
    embeddings = embed_catalog(songs)
    print("ready.\n")

    client = anthropic.Anthropic()

    print("Music Recommender  (type 'quit' to exit)\n")
    while True:
        try:
            query = input("What are you in the mood for? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        results = retrieve(query, songs, embeddings, k=5)
        print()
        _print_results(results)
        print()
        explanation = _explain(client, query, results)
        print(f"  {explanation}\n")


if __name__ == "__main__":
    main()
