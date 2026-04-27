"""
rag.py — vector retrieval layer for the music recommender.

Converts each song to a short text description, embeds the full catalog
with a local sentence-transformers model, then retrieves the closest
songs to a natural-language query using cosine similarity.
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def song_to_text(song: Dict) -> str:
    """
    Render a song as a rich natural-language description so that
    conversational queries ("gym music", "sad rainy day") embed close
    to the right songs.
    """
    energy    = float(song["energy"])
    valence   = float(song["valence"])
    dance     = float(song["danceability"])
    acoustic  = float(song["acousticness"])
    genre     = song["genre"]
    mood      = song["mood"]

    # Energy feel
    if energy > 0.85:
        energy_desc = "very high energy, intense and powerful"
    elif energy > 0.70:
        energy_desc = "high energy and upbeat"
    elif energy > 0.50:
        energy_desc = "moderately energetic"
    elif energy > 0.35:
        energy_desc = "calm and relaxed"
    else:
        energy_desc = "very calm, quiet and peaceful"

    # Emotional tone
    if valence > 0.80:
        valence_desc = "very happy, joyful and feel-good"
    elif valence > 0.60:
        valence_desc = "positive and uplifting"
    elif valence > 0.40:
        valence_desc = "neutral emotional tone"
    elif valence > 0.25:
        valence_desc = "dark or melancholic"
    else:
        valence_desc = "very sad, heavy and emotional"

    # Texture
    texture = "acoustic and organic" if acoustic > 0.70 else "produced or electronic sound"

    # Use-case hints bridge the gap between tags and how people talk
    hints: list[str] = []
    if genre in ("lofi", "ambient") or mood in ("chill", "focused", "calm", "relaxed"):
        hints.append("good for studying, focusing or late-night background listening")
    if energy > 0.80 and mood in ("intense", "energetic", "angry", "euphoric", "dark"):
        hints.append("great for the gym, workouts or high-intensity activity")
    if mood in ("romantic", "nostalgic", "moody"):
        hints.append("suits a date night, slow dance or reflective evening")
    if mood in ("sad", "melancholic"):
        hints.append("fits sad, rainy-day or heartbreak moments")
    if genre == "jazz" or mood == "relaxed":
        hints.append("coffee shop or cafe atmosphere")
    if dance > 0.80 and energy > 0.70:
        hints.append("great for dancing or parties")
    if mood in ("happy", "playful", "euphoric") and energy > 0.70:
        hints.append("fun, celebratory and feel-good")

    hint_str = ". ".join(hints) + "." if hints else ""

    return (
        f"{song['title']} by {song['artist']} is a {genre} song with a {mood} mood. "
        f"It is {energy_desc} with a {valence_desc} feel and {texture}. "
        f"{hint_str}"
    ).strip()


def embed_catalog(songs: List[Dict]) -> np.ndarray:
    """
    Embed every song in the catalog.

    Returns a (n_songs, dim) float32 matrix where each row is
    L2-normalised so that dot-product equals cosine similarity.
    """
    model = _get_model()
    texts = [song_to_text(s) for s in songs]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).astype(np.float32)


def retrieve(
    query: str,
    songs: List[Dict],
    catalog_embeddings: np.ndarray,
    k: int = 5,
) -> List[Tuple[Dict, float]]:
    """
    Embed `query` and return the top-k songs by cosine similarity.

    Parameters
    ----------
    query              : natural-language description of what the user wants
    songs              : song dicts in the same order as catalog_embeddings
    catalog_embeddings : pre-computed matrix from embed_catalog()
    k                  : number of results to return

    Returns
    -------
    List of (song_dict, similarity_score) sorted descending.
    """
    model = _get_model()
    q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_vec = (q_vec / np.linalg.norm(q_vec)).astype(np.float32)

    scores = (catalog_embeddings @ q_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:k]
    return [(songs[idx], float(scores[idx])) for idx in top_idx]
