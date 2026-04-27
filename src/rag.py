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
    """Render a song dict as a natural-language sentence for embedding."""
    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"Energy: {float(song['energy']):.2f}, "
        f"valence: {float(song['valence']):.2f}, "
        f"danceability: {float(song['danceability']):.2f}."
    )


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
