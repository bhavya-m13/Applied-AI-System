# Music Recommender System — Applied AI

This project started as a simple music recommender simulation from Module 2 and grew into a full AI-powered system with natural language input, semantic search and automated reliability testing.

---

## Base Project

This builds on my **Music Recommender Simulation** from Module 2. The original system scored 20 songs from a CSV catalog against a structured user taste profile using a weighted algorithm. Mood, energy and genre each had fixed point values and the top 5 results were returned with explanations. I also built adversarial user profiles specifically to expose flaws in the scoring logic, like what happens when someone wants high energy music but their mood is set to "chill."

The catalog has since been expanded from 20 to **50 songs** to support semantic retrieval. The additions fill in the mid-energy gap (0.40–0.65) flagged by the bias audit, add previously missing moods (`sad`, `dreamy`, `introspective`), and bring in new genres including `indie rock`, `pop`, `r&b`, and `soul`. All 30 new tracks are mainstream songs (e.g. Blinding Lights, Someone Like You, Lose Yourself, Shape of You, Jolene).

---

## What This Version Does

Instead of requiring structured input, you can now just type what you want in plain English. Something like *"something intense for the gym"* or *"chill acoustic music for studying."* The system embeds your query, retrieves the most relevant songs from the catalog using cosine similarity, scores them with the original algorithm and passes the results to Claude to generate a plain-English explanation for each recommendation.

I kept the original scoring logic because it's transparent and easy to audit. RAG acts as a pre-filter that handles the natural language side and the scorer handles ranking. This way you can still trace exactly why a song ranked where it did.

---

## System Architecture

![System Architecture](assets/system_diagram.png)

**User input** flows into the **RAG retriever** which embeds the query and searches the song catalog for the closest semantic matches. Those candidates go into the **scoring engine** which applies the original weighted algorithm (mood, energy, genre, valence, danceability, acoustic penalty). The top results go to the **AI response generator** which writes a personalized explanation. Underneath all of this sits the **reliability layer** which runs automated tests and logs confidence scores for every query.

---

## Setup

You need Python 3.9+ and an Anthropic API key.

```bash
git clone https://github.com/bhavya-m13/Applied-AI-System.git
cd Applied-AI-System

python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

pip install -r requirements.txt

export ANTHROPIC_API_KEY=your_key_here   # Mac/Linux
set ANTHROPIC_API_KEY=your_key_here      # Windows
```

### Running the system

```bash
# Conversational recommender
python -m src.chat

# Original preset profile suite
python -m src.main

# Bias audit
python -m src.bias_audit

# Automated evaluation harness
python tests/eval_harness.py
```

---

## Sample Interactions

**Chill acoustic query:**
```
You: I want something chill and acoustic for a rainy afternoon

1. Library Rain (score: 9.4)
   Matches your chill mood and acoustic preference. Low energy and high
   acousticness make this a natural fit for a quiet afternoon.

2. Midnight Coding (score: 8.7)
   Soft textures and calm energy throughout. A reliable lofi pick.

3. Stargazing (score: 8.1)
   Gentle tempo and high valence. Peaceful without feeling sleepy.
```

**High energy query:**
```
You: Give me something intense and fast for the gym

1. Storm Runner (score: 10.2)
   High energy, intense mood and strong danceability. Closest match to what you described.

2. Harlequin (score: 9.8)
   Near-perfect energy and valence alignment with an aggressive tempo.

3. Rooftop Lights (score: 8.9)
   Slightly less intense but still a strong high-energy pop pick.
```

**Sad mood query (now supported):**
```
You: I'm feeling really sad today

1. Someone Like You — Adele  (similarity: 0.74)
2. Circles — Post Malone     (similarity: 0.71)
3. good riddance — Gracie Abrams  (similarity: 0.68)

  These tracks share a low-energy, low-valence quality that fits
  a sad or reflective headspace. Adele and Gracie Abrams lean
  acoustic and melancholic; Circles is quieter pop with a
  wistful tone throughout.
```

Note: in the original 20-song catalog `sad` was not a mood label and the system silently degraded to energy/valence proximity. The expanded 50-song catalog includes three `sad`-mood tracks so this query now resolves correctly.

---

## Design Decisions

The biggest decision was keeping the original weighted scorer instead of replacing it with pure embedding ranking. Embedding similarity is useful for matching natural language to songs but it doesn't always reflect what a human would call musical similarity. The scorer gives you a traceable reason for every result.

The acoustic penalty is still one-directional which is a known flaw from the original project. Acoustic lovers get no positive signal, only non-acoustic fans get penalized. I kept it because fixing it would change the scoring behavior significantly and I wanted to document it rather than quietly patch it.

Mood labels are still discrete strings so "chill" and "relaxed" score differently even if they mean the same thing to a listener. This is the hardest problem to fix without a much larger catalog and better label taxonomy.

---

## Testing Summary

The evaluation harness runs 8 predefined profiles through the full pipeline and checks whether the top result matches the expected song, whether confidence scores exceed 0.7 and whether the explanation references the right mood or genre.

6 out of 8 tests passed. Both failures were edge cases: the "sad mood" profile where the mood doesn't exist in the catalog and the "all-median" profile where no strong signal fires. Confidence scores averaged 0.79 on passing tests and 0.58 on failing ones, which confirmed the threshold was actually catching the right cases.

The bias audit from the original project still holds. Mood weight dominates the score. A user asking for high-energy music with a "chill" mood label will consistently get calm songs regardless of their energy setting.

---

## Reflection

The part that surprised me most was how quickly the original system broke when I switched from structured input to natural language. Every assumption about input format stopped working immediately. RAG fixed the input problem but introduced a new one: the embedding model's idea of similarity doesn't always match what a human would consider musically similar.

The confidence scoring ended up being the most useful addition. Small catalogs like this one can return a ranked list that looks authoritative even when the system basically has no idea what to recommend. Surfacing that uncertainty to the user is more honest than pretending the top result is always right.

---

## Demo

🎥 Loom walkthrough: [link coming before submission]

---

## Repo Structure

```
Applied-AI-System/
├── assets/
│   └── system_diagram.png
├── data/
│   └── songs.csv
├── src/
│   ├── main.py
│   ├── chat.py
│   ├── rag.py
│   ├── recommender.py
│   └── bias_audit.py
├── tests/
│   ├── test_recommender.py
│   └── eval_harness.py
├── model_card.md
├── README.md
└── requirements.txt
```

---