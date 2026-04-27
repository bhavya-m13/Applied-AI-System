# Model Card: Music Recommender System

## 1. Model Name

M3 (Music Mood Matcher)

---

## 2. Intended Use

M3 is built for people who want music that matches how they feel right now without having to scroll through playlists or know exactly what genre they want. You type something in plain English like "something chill for studying" or "intense gym music" and the system figures out which songs fit best.

It's a classroom project, not a production app. The catalog is small and the scoring logic is simple enough to inspect and explain, which was the point. It was built to understand how recommender systems work from the inside rather than just using one.

---

## 3. How the Model Works

There are two layers working together now.

The first is a retrieval layer built with RAG. When a user types a query, the system converts it into a vector using a sentence-transformers model and compares it against embeddings of every song in the catalog. Songs are described in natural language before being embedded so that queries like "gym music" land near songs tagged as intense and high energy rather than just matching on exact keywords. The top matches come back ranked by cosine similarity.

The second is a scoring layer that applies the original weighted algorithm to the retrieved candidates. Genre match gives 1 point. Mood match gives 3 points because mood is the clearest signal of what someone wants. Energy proximity gives up to 4 points. Valence and danceability contribute smaller amounts. There is also a 1 point penalty if the user dislikes acoustic music and the song is heavily acoustic. Max possible score is 10.5.

When an Anthropic API key is available, Claude generates a short explanation of why the results match the query. If no key is available the retrieval table still shows up with similarity scores so the system is still usable.

---

## 4. Data

The catalog started at 20 songs and was expanded to 50 to support the RAG layer better. More songs means more diverse candidates for retrieval to find.

The 30 songs added are all mainstream tracks including Blinding Lights, Someone Like You, Lose Yourself, Shape of You and Jolene. They were picked to fill gaps the original catalog had. The mid-energy range between 0.40 and 0.65 was very thin before. Moods like sad, dreamy and introspective were completely missing. Genres like indie rock, r&b and soul had little to no representation.

The catalog still reflects a pretty specific slice of popular Western music from the last few decades. It skews toward English-language pop and rock. Someone with very different taste, like classical only or non-Western genres, would get poor results since there is almost nothing in the catalog that matches them.

---

## 5. Strengths

M3 works best when the user's query has a clear emotional direction. A query like "intense workout music" or "sad acoustic songs" reliably returns the right kind of songs because the RAG embedding puts those queries near the right part of the catalog. The eval harness confirmed this across 8 test cases covering different moods and genres.

The scoring layer gives every result an explainable reason. You can look at any song in the output and see exactly which components contributed to its score. That transparency is something pure embedding-based recommenders don't give you.

---

## 6. Limitations and Bias

Mood weight still dominates the scoring. Mood is worth 3 out of 10.5 points which sounds reasonable until you realize that a mood mismatch on everything else combined still can't cancel out a mood match. A user who wants high energy music but lists "chill" as their mood will keep getting quiet lofi songs at the top. The mood bonus just pulls too hard.

Genre matching is still exact string comparison. A user who likes "pop" gets no credit for "indie pop" or "dream pop" even though those are closely related. The scorer treats them as completely different categories.

The acoustic preference flag only punishes. If you dislike acoustic music and a song is heavily acoustic, you lose a point. But if you love acoustic music, nothing happens. Acoustic fans are invisible to the scorer.

The catalog is still small relative to what a real recommender works with. With 50 songs, one strong genre or mood cluster can dominate the entire ranked list. Some genres still only have one or two songs so users with niche taste run out of options fast.

The catalog also reflects a narrow slice of musical taste. Almost everything in it is English-language Western pop from the last 20 years or so. Someone who listens to Afrobeats, K-pop, classical or non-Western music would get poor results not because the algorithm is broken but because the data just doesn't represent them at all. That's a real bias and it would take a much more intentional data collection process to fix.

---

## 6b. Misuse Risks

A music recommender seems pretty harmless on the surface but there are a few ways it could go wrong.

The biggest one is if the mood input was used to make inferences about someone's mental state. Right now M3 just uses mood as a filter for what kind of music to return. But if someone always queries for sad or low-energy music, a more sophisticated version of this system could start flagging or profiling them based on those patterns. That's not something M3 does but it's easy to see how a production system could go in that direction, and that would be a pretty serious privacy concern.

Another risk is catalog bias being mistaken for taste. If the catalog only has certain genres and a user gets recommendations from only those genres, they might assume those are their only options rather than realizing the system just doesn't know about other music. It's a small version of the filter bubble problem that real platforms deal with at scale.

To reduce misuse, M3 doesn't store queries or user profiles between sessions. Each request is treated independently with no memory of what came before. Logging or storing that data would be the main thing to avoid if this ever moved beyond a classroom project.

---

## 7. Evaluation

Testing happened in two phases.

The first phase used the original adversarial profiles from the base project. Eight profiles were tested including baselines like High-Energy Pop and Chill Lofi and adversarial ones designed to expose specific flaws. The mood dominance flaw, the missing mood ceiling drop and the unscored tempo field all showed up exactly as expected.

The second phase used the eval harness built for the RAG system. It runs 8 predefined natural language queries through the retrieval pipeline and checks whether the top 3 results contain the expected genres and moods. Each test gets a pass or fail and a confidence score from the cosine similarity of the top result. The harness prints a summary table at the end showing which tests passed and what the average confidence was. This made it easy to see whether changes to the song descriptions improved retrieval without having to manually inspect every result.

---

## 8. Future Work

The catalog still needs to grow. Even at 50 songs some genres and moods only have one or two entries, which means a user with niche taste just runs out of real options. Adding more songs would also reduce the situation where the same songs keep floating to the top across unrelated queries because nothing else scores high enough to beat them.

The mood weight should probably be tuned or replaced. Right now it's a hard 3 point flat bonus which is too blunt. A sliding scale based on how many mood-matching songs exist in the catalog would make more sense.

Getting the API key working would also unlock the full experience. Right now the Claude explanation is silently skipped when there are no credits, which means the output is just a table with numbers. That works for demonstrating retrieval quality but the original vision was for the AI to actually explain the results in natural language.

---

## 9. What Surprised Me About Reliability Testing

The eval harness was more useful than I expected. I thought I'd run it once to confirm everything worked and that would be it, but it actually caught something real. After the first version of the RAG layer was built, the harness showed that several queries were returning the wrong genres in the top 3 results even when the right songs were clearly in the catalog. That pointed directly to the song description format being the problem, not the embedding model. Without the harness I probably would have assumed retrieval was fine and spent a lot of time looking in the wrong place.

The other surprise was how confident the output looks even when the system is clearly struggling. The eval harness showed that a query like "aggressive rap" still returned results with cosine similarity scores around 0.55 even though the top result was wrong. Those scores don't look low enough to raise a red flag but the recommendations were off. That means confidence scores alone aren't a reliable quality signal on a small catalog.

---

## 10. Personal Reflection

The RAG layer was the part that took the most debugging. The first version just formatted songs as "Genre: lofi. Mood: chill. Energy: 0.35" and the results were bad because that kind of text doesn't match how people describe music. Rewriting the descriptions to include phrases like "good for studying" and "great for the gym" made a noticeable difference in retrieval quality.

Getting the API working was also harder than expected. I got a key but had no credits on the account so the explanation step kept throwing a 400 error. The fix was just wrapping the API call in a try-except so the app doesn't crash when the credits run out. It was a small code change but it took a while to figure out what was actually going wrong since the error message was buried in a long traceback.

The most interesting thing I learned is that embedding similarity and human musical similarity are not the same thing. The model might think two songs are close because their descriptions share vocabulary but a human listener would hear them as completely different. That gap is hard to close without either a much better catalog or actual audio features rather than text tags.

---

## 11. Collaboration With AI

I used Claude throughout this project mostly for writing code I didn't know how to structure from scratch, like the RAG retrieval layer and the eval harness.

One genuinely helpful suggestion was when I had the retrieval returning bad results and Claude explained why: the song descriptions were too terse and structured so the embedding model had no way to connect "gym music" to a song tagged only as "rock, intense, energy 0.91." The fix was rewriting the descriptions to include natural language phrases about use cases. That was a real insight I wouldn't have arrived at quickly on my own and it actually fixed the problem.

One suggestion that was wrong was the original song_to_text format itself. Claude wrote it as a short structured string with the genre and mood listed as labels, which seemed fine at first. But when I actually ran queries the results were clearly off and that format turned out to be the root cause. So the same tool that created the problem also diagnosed and fixed it, which was a weird loop to be in. It was a good reminder that AI generated code can look correct and still have a non-obvious design flaw that only shows up when you test it against real inputs.
