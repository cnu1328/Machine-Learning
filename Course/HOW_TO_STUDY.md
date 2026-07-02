# How to Study This Course

This is not a passive video course. It is an active mentorship program. You learn by doing, deriving, and explaining.

---

## Weekly Schedule (5–8 hours)

| Session | Activity | Time |
|---------|----------|------|
| **Session 1** | Read module README + theory cells in notebooks | 2–3 hrs |
| **Session 2** | Code along: run every cell, modify parameters, break things intentionally | 1.5–2 hrs |
| **Session 3** | Exercises (attempt fully before opening solutions) | 1–1.5 hrs |
| **Session 4** | Assignment or mini project | 1–2 hrs |
| **Review** | Quiz + interview questions (say answers aloud) + update PROGRESS.md | 30 min |

---

## The 10-Step Learning Loop (Every Concept)

For each new concept in a notebook, ensure you can answer all ten:

1. **Intuition** — What is it in plain English?
2. **Why** — Why was it invented? What problem does it solve?
3. **Where** — Where is it used in industry/research?
4. **Math** — What is the equation? Can you write it from memory?
5. **Variables** — What does every symbol mean?
6. **Derivation** — Can you derive it step by step?
7. **Code** — Can you implement it without looking?
8. **Optimization** — What hyperparameters matter? How do you tune?
9. **Mistakes** — What do beginners get wrong?
10. **Interview** — How would you explain this in a 2-minute answer?

If you cannot answer any of these, re-read that section before moving on.

---

## Notebook Workflow

1. Open Jupyter Lab from the `Machine-Learning` directory:
   ```bash
   cd Machine-Learning
   source .venv/bin/activate   # if using venv
   jupyter lab
   ```
2. Navigate to the module folder
3. Read markdown cells carefully — theory is not optional
4. Run code cells one at a time; do not skip ahead
5. Complete exercise cells marked `# YOUR CODE HERE` before scrolling to solutions
6. At the end of each notebook, write 3 sentences summarizing what you learned

---

## Exercise Rules

- **Attempt first.** Spend at least 20 minutes before asking for hints.
- **Hints, not answers.** Your mentor gives hints, not full solutions.
- **Submit work.** Paste your exercise code in chat for review.
- **Solutions folder.** Only open `exercises/solutions/` after submitting your attempt.

---

## Module Gate (Required to Advance)

Before starting the next module, you must:

1. Complete all notebooks in the current module
2. Finish all exercises and the assignment
3. Score ≥80% on the module quiz
4. Answer 3 conceptual questions from your mentor in chat
5. Mark the module complete in [PROGRESS.md](PROGRESS.md)

**Example message to mentor:**

> I completed Module 01. Quiz score: 13/15. Assignment: EDA on Titanic attached. Ready for gate questions.

---

## Note-Taking Recommendations

Create a personal `Course/notes/` folder (gitignored) with:

- `math_derivations.md` — every derivation you work through
- `interview_answers.md` — your spoken answers to interview questions
- `mistakes_log.md` — bugs you hit and what you learned

---

## When You Get Stuck

1. Re-read the intuition section
2. Draw a diagram on paper
3. Print variable shapes (for NumPy/PyTorch)
4. Ask your mentor with **specific** questions:
   - Bad: "I don't understand broadcasting"
   - Good: "Why does `(3,4) + (4,)` work but `(3,4) + (3,)` raise an error?"

---

## Spaced Repetition

- After finishing a module, revisit its cheat sheet weekly for 4 weeks
- Before starting Module 03, skim Module 02 cheat sheet
- Before the capstone, review Modules 05, 07, and 11

---

## What Success Looks Like

By Module 12, you should be able to:

- Read a segmentation paper and implement the architecture
- Debug a training run without AI assistance
- Explain every line in `water-bodies-detection/train.py`
- Design a production ML pipeline for a new GeoSpatial project
- Pass a senior ML engineer technical interview

---

## Anti-Patterns (Avoid These)

| Anti-pattern | Why it fails |
|--------------|--------------|
| Skipping math cells | You will not understand why code works |
| Copy-pasting exercise solutions | Zero learning transfer |
| Rushing to Deep Learning | Classical ML intuition is foundational |
| Using AI to write all code | Defeats the purpose of this course |
| Skipping gate questions | Gaps compound in later modules |

---

**Next:** [PREREQUISITES.md](PREREQUISITES.md) → [00_Course_Introduction/](00_Course_Introduction/)
