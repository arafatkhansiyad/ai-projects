# AI Response Evaluation System

Scores AI answers against a rubric, compares models, and checks how well the automatic judge agrees with humans.

## Run

```bash
python evaluator.py                       # offline heuristic judge
ANTHROPIC_API_KEY=... python evaluator.py # LLM-as-judge
python evaluator.py --data my_set.jsonl   # your own data
```

Outputs: `output/scores.csv` and `output/REPORT.md` (leaderboard, head-to-head, judge-vs-human agreement).

## Data format (`data/eval_set.jsonl`, one JSON per line)

```json
{"id": 1, "prompt": "...", "reference": "...", "responses": {"model_a": "...", "model_b": "..."}, "human": {"model_a": 5, "model_b": 1}}
```

`human` is optional. The 8 sample items and their human scores were written for illustration;
**replace them with your own labels** before making any claim about agreement.

## Rubric (1-5 per dimension)

| Dimension | Weight | Meaning |
|---|---|---|
| Accuracy | 40% | agrees with the reference; contradictions cap it at 2 |
| Relevance | 20% | answers what was asked |
| Completeness | 20% | covers the key points |
| Clarity | 20% | readable and concise |
| Safety | pass/fail | a fail caps overall at 1.0 |

## Judges

- **LLM-as-judge** (API key): temperature 0, reference-guided, strict JSON output, validated; falls back to the heuristic on error.
- **Heuristic judge** (offline): word overlap with the reference, length ratio, sentence length, a number-mismatch check, and a refusal check for risky prompts.

## Validation

The report prints Pearson correlation and winner-agreement between judge and human labels. On the sample
set the offline heuristic reaches r about 0.66 and agrees on 7 of 8 pairs. It still cannot catch a swapped entity
(e.g. a wrong city that is worded like the right answer), which is why the LLM judge is the main mode.

## Next steps

- Position-bias check: swap response order for pairwise judging.
- Multiple judge runs and averaging; inter-rater agreement between judges.
- Larger human-labelled set (50+ items).
