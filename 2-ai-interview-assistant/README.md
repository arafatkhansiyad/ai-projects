# AI Interview Assistant

Practice interview questions for a chosen role and get structured feedback on each answer.

## Run

```bash
# interactive: type your answers
python interview.py --role ai-prompt-engineer --n 4

# demo with prepared answers
python interview.py --role ai-prompt-engineer --n 4 --answers sample_answers.json

# AI mode: questions generated from a real job description
python interview.py --role automation-specialist --n 5 --jd job.txt
```

Roles: `ai-prompt-engineer`, `automation-specialist`, `data-analyst`, `software-developer`.
Report is saved to `output/session_report.md` and `.json`.

## Feedback

**Rubric (always on, offline, 10 points)**

| Part | Behavioral questions | Technical questions |
|---|---|---|
| Length (2) | 60-200 words | 60-200 words |
| Structure | STAR coverage: Situation, Task, Action, Result (4) | key ideas from the question covered (5) |
| Numbers (2) | at least two figures | at least two figures |
| Ownership (1) | uses "I" for own actions | same |
| Fillers (1) | 2 or fewer | same |

Each missing part produces a specific tip ("Finish with the outcome, ideally with a number").

**AI coach (only with an API key)**: strengths, improvements and a stronger-answer outline built
only from facts the candidate already gave.

## Limitations

- The rubric checks *structure*, not whether the content is correct. Use the AI coach or a human for that.
- Regex-based STAR detection can miss unusual phrasing.
- Next steps: speech-to-text input, saved history across sessions, Streamlit UI.
