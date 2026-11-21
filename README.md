# AI Projects Portfolio

Four small, runnable projects around LLM applications. Each one works **offline** (demo mode) and
switches to a real model when you set an API key. Pure Python 3.10+, no installs needed.

| # | Project | What it shows |
|---|---|---|
| 1 | [AI Resume Builder](1-ai-resume-builder) | Tailors a resume to a job description, ATS keyword report, HTML/PDF/Markdown output |
| 2 | [AI Interview Assistant](2-ai-interview-assistant) | Role-based questions, rubric feedback (STAR / key ideas / metrics), optional AI coaching |
| 3 | [Prompt Optimization Case Study](3-prompt-optimization-case-study) | 4 prompt versions tested on 12 labelled tickets, measured improvement per change |
| 4 | [AI Response Evaluation System](4-ai-response-evaluation-system) | Rubric scoring (LLM-as-judge), model comparison, agreement with human labels |

## Quick start

```bash
cd 1-ai-resume-builder && python app.py --profile sample_profile.json --jd sample_job.txt
cd ../2-ai-interview-assistant && python interview.py --role ai-prompt-engineer --n 4 --answers sample_answers.json
cd ../3-prompt-optimization-case-study && python prompt_lab.py
cd ../4-ai-response-evaluation-system && python evaluator.py
```

## Use a real model

```bash
export ANTHROPIC_API_KEY=...     # or: export OPENAI_API_KEY=...
# optional: ANTHROPIC_MODEL=... / OPENAI_MODEL=... / LLM_PROVIDER=anthropic|openai|mock
```

Without a key, projects 1, 2 and 4 use transparent rule-based logic. Project 3 uses a **simulator**
whose numbers are labelled as simulated and must not be quoted as results.

## Layout

```
common/llm.py   shared minimal client (Anthropic / OpenAI / offline), standard library only
```
