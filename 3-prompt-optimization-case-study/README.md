# Prompt Optimization Case Study

**Question:** how much does each prompt-engineering technique improve a real task?

**Task:** classify customer-support tickets into `category`, `sentiment`, `urgency` as strict JSON.
**Data:** `data/tickets.json`, 12 tickets with human labels.

## Prompt versions (one change at a time)

| Version | Change |
|---|---|
| v1 | Vague one-liner |
| v2 | + role, ask for JSON with field names |
| v3 | + allowed values for every field |
| v4 | + urgency definition, output-only rule, 3 few-shot examples |

## Run

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
python prompt_lab.py
```

Writes `output/REPORT.md` (results table, remaining errors) and `output/results.json` (raw outputs).

**Metrics:** schema-valid JSON, clean output (JSON only), per-field accuracy, overall field accuracy, output tokens.

> **Important:** with no API key the script runs a seeded *simulator* so you can test the pipeline.
> Its report is stamped "SIMULATED" and the numbers are not real. For a CV or interview, run it with a real model
> and quote only those numbers.

## After your real run

Add a short findings section to this README, for example:
- which change gave the biggest jump and why you think so,
- which field stayed hardest and what the errors had in common,
- what you would try next (more data, a different model, self-consistency).

## Limitations

- 12 tickets is small: one ticket is about 8 percentage points. Increase the data before drawing strong conclusions.
- Labels are one person's judgement; urgency is subjective.
