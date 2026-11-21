# AI Resume Builder

Tailors a resume to a specific job description and shows how well it matches the job's keywords.

## Run

```bash
python app.py --profile sample_profile.json --jd sample_job.txt --out output
```

Outputs in `output/`: `resume.md`, `resume.html` (open in a browser, Print, Save as PDF), `ats_report.json`.

## How it works

1. **Profile** (`sample_profile.json`) holds your real data: skills, experience, projects, education.
2. **Keyword report**: extracts skills/tools from the job description and checks which appear in the resume (ATS match %). Lists missing ones so you can add them *only if true*.
3. **Rewrite**
   - *AI mode* (API key set): the LLM rewrites the summary and bullets using the job's wording. The prompt forbids inventing employers, tools or numbers, and the output is validated (same number of items and bullets) before use.
   - *Offline mode*: replaces weak openers ("worked on", "helped with") with action verbs, builds a targeted summary and moves job-relevant skills first.
4. Renders an ATS-friendly single-column resume.

## Design decisions

- Truthfulness first: the tool never adds skills you did not list.
- Fallback: if the AI reply is malformed, it falls back to offline mode instead of failing.
- Plain, single-column layout because ATS parsers handle it best.

## Limitations / next steps

- Keyword matching uses a fixed vocabulary (`VOCAB` in `app.py`); extend it for your field.
- Add a Streamlit UI or DOCX export.
