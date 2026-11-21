#!/usr/bin/env python3
"""Prompt Optimization Case Study: compare prompt versions on a labelled test set.

Task: triage customer-support tickets into category / sentiment / urgency (strict JSON).
For each prompt version (prompts/v1.txt ... v4.txt) the script runs every ticket in
data/tickets.json, parses the output and scores it against the human labels.

    python prompt_lab.py                 # real run if an API key is set, otherwise a SIMULATION
    python prompt_lab.py --versions v1,v4

IMPORTANT: without an API key the "model" is a seeded simulator. Its numbers are for testing the
pipeline only and must NOT be quoted as results. Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import llm  # noqa: E402

HERE = Path(__file__).resolve().parent
FIELDS = ["category", "sentiment", "urgency"]
CATS = ["billing", "bug", "account", "feature_request", "other"]
SENTS = ["positive", "neutral", "negative"]
URGS = ["low", "medium", "high"]
NOTES = {
    "v1": "Vague one-line instruction",
    "v2": "Added a role and asked for JSON with field names",
    "v3": "Added the allowed values for every field",
    "v4": "Added an urgency definition, an output-only rule and 3 few-shot examples",
}


# ---------- offline simulator (clearly NOT a real model) ----------
def simulate(prompt: str, ticket: dict, rng: random.Random) -> str:
    p = prompt.lower()
    role, js = "you are" in p, "json" in p
    allowed, rules, shots = "one of" in p, "rules:" in p, "examples:" in p
    valid_p = min(0.98, 0.30 + 0.30 * js + 0.15 * allowed + 0.15 * rules + 0.10 * shots)
    acc = min(0.97, 0.50 + 0.04 * role + 0.08 * allowed + 0.06 * rules + 0.12 * shots)
    truth = ticket["labels"]
    if rng.random() > valid_p:
        return f"This looks like a {truth['category'].replace('_', ' ')} issue and the customer seems {truth['sentiment']}."
    out = {}
    for f, pool in zip(FIELDS, (CATS, SENTS, URGS)):
        out[f] = truth[f] if rng.random() < acc else rng.choice([v for v in pool if v != truth[f]])
    text = json.dumps(out)
    if rng.random() > min(0.98, 0.40 + 0.15 * js + 0.40 * rules):
        text = "Here is the classification:\n```json\n" + text + "\n```"
    return text


# ---------- scoring ----------
def score(raw: str, truth: dict) -> dict:
    clean = raw.strip().startswith("{") and raw.strip().endswith("}")
    try:
        obj = llm.extract_json(raw)
    except ValueError:
        obj = None
    if not isinstance(obj, dict):
        return {"parsed": False, "valid": False, "clean": False, "correct": {f: False for f in FIELDS}}
    valid = obj.get("category") in CATS and obj.get("sentiment") in SENTS and obj.get("urgency") in URGS
    return {"parsed": True, "valid": valid, "clean": clean and valid,
            "correct": {f: obj.get(f) == truth[f] for f in FIELDS}}


def run(versions: list[str], tickets: list[dict]) -> dict:
    live = llm.is_live()
    results = {}
    for v in versions:
        template = (HERE / "prompts" / f"{v}.txt").read_text(encoding="utf-8")
        rows = []
        for t in tickets:
            prompt = template.replace("{ticket}", t["text"])
            if live:
                try:
                    r = llm.complete(prompt, max_tokens=300, temperature=0.0)
                    raw, toks = r.text, r.output_tokens
                except llm.LLMError as exc:
                    print(f"[warn] {v} ticket {t['id']}: {exc}")
                    raw, toks = "", 0
            else:
                raw = simulate(prompt, t, random.Random(f"{v}-{t['id']}"))
                toks = max(1, len(raw) // 4)
            rows.append({"id": t["id"], "raw": raw, "tokens": toks, **score(raw, t["labels"])})
        results[v] = rows
        print(f"  {v}: done ({len(rows)} tickets)")
    return results


def summarize(results: dict) -> dict:
    out = {}
    for v, rows in results.items():
        n = len(rows)
        s = {"valid": sum(r["valid"] for r in rows) / n, "clean": sum(r["clean"] for r in rows) / n,
             "tokens": sum(r["tokens"] for r in rows) / n}
        for f in FIELDS:
            s[f] = sum(r["correct"][f] for r in rows) / n
        s["overall"] = sum(s[f] for f in FIELDS) / len(FIELDS)
        out[v] = s
    return out


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def report(summary: dict, results: dict, tickets: list[dict], mode: str, live: bool) -> str:
    by_id = {t["id"]: t for t in tickets}
    L = ["# Prompt Optimization Case Study: results", ""]
    if live:
        L += [f"Model: `{mode}` | Date: {date.today()} | Test set: {len(tickets)} labelled tickets | temperature 0", ""]
    else:
        L += ["> **SIMULATED RESULTS (offline demo, no real model was called).** These numbers only prove the pipeline works. "
              "Do NOT quote them in a CV or interview. Set an API key and re-run to get real results.", ""]
    L += ["| Version | What changed | Valid schema | Clean JSON only | Category | Sentiment | Urgency | **Overall field accuracy** | Avg output tokens |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v, s in summary.items():
        L.append(f"| {v} | {NOTES.get(v, '')} | {pct(s['valid'])} | {pct(s['clean'])} | {pct(s['category'])} | "
                 f"{pct(s['sentiment'])} | {pct(s['urgency'])} | **{pct(s['overall'])}** | {s['tokens']:.0f} |")
    vs = list(summary)
    best = max(vs, key=lambda v: summary[v]["overall"])
    gain = summary[best]["overall"] - summary[vs[0]]["overall"]
    L += ["", f"**Best version: {best}** ({pct(summary[best]['overall'])} overall field accuracy, "
          f"{'+' if gain >= 0 else ''}{100 * gain:.0f} points vs {vs[0]}).", ""]
    weakest_fields = sorted(FIELDS, key=lambda f: summary[best][f])
    L += [f"Hardest field for {best}: **{weakest_fields[0]}** ({pct(summary[best][weakest_fields[0]])}).", "",
          f"## Remaining errors in {best}", ""]
    errs = []
    for r in results[best]:
        wrong = [f for f in FIELDS if not r["correct"][f]]
        if wrong:
            errs.append(f"- Ticket {r['id']} ({', '.join(wrong)} wrong): \"{by_id[r['id']]['text'][:80]}\"")
    L += errs[:8] or ["- none"]
    L += ["", "## Method", "",
          "- Same tickets, same settings, one prompt change per version so each gain can be attributed.",
          "- Metrics: schema-valid JSON, clean output (JSON only, no extra text), per-field accuracy against human labels, output tokens.",
          f"- Test set is small ({len(tickets)} tickets), so treat differences of one ticket (~{100 // len(tickets)} points) as noise."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", default="v1,v2,v3,v4")
    ap.add_argument("--out", default="output")
    ap.add_argument("--offline", action="store_true", help="force the simulator")
    args = ap.parse_args()
    if args.offline:
        os.environ["LLM_PROVIDER"] = "mock"

    tickets = json.loads((HERE / "data" / "tickets.json").read_text(encoding="utf-8"))
    versions = [v.strip() for v in args.versions.split(",")]
    live = llm.is_live()
    print(("LIVE run with " + llm.model_name()) if live else "SIMULATION (no API key): numbers are NOT real results")
    results = run(versions, tickets)
    summary = summarize(results)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"live": live, "model": llm.model_name(), "summary": summary, "rows": results}, indent=2), encoding="utf-8")
    md = report(summary, results, tickets, llm.model_name(), live)
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Saved {out}/REPORT.md and results.json")


if __name__ == "__main__":
    main()
