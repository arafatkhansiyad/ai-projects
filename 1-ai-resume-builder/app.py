#!/usr/bin/env python3
"""AI Resume Builder: tailor a resume to a job description and report ATS keyword coverage.

Usage:
    python app.py --profile sample_profile.json --jd sample_job.txt --out output

With ANTHROPIC_API_KEY or OPENAI_API_KEY set, the summary and bullets are rewritten by an LLM.
Without a key it runs an offline rule-based mode (stronger verbs, skills reordering, ATS report).
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import llm  # noqa: E402

VOCAB = [
    "python", "javascript", "sql", "api", "json", "prompt engineering", "llm", "openai", "claude",
    "gemini", "rag", "evaluation", "automation", "zapier", "n8n", "webhook", "github", "git",
    "streamlit", "flask", "react", "excel", "google sheets", "data analysis", "nlp",
    "machine learning", "chatbot", "documentation", "testing", "debugging", "docker", "aws",
    "notion", "airtable", "fine-tuning", "workflow", "integration", "communication",
    "problem solving", "a/b testing", "rubric",
]

WEAK_STARTS = [
    ("worked on", "Built"), ("responsible for", "Owned"), ("helped with", "Contributed to"),
    ("helped to", "Helped"), ("was involved in", "Contributed to"), ("made", "Built"),
    ("did", "Delivered"), ("used", "Applied"),
]

SYSTEM = (
    "You are an expert resume writer. Never invent employers, dates, tools or numbers "
    "that are not present in the candidate data."
)


def has_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def strengthen(bullet: str) -> str:
    b = bullet.strip().rstrip(".")
    low = b.lower()
    for weak, strong in WEAK_STARTS:
        if low.startswith(weak + " "):
            b = strong + b[len(weak):]
            break
    return b[0].upper() + b[1:] + "."


def resume_text(p: dict) -> str:
    parts = [p.get("title", ""), p.get("summary", ""), " ".join(p.get("skills", []))]
    for item in p.get("experience", []) + p.get("projects", []):
        parts += [item.get("role", ""), item.get("name", ""), item.get("tools", "")] + item.get("bullets", [])
    return " ".join(parts)


def keyword_report(p: dict, jd: str) -> dict:
    jd_l = jd.lower()
    terms = {t for t in VOCAB if has_term(jd_l, t)}
    terms |= {s.lower() for s in p.get("skills", []) if has_term(jd_l, s.lower())}
    text = resume_text(p).lower()
    matched = sorted(t for t in terms if has_term(text, t))
    missing = sorted(terms - set(matched))
    score = round(100 * len(matched) / len(terms)) if terms else 0
    return {"ats_score": score, "jd_keywords": sorted(terms), "matched": matched, "missing": missing}


def order_skills(p: dict, jd: str) -> list[str]:
    jd_l = jd.lower()
    return sorted(p.get("skills", []), key=lambda s: (not has_term(jd_l, s.lower()), p["skills"].index(s)))


def rule_tailor(p: dict, jd: str) -> dict:
    out = copy.deepcopy(p)
    top = [s for s in order_skills(p, jd) if has_term(jd.lower(), s.lower())][:4]
    base = p.get("summary", "").strip()
    lead = f"{p.get('title', 'Candidate')} with hands-on experience in {', '.join(top)}." if top else ""
    out["summary"] = (lead + " " + base).strip()
    for item in out.get("experience", []) + out.get("projects", []):
        item["bullets"] = [strengthen(b) for b in item["bullets"]]
    return out


def llm_tailor(p: dict, jd: str) -> dict:
    prompt = (
        "Rewrite parts of this resume for the job below.\n"
        'Return ONLY JSON: {"summary": "2-3 sentences", '
        '"experience_bullets": [[...], ...], "project_bullets": [[...], ...]}\n'
        "- Keep the same number of bullets per item and the same order.\n"
        "- Start each bullet with a strong action verb; max 25 words.\n"
        "- Mirror job-description wording only where it is truthful for the candidate.\n"
        "- Do NOT add numbers, tools or achievements that are not in the candidate data.\n\n"
        f"JOB DESCRIPTION:\n{jd}\n\nCANDIDATE DATA (JSON):\n{json.dumps(p, indent=2)}"
    )
    data = llm.extract_json(llm.complete(prompt, system=SYSTEM, max_tokens=1800, temperature=0.3).text)
    out = copy.deepcopy(p)
    out["summary"] = str(data["summary"]).strip()
    for key, field in (("experience_bullets", "experience"), ("project_bullets", "projects")):
        new = data[key]
        if len(new) != len(out.get(field, [])):
            raise ValueError(f"{key}: item count mismatch")
        for item, bullets in zip(out.get(field, []), new):
            if len(bullets) != len(item["bullets"]):
                raise ValueError(f"{key}: bullet count mismatch")
            item["bullets"] = [str(b).strip() for b in bullets]
    return out


def contact_line(p: dict) -> str:
    c = p.get("contact", {})
    return " | ".join(x for x in [c.get("email"), c.get("phone"), c.get("location"), *c.get("links", [])] if x)


def render_md(p: dict) -> str:
    L = [f"# {p['name']}", f"**{p.get('title', '')}**", "", contact_line(p), "",
         "## Summary", p.get("summary", ""), "", "## Skills", ", ".join(p.get("skills", [])), ""]
    if p.get("experience"):
        L.append("## Experience")
        for e in p["experience"]:
            L += [f"### {e['role']}, {e['company']} ({e['dates']})"] + [f"- {b}" for b in e["bullets"]] + [""]
    if p.get("projects"):
        L.append("## Projects")
        for pr in p["projects"]:
            link = f" | {pr['link']}" if pr.get("link") else ""
            L += [f"### {pr['name']} | {pr.get('tools', '')}{link}"] + [f"- {b}" for b in pr["bullets"]] + [""]
    if p.get("education"):
        L.append("## Education")
        L += [f"- {e['degree']}, {e['school']} ({e['dates']})" for e in p["education"]]
    return "\n".join(L).strip() + "\n"


def render_html(p: dict) -> str:
    e = html.escape

    def ul(items):
        return "<ul>" + "".join(f"<li>{e(i)}</li>" for i in items) + "</ul>"

    body = [f"<h1>{e(p['name'])}</h1>", f"<p class='sub'>{e(p.get('title', ''))}</p>",
            f"<p class='contact'>{e(contact_line(p))}</p>",
            f"<h2>Summary</h2><p>{e(p.get('summary', ''))}</p>",
            f"<h2>Skills</h2><p>{e(', '.join(p.get('skills', [])))}</p>"]
    if p.get("experience"):
        body.append("<h2>Experience</h2>")
        for x in p["experience"]:
            body.append(f"<h3>{e(x['role'])}, {e(x['company'])} <span>{e(x['dates'])}</span></h3>{ul(x['bullets'])}")
    if p.get("projects"):
        body.append("<h2>Projects</h2>")
        for x in p["projects"]:
            link = f" | {e(x['link'])}" if x.get("link") else ""
            body.append(f"<h3>{e(x['name'])} <span>{e(x.get('tools', ''))}{link}</span></h3>{ul(x['bullets'])}")
    if p.get("education"):
        body.append("<h2>Education</h2>" + ul(f"{x['degree']}, {x['school']} ({x['dates']})" for x in p["education"]))
    css = (
        "body{font:14px/1.45 Arial,Helvetica,sans-serif;max-width:780px;margin:32px auto;color:#111;padding:0 16px}"
        "h1{margin:0;font-size:26px}.sub{margin:2px 0;font-weight:600}.contact{color:#444;margin:2px 0 12px}"
        "h2{font-size:15px;text-transform:uppercase;border-bottom:1px solid #999;margin:18px 0 6px;padding-bottom:2px}"
        "h3{font-size:14px;margin:10px 0 2px}h3 span{font-weight:400;color:#555;font-size:13px}"
        "ul{margin:2px 0 6px 18px;padding:0}@media print{body{margin:0}}"
    )
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{e(p['name'])} - Resume</title><style>{css}</style></head><body>{''.join(body)}</body></html>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="candidate profile JSON")
    ap.add_argument("--jd", required=True, help="job description text file")
    ap.add_argument("--out", default="output", help="output folder")
    ap.add_argument("--offline", action="store_true", help="force offline rule-based mode")
    args = ap.parse_args()
    if args.offline:
        os.environ["LLM_PROVIDER"] = "mock"

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    jd = Path(args.jd).read_text(encoding="utf-8")
    before = keyword_report(profile, jd)

    mode, tailored = "offline rule-based", None
    if llm.is_live():
        try:
            tailored, mode = llm_tailor(profile, jd), f"AI rewrite ({llm.model_name()})"
        except (llm.LLMError, ValueError, KeyError, TypeError) as exc:
            print(f"[warn] AI step failed ({exc}); falling back to offline mode.")
    if tailored is None:
        tailored = rule_tailor(profile, jd)
    tailored["skills"] = order_skills(tailored, jd)
    after = keyword_report(tailored, jd)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume.md").write_text(render_md(tailored), encoding="utf-8")
    (out / "resume.html").write_text(render_html(tailored), encoding="utf-8")
    report = {"mode": mode, "before": before, "after": after}
    (out / "ats_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Mode: {mode}")
    print(f"ATS keyword match: {before['ats_score']}% -> {after['ats_score']}%  ({len(after['matched'])}/{len(after['jd_keywords'])} JD keywords)")
    if after["missing"]:
        print("JD keywords not in your resume (add ONLY if they are true for you): " + ", ".join(after["missing"]))
    print(f"Saved: {out}/resume.md, resume.html (open in a browser -> Print -> Save as PDF), ats_report.json")


if __name__ == "__main__":
    main()
