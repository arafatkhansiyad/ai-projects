#!/usr/bin/env python3
"""AI Interview Assistant: practice interview questions and get rubric-based feedback.

Interactive:   python interview.py --role ai-prompt-engineer --n 4
Batch (demo):  python interview.py --role ai-prompt-engineer --n 4 --answers sample_answers.json

Feedback has two layers:
  1. A transparent rubric (always runs, offline): length, STAR structure / keyword coverage,
     numbers, ownership, filler words.
  2. LLM coaching (only if ANTHROPIC_API_KEY or OPENAI_API_KEY is set): strengths, fixes and a
     stronger-answer outline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import llm  # noqa: E402

B, T = "behavioral", "technical"
BANK = {
    "ai-prompt-engineer": [
        {"type": B, "q": "Tell me about a time you improved the quality of an AI model's output.", "keywords": []},
        {"type": T, "q": "A prompt gives inconsistent answers. How would you make it more reliable?",
         "keywords": ["role", "format", "example", "constraint", "temperature", "test", "iterate", "schema"]},
        {"type": B, "q": "Describe a project where you had to learn a new tool quickly.", "keywords": []},
        {"type": T, "q": "How would you evaluate whether prompt B is better than prompt A?",
         "keywords": ["test set", "metric", "accuracy", "baseline", "compare", "sample", "rubric", "same"]},
        {"type": B, "q": "Tell me about a time you received critical feedback and what you did with it.", "keywords": []},
        {"type": T, "q": "What is hallucination in LLMs and how can you reduce it?",
         "keywords": ["hallucinat", "ground", "retriev", "source", "verify", "instruction", "temperature", "cite"]},
    ],
    "automation-specialist": [
        {"type": B, "q": "Tell me about an automation you built that saved someone time.", "keywords": []},
        {"type": T, "q": "A workflow fails silently at night. How do you detect and fix it?",
         "keywords": ["log", "alert", "retry", "error", "monitor", "notification", "test", "fallback"]},
        {"type": B, "q": "Describe a time a client's requirements were unclear. What did you do?", "keywords": []},
        {"type": T, "q": "Explain webhooks vs polling and when you would use each.",
         "keywords": ["webhook", "polling", "event", "real-time", "rate limit", "interval", "trigger", "push"]},
        {"type": B, "q": "Tell me about a mistake you made on a project and how you handled it.", "keywords": []},
        {"type": T, "q": "How do you keep API keys and customer data safe in automations?",
         "keywords": ["secret", "environment", "permission", "least", "encrypt", "rotate", "access", "never"]},
    ],
    "data-analyst": [
        {"type": B, "q": "Tell me about a time your analysis changed a decision.", "keywords": []},
        {"type": T, "q": "You find 15% missing values in a key column. What do you do?",
         "keywords": ["missing", "impute", "drop", "pattern", "bias", "mean", "median", "document"]},
        {"type": B, "q": "Describe how you explained a technical result to a non-technical person.", "keywords": []},
        {"type": T, "q": "What is the difference between correlation and causation? Give an example.",
         "keywords": ["correlation", "causation", "confound", "experiment", "example", "random", "third"]},
        {"type": B, "q": "Tell me about a time you worked under a tight deadline.", "keywords": []},
        {"type": T, "q": "How would you check that a dashboard's numbers are correct?",
         "keywords": ["source", "reconcile", "sample", "query", "definition", "test", "compare", "validate"]},
    ],
    "software-developer": [
        {"type": B, "q": "Tell me about a bug that was hard to find and how you solved it.", "keywords": []},
        {"type": T, "q": "How do you decide when code needs refactoring?",
         "keywords": ["duplicate", "readab", "test", "complex", "maintain", "small", "risk", "tradeoff"]},
        {"type": B, "q": "Describe a time you disagreed with a teammate.", "keywords": []},
        {"type": T, "q": "Explain REST APIs and what makes a good one.",
         "keywords": ["http", "resource", "status", "stateless", "json", "versioning", "auth", "idempotent"]},
        {"type": B, "q": "Tell me about a project you are proud of.", "keywords": []},
        {"type": T, "q": "What is the difference between a process and a thread?",
         "keywords": ["memory", "process", "thread", "share", "concurrent", "context", "isolated"]},
    ],
}

FILLERS = r"\b(um+|uh+|like|basically|you know|kind of|sort of|actually|literally)\b"
STAR = {
    "Situation": r"\b(when|during|while|at my|in my|last (year|month|semester)|project|team|class|internship|client)\b",
    "Task": r"\b(my (task|goal|role|job)|i (had|needed|was asked|was responsible)|needed to|goal was|challenge|problem was)\b",
    "Action": r"\bi (?:then |also |first |later |quickly )?(\w+ed|built|wrote|read|made|ran|set up|chose|took|found|led|did|put|got|gave|taught|began)\b",
    "Result": r"\b(result|resulting|reduced|improved|increased|saved|saves|saving|cut|achieved|so that|outcome|led to|ended up|went from|faster|fewer|now (uses|works|runs))\b|\d+\s*%|from \d+ .{0,25}to \d+",
}


def analyze(question: dict, answer: str) -> dict:
    a = answer.strip()
    low = a.lower()
    words = re.findall(r"[A-Za-z0-9'%]+", a)
    wc = len(words)
    numbers = len(re.findall(r"\d+", a))
    fillers = len(re.findall(FILLERS, low))
    i_count = len(re.findall(r"\bi\b", low))
    tips, parts = [], {}

    # length (2 pts)
    if 60 <= wc <= 200:
        parts["length"] = 2
    elif 35 <= wc < 60 or 200 < wc <= 260:
        parts["length"] = 1
        tips.append(f"Length is {wc} words; aim for roughly 60-200 (about 45-90 seconds spoken).")
    else:
        parts["length"] = 0
        tips.append(f"Length is {wc} words; {'add more detail' if wc < 35 else 'trim it down'} (aim for 60-200).")

    if question["type"] == B:
        hit = {k: bool(re.search(p, low)) for k, p in STAR.items()}
        parts["structure"] = sum(hit.values())  # 4 pts
        for k, ok in hit.items():
            if not ok:
                tips.append(f"STAR: no clear {k}. " + {
                    "Situation": "Set the scene in one sentence (where / when / who).",
                    "Task": "Say what you were responsible for or trying to achieve.",
                    "Action": "Describe what YOU did, step by step, using 'I built / tested / decided...'.",
                    "Result": "Finish with the outcome, ideally with a number or concrete change.",
                }[k])
        detail = {"STAR coverage": [k for k, ok in hit.items() if ok]}
    else:
        kws = question.get("keywords", [])
        found = [k for k in kws if k in low]
        parts["structure"] = round(5 * len(found) / len(kws)) if kws else 0  # 5 pts
        missing = [k for k in kws if k not in low]
        if missing:
            tips.append("Key ideas not mentioned: " + ", ".join(missing[:5]) + ".")
        if not re.search(r"for example|for instance|e\.g\.|such as", low):
            tips.append("Add a concrete example ('for example...').")
        detail = {"key ideas covered": found}

    # numbers (2 pts), ownership (1 pt), fillers (1 pt)
    parts["numbers"] = min(2, numbers)
    if numbers == 0:
        tips.append("Add at least one number (time saved, accuracy, users, size of data).")
    parts["ownership"] = 1 if i_count >= 3 else 0
    if i_count < 3:
        tips.append("Use 'I' more: interviewers want to know what you personally did.")
    parts["fillers"] = 1 if fillers <= 2 else 0
    if fillers > 2:
        tips.append(f"{fillers} filler words found (like / basically / you know...). Cut them.")

    maximum = 2 + (4 if question["type"] == B else 5) + 2 + 1 + 1
    score = round(10 * sum(parts.values()) / maximum, 1)
    return {"score": score, "words": wc, "numbers": numbers, "fillers": fillers,
            "parts": parts, "detail": detail, "tips": tips}


def get_questions(role: str, n: int, jd: str | None) -> list[dict]:
    if jd and llm.is_live():
        try:
            prompt = (
                f"Write {n} interview questions for a fresher applying for a {role} role, based on this job description.\n"
                "Mix behavioral and technical. Return ONLY JSON: "
                '[{"type": "behavioral" or "technical", "q": "...", "keywords": ["3-8 lowercase key ideas a good technical answer mentions; [] for behavioral"]}]\n\n'
                f"JOB DESCRIPTION:\n{jd}"
            )
            data = llm.extract_json(llm.complete(prompt, max_tokens=1200, temperature=0.5).text)
            qs = [d for d in data if d.get("type") in (B, T) and d.get("q")][:n]
            if qs:
                return [{"type": d["type"], "q": d["q"], "keywords": [k.lower() for k in d.get("keywords", [])]} for d in qs]
        except (llm.LLMError, ValueError, TypeError, AttributeError) as exc:
            print(f"[warn] question generation failed ({exc}); using built-in bank.")
    return BANK[role][:n]


def coach(question: dict, answer: str) -> dict | None:
    if not llm.is_live():
        return None
    prompt = (
        "You are a supportive interview coach for a fresher. Review the answer below.\n"
        'Return ONLY JSON: {"strengths": ["..."], "improvements": ["..."], "stronger_answer_outline": "3-4 sentences using only facts the candidate already gave"}\n'
        "Do not invent experience the candidate did not mention.\n\n"
        f"QUESTION ({question['type']}): {question['q']}\n\nANSWER:\n{answer}"
    )
    try:
        return llm.extract_json(llm.complete(prompt, max_tokens=700, temperature=0.3).text)
    except (llm.LLMError, ValueError):
        return None


def write_report(out: Path, role: str, mode: str, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "session_report.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    scored = [r for r in rows if r["feedback"]]
    avg = round(sum(r["feedback"]["score"] for r in scored) / len(scored), 1) if scored else 0
    L = [f"# Interview practice report: {role}", "", f"Mode: {mode}  |  Questions answered: {len(scored)}  |  Average rubric score: **{avg}/10**", ""]
    for i, r in enumerate(rows, 1):
        L += [f"## Q{i} ({r['type']}): {r['question']}", ""]
        if not r["feedback"]:
            L += ["_Skipped._", ""]
            continue
        f = r["feedback"]
        L += [f"**Score: {f['score']}/10**  ({f['words']} words, {f['numbers']} numbers, {f['fillers']} fillers)", "",
              "> " + r["answer"].replace("\n", " "), ""]
        if f["tips"]:
            L += ["Rubric tips:"] + [f"- {t}" for t in f["tips"]] + [""]
        c = r.get("coach")
        if c:
            L += ["AI coach:"] + [f"- Strength: {s}" for s in c.get("strengths", [])]
            L += [f"- Improve: {s}" for s in c.get("improvements", [])]
            L += [f"- Stronger outline: {c.get('stronger_answer_outline', '')}", ""]
    (out / "session_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="ai-prompt-engineer", choices=sorted(BANK))
    ap.add_argument("--n", type=int, default=4, help="number of questions")
    ap.add_argument("--jd", help="optional job description file (AI mode generates tailored questions)")
    ap.add_argument("--answers", help="JSON list of answers for non-interactive mode")
    ap.add_argument("--out", default="output")
    ap.add_argument("--offline", action="store_true", help="force offline rubric-only mode")
    args = ap.parse_args()
    if args.offline:
        os.environ["LLM_PROVIDER"] = "mock"

    jd = Path(args.jd).read_text(encoding="utf-8") if args.jd else None
    questions = get_questions(args.role, args.n, jd)
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8")) if args.answers else None
    mode = f"rubric + AI coach ({llm.model_name()})" if llm.is_live() else "offline rubric only"
    print(f"Role: {args.role} | Mode: {mode}\n")

    rows = []
    for i, q in enumerate(questions):
        print(f"Q{i + 1} [{q['type']}] {q['q']}")
        if answers is not None:
            ans = answers[i] if i < len(answers) else ""
            if ans:
                print(f"   (answer from file) {ans[:90]}{'...' if len(ans) > 90 else ''}")
        else:
            ans = input("Your answer (type 'skip' to skip): ").strip()
            if ans.lower() == "skip":
                ans = ""
        if not ans:
            rows.append({"question": q["q"], "type": q["type"], "answer": "", "feedback": None})
            print()
            continue
        fb = analyze(q, ans)
        row = {"question": q["q"], "type": q["type"], "answer": ans, "feedback": fb, "coach": coach(q, ans)}
        rows.append(row)
        print(f"   Score: {fb['score']}/10")
        for t in fb["tips"][:3]:
            print(f"   - {t}")
        print()

    write_report(Path(args.out), args.role, mode, rows)
    print(f"Report saved to {args.out}/session_report.md")


if __name__ == "__main__":
    main()
