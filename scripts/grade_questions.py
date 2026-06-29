#!/usr/bin/env python3
"""Grade the QA-style grading_questions.json using the repo's answer_node.

Produces outputs/grading_metrics.json with per-question results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph_agent_lab.nodes import answer_node
from langgraph_agent_lab.state import AgentState


def load_questions(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def grade_one(q: dict[str, Any]) -> dict[str, Any]:
    state: AgentState = {
        "thread_id": f"grade-{q.get('id')}",
        "scenario_id": q.get("id"),
        "query": q.get("question"),
        "route": "simple",
        "risk_level": "low",
        "attempt": 0,
        "max_attempts": 1,
        "final_answer": None,
        "evaluation_result": False,
        "pending_question": None,
        "proposed_action": {},
        "approval": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }

    out = answer_node(state)
    final = out.get("final_answer") or ""
    # simple normalization
    text = final.lower()

    must_any = q.get("must_contain_any", [])
    must_not = q.get("must_not_contain", [])
    expect_doc = q.get("expect_top1_doc_id")

    # Criterion checks
    if must_any:
        passed_any = any(tok.lower() in text for tok in must_any)
    else:
        passed_any = True

    passed_not = all(tok.lower() not in text for tok in must_not) if must_not else True

    passed_doc = True
    if expect_doc:
        # best-effort: check if expected doc id token appears in the answer text
        passed_doc = expect_doc.lower() in text

    # Weighted scoring: must_any (0.6), must_not (0.2), expect_doc (0.2)
    w_any = 0.6
    w_not = 0.2
    w_doc = 0.2
    score = (w_any * (1.0 if passed_any else 0.0)) + (w_not * (1.0 if passed_not else 0.0)) + (w_doc * (1.0 if passed_doc else 0.0))

    return {
        "id": q.get("id"),
        "question": q.get("question"),
        "answer": final,
        "score": round(score, 3),
        "passed_any": passed_any,
        "passed_not": passed_not,
        "passed_doc": passed_doc,
        "must_contain_any": must_any,
        "must_not_contain": must_not,
        "expect_top1_doc_id": expect_doc,
    }


def main():
    repo = Path(__file__).resolve().parents[1]
    data_path = repo / "data" / "grading_questions.json"
    out_path = repo / "outputs" / "grading_metrics.json"
    questions = load_questions(data_path)
    results = [grade_one(q) for q in questions]
    # consider a question 'passed' if score >= 0.7
    threshold = 0.7
    passed_count = sum(1 for r in results if float(r.get("score", 0)) >= threshold)
    avg_score = sum(float(r.get("score", 0)) for r in results) / len(results) if results else 0.0
    out = {
        "total": len(results),
        "results": results,
        "summary": {"passed_count": passed_count, "threshold": threshold, "avg_score": round(avg_score, 3)},
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote grading metrics to {out_path}")


if __name__ == "__main__":
    main()
