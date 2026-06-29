"""Report generation helper.

TODO(student): implement report rendering using MetricsReport data
and the template in reports/lab_report_template.md.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    TODO(student): Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    lines: list[str] = []
    lines.append(f"# Lab Report\n")
    lines.append(f"## Metrics Summary\n")
    lines.append(f"- Total scenarios: {metrics.total_scenarios}\n")
    lines.append(f"- Success rate: {metrics.success_rate:.2%}\n")
    lines.append(f"- Avg nodes visited: {metrics.avg_nodes_visited:.2f}\n")
    lines.append(f"- Total retries: {metrics.total_retries}\n")
    lines.append("\n## Per-scenario Results\n")
    lines.append("| scenario_id | expected | actual | attempts | status |\n")
    lines.append("|---|---|---|---:|---|")
    for r in metrics.scenario_metrics:
        lines.append(f"| {r.scenario_id} | {r.expected_route} | {r.actual_route} | {r.retry_count} | {'PASS' if r.success else 'FAIL'} |\n")

    lines.append("\n## Architecture\n")
    lines.append("Implemented a state-driven LangGraph workflow with nodes: intake, classify, tool, evaluate, clarify, risky_action, approval, retry, dead_letter, answer, finalize. State includes evaluation_result, pending_question, proposed_action, approval, attempt counters and append-only event logs.\n")

    lines.append("\n## Failure Analysis\n")
    lines.append("- Transient tool failures → handled with bounded retry loop and dead-letter.\n")
    lines.append("- Missing information from user → handled by clarification node to avoid hallucination.\n")

    lines.append("\n## Improvements\n")
    lines.append("- Use LLM-as-judge in evaluate_node for better retry gating.\n")
    lines.append("- Persist state changes to SQLite and surface replay for crash recovery.\n")
    lines.append("- Add HITL interrupt UI for real approvals.\n")

    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
