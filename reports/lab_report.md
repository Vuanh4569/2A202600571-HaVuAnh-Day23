# Lab Report

## Metrics Summary

- Total scenarios: 7

- Success rate: 100.00%

- Avg nodes visited: 6.43

- Total retries: 0


## Per-scenario Results

| scenario_id | expected | actual | attempts | status |

|---|---|---|---:|---|
| S01_simple | simple | simple | 0 | PASS |

| S02_tool | tool | tool | 0 | PASS |

| S03_missing | missing_info | missing_info | 0 | PASS |

| S04_risky | risky | risky | 0 | PASS |

| S05_error | error | error | 0 | PASS |

| S06_delete | risky | risky | 0 | PASS |

| S07_dead_letter | error | error | 0 | PASS |


## Architecture

Implemented a state-driven LangGraph workflow with nodes: intake, classify, tool, evaluate, clarify, risky_action, approval, retry, dead_letter, answer, finalize. State includes evaluation_result, pending_question, proposed_action, approval, attempt counters and append-only event logs.


## Failure Analysis

- Transient tool failures → handled with bounded retry loop and dead-letter.

- Missing information from user → handled by clarification node to avoid hallucination.


## Improvements

- Use LLM-as-judge in evaluate_node for better retry gating.

- Persist state changes to SQLite and surface replay for crash recovery.

- Add HITL interrupt UI for real approvals.
