"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

from .state import AgentState, make_event, ApprovalDecision
from .llm import get_llm
import os
try:
    import openai
except Exception:
    openai = None
import json


def _normalize_route(raw_route: str, risk_level: str, query: str) -> str:
    if not raw_route:
        raw_route = ""
    r = raw_route.strip().lower()
    q = (query or "").lower()
    # direct allowed routes
    allowed = {"risky", "tool", "missing_info", "error", "simple"}
    if r in allowed:
        # If model returned 'tool' but the query contains explicit risky keywords,
        # prefer 'risky' to avoid misrouting (e.g., refund requests).
        if r == "tool":
            risk_words = ["refund", "delete", "remove", "cancel"]
            if any(w in q for w in risk_words):
                return "risky"
        # If model returned 'risky' but the query looks like a system error, prefer 'error'.
        if r == "risky":
            error_words = ["timeout", "failure", "cannot recover", "cannot recover", "crash"]
            if any(w in q for w in error_words):
                return "error"
        return r
    # common synonyms mapping
    synonyms = {
        "order_status": "tool",
        "order-status": "tool",
        "order status": "tool",
        "status": "tool",
        "lookup": "tool",
        "refund": "risky",
        "refund_confirmation": "risky",
        "refund-confirmation": "risky",
        "delete": "risky",
        "remove": "risky",
        "support": "simple",
        "unknown": "missing_info",
        "high": "risky",
        "low": "simple",
    }
    if r in synonyms:
        return synonyms[r]
    # if only risk_level returned
    if r in ("high", "low"):
        return "risky" if r == "high" else "simple"
    # infer from query keywords as last resort
    risk_words = ["refund", "delete", "remove", "cancel"]
    tool_words = ["status", "order", "track", "lookup", "search"]
    error_words = ["timeout", "failure", "error", "crash"]
    if any(w in q for w in risk_words):
        return "risky"
    if any(w in q for w in tool_words):
        return "tool"
    if any(w in q for w in error_words):
        return "error"
    if len(q.split()) < 3:
        return "missing_info"
    return "simple"


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")

    # Build a JSON-output prompt so parsing is deterministic
    json_prompt = (
        "Classify the user's support query into one of the canonical routes: 'risky', 'tool', 'missing_info', 'error', 'simple'.\n"
        "Priority (if multiple apply): risky > tool > missing_info > error > simple.\n"
        "REPLY STRICTLY WITH A SINGLE JSON OBJECT AND NOTHING ELSE. No explanation, no backticks, no markdown.\n"
        "The JSON schema MUST be: {\n  \"route\": <one of ['risky','tool','missing_info','error','simple']>,\n  \"risk_level\": <'high' or 'low'>\n}\n"
        "If unsure, set 'route' to 'missing_info' and 'risk_level' to 'low'.\n"
        "Example output exactly (including quotes and braces):\n"
        "{\"route\": \"tool\", \"risk_level\": \"low\"}\n\n"
        f"User query: {query}"
    )

    # 1) Prefer direct OpenAI SDK when available (ensures deterministic output)
    if openai and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": json_prompt}])
            text = getattr(resp.choices[0].message, "content", None) or resp.choices[0].message
            text = text if isinstance(text, str) else str(text)
            # attempt to locate JSON substring
            try:
                payload = json.loads(text)
            except Exception:
                # try to extract JSON block
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    try:
                        payload = json.loads(text[start:end+1])
                    except Exception:
                        payload = None
                else:
                    payload = None
            if payload:
                raw_route = str(payload.get("route", "")).lower()
                risk_level = str(payload.get("risk_level", "low")).lower()
                route = _normalize_route(raw_route, risk_level, query)
                ev_msg = f"openai classified raw={raw_route} -> route={route}"
                return {"route": route, "risk_level": risk_level, "events": [make_event("classify", "completed", ev_msg)]}
        except Exception:
            pass

    # 2) Try LangChain LLM and request JSON output
    try:
        llm = get_llm(temperature=0.0)
        raw = None
        try:
            raw = llm(json_prompt)
        except Exception:
            try:
                raw = llm.predict(json_prompt)
            except Exception:
                try:
                    raw = llm.generate([json_prompt])
                except Exception:
                    raw = None
        if raw is not None:
            try:
                raw_text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
            except Exception:
                raw_text = str(raw)
            # parse JSON
            try:
                payload = json.loads(raw_text)
            except Exception:
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1:
                    try:
                        payload = json.loads(raw_text[start:end+1])
                    except Exception:
                        payload = None
                else:
                    payload = None
            if payload:
                raw_route = str(payload.get("route", "")).lower()
                risk_level = str(payload.get("risk_level", "low")).lower()
                route = _normalize_route(raw_route, risk_level, query)
                ev_msg = f"llm classified raw={raw_route} -> route={route}"
                return {"route": route, "risk_level": risk_level, "events": [make_event("classify", "completed", ev_msg)]}
    except Exception:
        pass

    # 3) Heuristic fallback (last resort)
    q = query.lower()
    route = "simple"
    risk_words = ["refund", "delete", "send email", "cancel", "remove"]
    tool_words = ["lookup", "order", "status", "track", "search"]
    error_words = ["timeout", "failure", "error", "crash"]
    if any(w in q for w in risk_words):
        route = "risky"
    elif any(w in q for w in tool_words):
        route = "tool"
    elif any(w in q for w in error_words):
        route = "error"
    elif len(q.split()) < 3 or q.strip().endswith("fix it"):
        route = "missing_info"
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [make_event("classify", "fallback", f"heuristic route={route}")],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0))
    route = state.get("route", "")
    scenario = state.get("scenario_id", "unknown")
    if route == "error" and attempt < 2:
        result = f"ERROR: simulated transient failure (attempt={attempt})"
        ev_msg = "tool failed"
    else:
        result = f"OK: tool result for {scenario}"
        ev_msg = "tool succeeded"
    return {
        "tool_results": [result],
        "events": [make_event("tool_node", "completed", ev_msg, result=result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    results = state.get("tool_results", [])
    latest = results[-1] if results else ""
    # Simple heuristic: if latest contains ERROR, mark as needs retry
    needs_retry = "ERROR" in str(latest)
    evaluation_result = "needs_retry" if needs_retry else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event("evaluate_node", "completed", f"evaluation={evaluation_result}", latest=latest)
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    context_parts = [f"Query: {query}"]
    if tool_results:
        context_parts.append(f"Tool results: {tool_results[-1]}")
    if approval:
        context_parts.append(f"Approval: {approval}")
    context = "\n".join(context_parts)

    try:
        llm = get_llm(temperature=0.0)
        prompt = (
            "You are a helpful support agent. Generate a concise, accurate, and grounded response "
            "based on the available context below. Do not hallucinate.\n\n"
            f"{context}\n\nRespond:"
        )

        raw = None
        try:
            raw = llm(prompt)
        except Exception:
            try:
                raw = llm.predict(prompt)
            except Exception:
                try:
                    raw = llm.generate([prompt])
                except Exception:
                    raw = None

        if raw is not None:
            try:
                final = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
            except Exception:
                final = str(raw)
            return {
                "final_answer": final,
                "events": [make_event("answer", "completed", "llm generated answer")],
            }
    except Exception:
        pass

    # Try direct OpenAI SDK as a fallback if available
    if openai and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
            resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": context}])
            final = getattr(resp.choices[0].message, "content", None) or resp.choices[0].message
            final = final if isinstance(final, str) else str(final)
            return {"final_answer": final, "events": [make_event("answer", "completed", "openai generated answer")]}
        except Exception:
            pass

    # Fallback: craft a simple grounded answer
    parts = [f"Answer to: {query}"]
    if tool_results:
        parts.append(f"Tool: {tool_results[-1]}")
    if approval:
        parts.append(f"Approval: {approval}")
    final = " -- ".join(parts)
    return {
        "final_answer": final,
        "events": [make_event("answer", "fallback", "answered without LLM")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    question = f"Can you provide more details about: '{query}'? What specifically do you need?"
    return {
        "pending_question": question,
        "final_answer": None,
        "events": [make_event("ask_clarification", "requested", "clarification asked")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    action = {"description": f"Proposed action for query: {query}", "requires_approval": True}
    return {
        "proposed_action": action,
        "events": [make_event("risky_action", "prepared", "proposed risky action", action=action)],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str}, "events": [make_event(...)]}
    """
    # Default mock approval to allow offline tests to pass
    decision = ApprovalDecision(approved=True, reviewer="mock-reviewer", comment="auto-approved")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", "mock approval", approved=decision.approved)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    error_msg = f"retry triggered, new attempt={attempt}"
    return {
        "attempt": attempt,
        "errors": [error_msg],
        "events": [make_event("retry_or_fallback", "retry", error_msg, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    final = "Request could not be completed after maximum retry attempts."
    return {
        "final_answer": final,
        "events": [make_event("dead_letter", "completed", "moved to dead letter")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
