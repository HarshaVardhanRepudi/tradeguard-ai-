"""
TradeGuard AI — Local Agent Stand-In (NOT the real agent)
-------------------------------------------------------------
This is NOT an LLM. It's a simple keyword-based router that mimics the
SHAPE of agentic tool-selection (look at the question, decide which
tools to call, call them in sequence, assemble an answer) so you can see
and test that behavior right now, without internet access or an API key.

Use this to verify the tool-calling PATTERN works end-to-end. The real
decision-making — actually understanding an arbitrary question and
choosing tools intelligently — only happens in agent/llm_agent.py, which
requires the real Claude API.

Run:  python3 agent/mock_router.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import tools as t


def route(question: str) -> dict:
    """Very simple pattern matching — this is the part a real LLM replaces."""
    q = question.lower()
    trace = []

    # Extract a team code if present (crude, real LLM would just understand it)
    team_match = re.search(r"\b(BOS|LAL|GSW|CLE|DAL|PHX|MIA|MIL|OKC|NYK|DEN)\b", question.upper())
    team = team_match.group(1) if team_match else None

    if "risk" in q or "why" in q and "trade" in q:
        # "why is this risky" style -> needs the full chain
        trace.append("Decision: question asks about risk/explanation -> calling explain_trade (full chain)")
        if team:
            result = t.explain_trade(team, outgoing_salaries=[10_000_000], incoming_salaries=[11_500_000],
                                       question=question)
            trace.append(f"  explain_trade({team}, ...) -> permitted={result['permitted']}, "
                         f"risk={result['ml_risk']['risk_score']}")
            return {"trace": trace, "result": result}

    if "cap space" in q or "payroll" in q or "apron status" in q:
        trace.append(f"Decision: simple factual lookup -> calling get_team_cap_status only")
        if team:
            result = t.get_team_cap_status(team)
            trace.append(f"  get_team_cap_status({team}) -> {result}")
            return {"trace": trace, "result": result}

    if "rule" in q or "cba" in q or "apron restriction" in q:
        trace.append("Decision: rules question -> calling retrieve_rules only (no DB lookup needed)")
        result = t.retrieve_rules(question, k=2)
        trace.append(f"  retrieve_rules(...) -> {len(result)} chunks retrieved")
        return {"trace": trace, "result": result}

    if "trade" in q and team:
        trace.append("Decision: trade legality question -> calling evaluate_trade (DB lookup + rules engine)")
        result = t.evaluate_trade(team, outgoing_salaries=[10_000_000], incoming_salaries=[11_500_000])
        trace.append(f"  evaluate_trade({team}, ...) -> permitted={result['permitted']}")
        return {"trace": trace, "result": result}

    trace.append("Decision: no clear tool match — a real LLM would ask a clarifying question here")
    return {"trace": trace, "result": None}


if __name__ == "__main__":
    test_questions = [
        "What's BOS's current cap space?",
        "What CBA rule restricts second apron trades?",
        "Can CLE make a legal trade right now?",
        "Why would a trade for GSW be risky?",
    ]
    for q in test_questions:
        print(f"\nQUESTION: {q}")
        out = route(q)
        for line in out["trace"]:
            print(f"  {line}")
