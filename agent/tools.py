"""
TradeGuard AI — Agent Tools
----------------------------
These are the functions the agent calls. Each one is a single, narrow
capability (DB lookup, or the salary-matching calculation) — the agent's
job is to decide which ones to call and in what order, not to contain
this logic itself.

Run directly for a demo:  python3 agent/tools.py
"""

import sqlite3
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etl.salary_matching import TradeProposal, ApronStatus, check_salary_match
from rag.retrieve import retrieve_rules  # re-exported so agent/llm_agent.py can dispatch to it by name
from ml.fit_score import recommend_trade_targets

DB_PATH = Path(__file__).resolve().parent.parent / "etl" / "tradeguard.db"
MODEL_PATH = Path(__file__).resolve().parent.parent / "ml" / "risk_model.pkl"
APRON_ORDER = ["under_cap", "over_cap", "over_tax", "first_apron", "second_apron"]

_model_cache = None

def _load_model():
    global _model_cache
    if _model_cache is None:
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
    return _model_cache


def predict_trade_risk(apron_status: str, n_outgoing: int, outgoing_total: float, incoming_total: float) -> dict:
    """Tool 5: ML-predicted risk score for a trade, independent of the deterministic engine."""
    bundle = _load_model()
    clf = bundle["model"]
    apron_rank = APRON_ORDER.index(apron_status) if apron_status in APRON_ORDER else 0
    ratio = round(incoming_total / max(outgoing_total, 1), 3)
    X = [[apron_rank, n_outgoing, outgoing_total, incoming_total, ratio]]
    proba_approved = clf.predict_proba(X)[0][1]
    return {
        "risk_score": round(float(1 - proba_approved), 3),  # risk = probability of rejection
        "predicted_outcome": "approved" if proba_approved >= 0.5 else "rejected",
        "model_confidence": round(float(max(proba_approved, 1 - proba_approved)), 3),
    }


def _conn():
    return sqlite3.connect(DB_PATH)


def get_team_cap_status(team_id: str) -> dict:
    """Tool 1: look up a team's current payroll and apron status."""
    conn = _conn()
    row = conn.execute(
        "SELECT team_id, name, total_payroll, cap_space, apron_status, season "
        "FROM teams WHERE team_id = ?", (team_id,)
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"Unknown team_id '{team_id}'"}
    return {
        "team_id": row[0], "name": row[1], "total_payroll": row[2],
        "cap_space": row[3], "apron_status": row[4], "season": row[5],
    }


def get_player_salary(player_name: str) -> dict:
    """Tool 2: look up a player's current salary by name (fuzzy-ish match)."""
    conn = _conn()
    row = conn.execute(
        "SELECT p.name, p.team_id, c.salary, c.years_remaining "
        "FROM players p JOIN contracts c ON p.player_id = c.player_id "
        "WHERE p.name LIKE ?", (f"%{player_name}%",)
    ).fetchone()
    conn.close()
    if not row:
        return {"error": f"No player found matching '{player_name}'"}
    return {"name": row[0], "team_id": row[1], "salary": row[2], "years_remaining": row[3]}


def evaluate_trade(team_from: str, outgoing_salaries: list, incoming_salaries: list) -> dict:
    """
    Tool 3 (the composite one): look up team_from's real apron status from the
    DB, then run it through the salary-matching engine. This is the function
    that answers "is this trade legal?"
    """
    status = get_team_cap_status(team_from)
    if "error" in status:
        return status

    trade = TradeProposal(
        team_from=team_from,
        team_to="",
        outgoing_salaries=outgoing_salaries,
        incoming_salaries=incoming_salaries,
        team_from_apron_status=ApronStatus(status["apron_status"])
        if status["apron_status"] in ApronStatus._value2member_map_
        else ApronStatus.UNDER_CAP,
    )
    result = check_salary_match(trade)
    return {
        "team_from": team_from,
        "team_from_apron_status": status["apron_status"],
        "permitted": result.permitted,
        "reason": result.reason,
        "rule_applied": result.rule_applied,
        "citation": result.citation,
    }


def explain_trade(team_from: str, outgoing_salaries: list, incoming_salaries: list, question: str = None) -> dict:
    """
    Tool 4 (the full agent chain): evaluate the trade AND retrieve the actual
    rule text that backs the verdict, so the answer is grounded in retrieved
    source text rather than just a citation string embedded in code.
    """
    verdict = evaluate_trade(team_from, outgoing_salaries, incoming_salaries)
    if "error" in verdict:
        return verdict

    query = question or f"salary matching rule for a team that is {verdict['team_from_apron_status']}"
    retrieved = retrieve_rules(query, k=2)

    risk = predict_trade_risk(
        apron_status=verdict["team_from_apron_status"],
        n_outgoing=len(outgoing_salaries),
        outgoing_total=sum(outgoing_salaries),
        incoming_total=sum(incoming_salaries),
    )

    return {
        **verdict,
        "retrieved_rules": retrieved,
        "ml_risk": risk,
    }


def recommend_legal_targets(team_id: str, needed_position: str, salary_min: float,
                              salary_max: float, giving_up_salary: float, top_k: int = 3) -> dict:
    """
    Tool 6 (the merge): rank candidate players by fit for a team's need, AND
    check whether trading for each one would actually be legal given the
    team's real apron status. This is the CricketIQ 'what should I do' layer
    combined with TradeGuard's compliance checking — one recommendation that
    is both good AND legal, not just good.
    """
    status = get_team_cap_status(team_id)
    if "error" in status:
        return status

    candidates = recommend_trade_targets(team_id, needed_position, salary_min, salary_max, top_k=10)

    results = []
    for c in candidates:
        trade = TradeProposal(
            team_from=team_id, team_to=c.team_id,
            outgoing_salaries=[giving_up_salary], incoming_salaries=[c.salary],
            team_from_apron_status=ApronStatus(status["apron_status"]),
        )
        match = check_salary_match(trade)
        results.append({
            "name": c.name, "team_id": c.team_id, "salary": c.salary,
            "fit_score": c.fit_score, "fit_breakdown": c.breakdown,
            "trade_legal": match.permitted, "legal_reason": match.reason,
        })
        if len(results) >= top_k:
            break

    legal_only = [r for r in results if r["trade_legal"]]
    return {
        "team_id": team_id, "apron_status": status["apron_status"],
        "candidates_checked": len(results),
        "legal_recommendations": legal_only,
        "all_candidates": results,
    }


if __name__ == "__main__":
    print("=== Tool 1: get_team_cap_status ===")
    print(get_team_cap_status("BOS"))
    print(get_team_cap_status("MIA"))

    print("\n=== Tool 3: evaluate_trade — BOS (2nd apron) tries to send 2 players for 1 ===")
    r = evaluate_trade("BOS", outgoing_salaries=[18_500_000, 9_800_000], incoming_salaries=[26_000_000])
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== evaluate_trade — LAL (1st apron) one-for-one, $15M for $18M ===")
    r = evaluate_trade("LAL", outgoing_salaries=[15_400_000], incoming_salaries=[17_000_000])
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== evaluate_trade — MIA (under cap) one-for-one, $10M for $12M ===")
    r = evaluate_trade("MIA", outgoing_salaries=[11_900_000], incoming_salaries=[12_400_000])
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== Tool 4: explain_trade — full agent chain (verdict + retrieved rule text) ===")
    r = explain_trade("BOS", outgoing_salaries=[18_500_000, 9_800_000], incoming_salaries=[26_000_000],
                       question="Can a second apron team combine two contracts in a trade?")
    print(f"  Verdict: {'PERMITTED' if r['permitted'] else 'NOT PERMITTED'}")
    print(f"  Reason: {r['reason']}")
    print(f"  Retrieved rules backing this verdict:")
    for rule in r["retrieved_rules"]:
        print(f"    [{rule['relevance']}] {rule['citation']}: {rule['text'][:100]}...")
