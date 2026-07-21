"""
TradeGuard AI — Player Fit Recommendation Engine
----------------------------------------------------
Answers "which players would be a good target for this team's need" —
the CricketIQ-style prescriptive layer, merged into TradeGuard.

Honest scope note: we don't have real performance stats (PPG, defensive
rating, etc.) loaded in this database yet — only position, salary, and
contract data. So "fit" here is deliberately defined narrowly and
transparently:

  fit_score = position_match_weight * position_match
            + salary_fit_weight     * salary_proximity
            + value_weight          * contract_efficiency

This is the same technique as CricketIQ's flaw_score: a transparent,
weighted formula over real features, not a trained black-box. Documented
here as a scoring FUNCTION, not a trained model — if real performance
data is added later, this is the file where a genuine trained model
would replace the formula, following the same eval discipline as
ml/train_model.py (train/test split, real metrics).

Run:  python3 ml/fit_score.py
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "etl" / "tradeguard.db"

POSITION_GROUPS = {
    "PG": {"PG", "G"}, "SG": {"SG", "G"}, "SF": {"SF", "F"},
    "PF": {"PF", "F"}, "C": {"C"},
}

WEIGHTS = {"position": 0.5, "salary_fit": 0.35, "value": 0.15}


@dataclass
class Candidate:
    name: str
    team_id: str
    position: str
    salary: float
    fit_score: float
    breakdown: dict


def _position_match(candidate_pos: str, needed_pos: str) -> float:
    if candidate_pos == needed_pos:
        return 1.0
    group = POSITION_GROUPS.get(needed_pos, set())
    if candidate_pos in group:
        return 0.6
    return 0.15


def _salary_proximity(salary: float, target_min: float, target_max: float) -> float:
    if target_min <= salary <= target_max:
        return 1.0
    midpoint = (target_min + target_max) / 2
    span = max(target_max - target_min, 1)
    distance = abs(salary - midpoint) / span
    return max(0.0, 1.0 - distance * 0.5)


def _contract_efficiency(salary: float, target_max: float) -> float:
    """Cheaper-than-budget-ceiling contracts score slightly higher — a proxy
    for 'value', since we don't have real performance stats to judge value
    against production."""
    if target_max <= 0:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (salary / target_max) * 0.3))


def recommend_trade_targets(exclude_team: str, needed_position: str,
                              target_salary_min: float, target_salary_max: float,
                              top_k: int = 5) -> list:
    """
    Tool: rank real players from the database (excluding the requesting
    team's own roster) by fit for a stated need.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT p.name, p.team_id, p.position, c.salary FROM players p "
        "JOIN contracts c ON p.player_id = c.player_id WHERE p.team_id != ?",
        (exclude_team,)
    ).fetchall()
    conn.close()

    candidates = []
    for name, team_id, position, salary in rows:
        pos_score = _position_match(position, needed_position)
        sal_score = _salary_proximity(salary, target_salary_min, target_salary_max)
        val_score = _contract_efficiency(salary, target_salary_max)

        total = (WEIGHTS["position"] * pos_score +
                 WEIGHTS["salary_fit"] * sal_score +
                 WEIGHTS["value"] * val_score)

        candidates.append(Candidate(
            name=name, team_id=team_id, position=position, salary=salary,
            fit_score=round(total, 3),
            breakdown={"position_match": round(pos_score, 2),
                       "salary_fit": round(sal_score, 2),
                       "value": round(val_score, 2)},
        ))

    candidates.sort(key=lambda c: c.fit_score, reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    print("=== BOS needs a Center, budget $8M-$14M ===")
    results = recommend_trade_targets("BOS", "C", 8_000_000, 14_000_000, top_k=5)
    for c in results:
        print(f"  {c.fit_score:.3f}  {c.name:22s} ({c.team_id}, {c.position})  "
              f"${c.salary:,.0f}  breakdown={c.breakdown}")

    print("\n=== CLE needs a Point Guard, budget $20M-$35M ===")
    results = recommend_trade_targets("CLE", "PG", 20_000_000, 35_000_000, top_k=5)
    for c in results:
        print(f"  {c.fit_score:.3f}  {c.name:22s} ({c.team_id}, {c.position})  "
              f"${c.salary:,.0f}  breakdown={c.breakdown}")
