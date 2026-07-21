"""
TradeGuard AI — Training Data Generator
------------------------------------------
Builds trade_history.csv with two sources of rows:

1. SYNTHETIC (bulk): randomly generated trades across all apron tiers,
   labeled by running them through our already-verified salary_matching.py
   engine. This is NOT historical ground truth — it's a rule-grounded
   approximation, generated so a classifier can learn to predict the
   engine's own logic from raw features (useful for speed/generalization,
   not for claiming "real outcomes"). Labeled honestly as such.

2. REAL (anchor examples): a handful of actual trades from the 2025-26
   trade deadline (source: ESPN trade tracker, Feb 2026), labeled
   'approved' because they were reported as completed. Salary figures are
   sourced from public reporting where available; used as sanity-check
   anchors, not as the bulk of the training signal.

Run:  python3 ml/generate_training_data.py
"""

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etl.salary_matching import TradeProposal, ApronStatus, check_salary_match

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "trade_history_seed.csv"

random.seed(42)

APRON_TIERS = [
    ApronStatus.UNDER_CAP, ApronStatus.OVER_CAP, ApronStatus.OVER_TAX,
    ApronStatus.FIRST_APRON, ApronStatus.SECOND_APRON,
]

def random_trade():
    status = random.choice(APRON_TIERS)
    n_outgoing = random.choice([1, 1, 1, 2, 2, 3])  # weighted toward 1-for-1
    outgoing = [round(random.uniform(1_000_000, 50_000_000), -3) for _ in range(n_outgoing)]
    outgoing_total = sum(outgoing)

    # Sample incoming salary across a wide range relative to outgoing so we
    # get both permitted and rejected examples, not all-permitted.
    ratio = random.uniform(0.5, 1.8)
    incoming_total = round(outgoing_total * ratio, -3)
    incoming = [incoming_total]  # single incoming contract for simplicity

    return TradeProposal(
        team_from="SIM", team_to="SIM2",
        outgoing_salaries=outgoing, incoming_salaries=incoming,
        team_from_apron_status=status,
    )


def generate_synthetic(n=400):
    rows = []
    for i in range(n):
        trade = random_trade()
        result = check_salary_match(trade)
        rows.append({
            "trade_id": f"SYN{i:04d}",
            "source": "synthetic",
            "team_from_apron_status": trade.team_from_apron_status.value,
            "n_outgoing_contracts": len(trade.outgoing_salaries),
            "outgoing_total": result.outgoing_total,
            "incoming_total": result.incoming_total,
            "salary_ratio": round(result.incoming_total / max(result.outgoing_total, 1), 3),
            "outcome": "approved" if result.permitted else "rejected",
        })
    return rows


def real_anchor_trades():
    """
    Real trades from the 2025-26 season trade deadline (Feb 2026), per ESPN's
    trade tracker. All labeled 'approved' since they were reported completed.
    Salary figures are best-effort from public reporting / our own loaded DB
    where the player overlaps (e.g. James Harden's real 2025-26 salary).
    """
    return [
        {
            "trade_id": "REAL001", "source": "real",
            "note": "Clippers-Cavaliers swap: Harden traded to CLE for Garland (Feb 2026)",
            "team_from_apron_status": "second_apron",  # CLE was over 2nd apron
            "n_outgoing_contracts": 1,
            "outgoing_total": 39_446_090,  # Harden's real cap hit (loaded in our DB)
            "incoming_total": 39_600_000,  # Garland's approx. reported salary
            "salary_ratio": round(39_600_000 / 39_446_090, 3),
            "outcome": "approved",
        },
        {
            "trade_id": "REAL002", "source": "real",
            "note": "Warriors-Hawks: Kuminga + Hield to ATL for Porzingis (Feb 2026)",
            "team_from_apron_status": "first_apron",  # GSW was over 1st apron
            "n_outgoing_contracts": 2,
            "outgoing_total": 25_500_000,  # approx combined Kuminga+Hield
            "incoming_total": 24_500_000,  # approx Porzingis salary at trade time
            "salary_ratio": round(24_500_000 / 25_500_000, 3),
            "outcome": "approved",
        },
        {
            "trade_id": "REAL003", "source": "real",
            "note": "Bulls-Celtics: Vucevic to CHI for Simons (Feb 2026)",
            "team_from_apron_status": "over_cap",
            "n_outgoing_contracts": 1,
            "outgoing_total": 21_500_000,  # approx Vucevic salary
            "incoming_total": 27_700_000,  # approx Simons salary
            "salary_ratio": round(27_700_000 / 21_500_000, 3),
            "outcome": "approved",
        },
        {
            "trade_id": "REAL004", "source": "real",
            "note": "Hawks-Wizards: Trae Young to WAS for McCollum + Kispert (Jan 2026)",
            "team_from_apron_status": "over_tax",
            "n_outgoing_contracts": 1,
            "outgoing_total": 43_000_000,  # approx Young salary
            "incoming_total": 33_000_000,  # approx McCollum+Kispert combined
            "salary_ratio": round(33_000_000 / 43_000_000, 3),
            "outcome": "approved",
        },
        {
            "trade_id": "REAL005", "source": "real",
            "note": "Mavericks-Wizards: Anthony Davis to WAS (large deal, Feb 2026)",
            "team_from_apron_status": "over_cap",
            "n_outgoing_contracts": 1,
            "outgoing_total": 54_000_000,  # approx Davis salary
            "incoming_total": 52_000_000,  # approx return package total
            "salary_ratio": round(52_000_000 / 54_000_000, 3),
            "outcome": "approved",
        },
    ]


def main():
    synthetic = generate_synthetic(400)
    real = real_anchor_trades()

    fieldnames = ["trade_id", "source", "team_from_apron_status", "n_outgoing_contracts",
                  "outgoing_total", "incoming_total", "salary_ratio", "outcome"]

    with open(OUT_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in synthetic + real:
            w.writerow(row)

    n_approved = sum(1 for r in synthetic + real if r["outcome"] == "approved")
    n_rejected = sum(1 for r in synthetic + real if r["outcome"] == "rejected")
    print(f"Wrote {len(synthetic)} synthetic + {len(real)} real rows -> {OUT_PATH}")
    print(f"Class balance: {n_approved} approved / {n_rejected} rejected")


if __name__ == "__main__":
    main()
