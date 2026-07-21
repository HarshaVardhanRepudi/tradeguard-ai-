"""
TradeGuard AI — ETL Loader
---------------------------
Reads raw CSVs -> validates -> computes derived features (team payroll,
cap space, apron status) -> writes to SQLite (etl/tradeguard.db).

Run:  python3 etl/load.py
"""

import csv
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
DB_PATH = BASE / "etl" / "tradeguard.db"
SCHEMA_PATH = BASE / "etl" / "schema.sql"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def validate_contracts(rows):
    """Basic data validation — reject bad rows, don't silently corrupt the DB."""
    clean, rejected = [], []
    seen_ids = set()
    for r in rows:
        try:
            salary = float(r["salary"])
            years = int(r["years_remaining"])
            if salary <= 0 or years < 0:
                raise ValueError("non-positive salary or negative years")
            if r["player_id"] in seen_ids:
                raise ValueError("duplicate player_id")
            seen_ids.add(r["player_id"])
            clean.append(r)
        except (ValueError, KeyError) as e:
            rejected.append((r, str(e)))
    return clean, rejected


def apron_status_for(payroll, thresholds):
    if payroll >= thresholds["second_apron"]:
        return "second_apron"
    if payroll >= thresholds["first_apron"]:
        return "first_apron"
    if payroll >= thresholds["luxury_tax"]:
        return "over_tax"
    if payroll >= thresholds["salary_cap"]:
        return "over_cap"
    return "under_cap"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())

    # Full refresh: clear tables that are fully re-derived from the raw CSVs
    # each run. Without this, rows removed from a CSV (e.g. old placeholder
    # players swapped for real ones) become orphaned stale rows that never
    # get cleaned up, since INSERT OR REPLACE only overwrites matching
    # primary keys — it can't delete rows that no longer exist upstream.
    conn.execute("DELETE FROM contracts")
    conn.execute("DELETE FROM players")
    conn.execute("DELETE FROM teams")

    teams = read_csv(RAW / "teams_seed.csv")
    contracts_raw = read_csv(RAW / "contracts_seed.csv")
    thresholds_rows = read_csv(RAW / "league_thresholds_seed.csv")

    clean_contracts, rejected = validate_contracts(contracts_raw)
    print(f"Contracts loaded: {len(clean_contracts)}  |  Rejected: {len(rejected)}")
    for r, reason in rejected:
        print(f"  REJECTED {r.get('player_id', '?')}: {reason}")

    # league_thresholds
    for t in thresholds_rows:
        conn.execute(
            "INSERT OR REPLACE INTO league_thresholds VALUES (?,?,?,?,?)",
            (t["season"], float(t["salary_cap"]), float(t["luxury_tax"]),
             float(t["first_apron"]), float(t["second_apron"])),
        )
    thresholds = {k: float(v) for k, v in thresholds_rows[0].items() if k != "season"}

    # players + contracts
    payroll_by_team = {}
    for r in clean_contracts:
        conn.execute(
            "INSERT OR REPLACE INTO players VALUES (?,?,?,?)",
            (r["player_id"], r["name"], r["team_id"], r["position"]),
        )
        conn.execute(
            "INSERT OR REPLACE INTO contracts VALUES (?,?,?,?,?,?,?)",
            (f"C_{r['player_id']}", r["player_id"], r["season"], float(r["salary"]),
             int(r["years_remaining"]), int(r["no_trade_clause"]), float(r["trade_kicker_pct"])),
        )
        payroll_by_team[r["team_id"]] = payroll_by_team.get(r["team_id"], 0) + float(r["salary"])

    # teams, with computed payroll + apron status
    for t in teams:
        payroll = payroll_by_team.get(t["team_id"], 0.0)
        cap_space = max(thresholds["salary_cap"] - payroll, 0.0)
        status = apron_status_for(payroll, thresholds)
        conn.execute(
            "INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?,?,?,?)",
            (t["team_id"], t["name"], t["conference"], payroll, cap_space,
             status, t["season"], "2026-07-20"),
        )

    conn.commit()

    print("\nTeam payroll summary:")
    for row in conn.execute(
        "SELECT team_id, name, total_payroll, apron_status FROM teams ORDER BY total_payroll DESC"
    ):
        print(f"  {row[0]:5s} {row[1]:24s} ${row[2]:>14,.0f}   {row[3]}")

    conn.close()
    print(f"\nDatabase written to {DB_PATH}")


if __name__ == "__main__":
    main()
