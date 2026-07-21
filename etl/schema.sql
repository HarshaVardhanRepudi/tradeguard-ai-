-- TradeGuard AI — Core Schema
-- Works on SQLite (dev) and Postgres (deploy) with minor type tweaks noted inline.

CREATE TABLE IF NOT EXISTS teams (
    team_id         TEXT PRIMARY KEY,      -- e.g. 'BOS', 'LAL'
    name            TEXT NOT NULL,
    conference      TEXT,
    total_payroll   REAL,                  -- sum of all active contracts, computed by ETL
    cap_space       REAL,                  -- salary_cap - total_payroll (floor at 0)
    apron_status    TEXT,                  -- 'under_cap' | 'over_cap' | 'over_tax' | 'first_apron' | 'second_apron'
    season          TEXT NOT NULL,         -- e.g. '2025-26'
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    team_id         TEXT,
    position        TEXT,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS contracts (
    contract_id         TEXT PRIMARY KEY,
    player_id           TEXT NOT NULL,
    season              TEXT NOT NULL,
    salary              REAL NOT NULL,
    years_remaining     INTEGER,
    no_trade_clause     INTEGER DEFAULT 0,   -- boolean 0/1
    trade_kicker_pct    REAL DEFAULT 0,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS trade_history (
    trade_id            TEXT PRIMARY KEY,
    trade_date           TEXT,
    team_from           TEXT,
    team_to              TEXT,
    players_out          TEXT,   -- comma-separated player_ids
    players_in           TEXT,
    salary_out           REAL,
    salary_in            REAL,
    team_from_apron_status_at_trade TEXT,
    outcome              TEXT,   -- 'approved' | 'rejected' | 'restructured'
    FOREIGN KEY (team_from) REFERENCES teams(team_id),
    FOREIGN KEY (team_to) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS league_thresholds (
    season          TEXT PRIMARY KEY,
    salary_cap      REAL NOT NULL,
    luxury_tax      REAL NOT NULL,
    first_apron     REAL NOT NULL,
    second_apron    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rule_chunks (
    chunk_id        TEXT PRIMARY KEY,
    source_doc      TEXT,       -- 'CBA_official' | 'Coon_FAQ'
    section         TEXT,       -- e.g. 'Article VII, Sec 3(c)'
    topic           TEXT,       -- 'salary_matching' | 'aggregation_ban' | 'trade_exception' | 'hard_cap'
    text            TEXT NOT NULL,
    embedding       BLOB        -- populated later by the embedding step
);
