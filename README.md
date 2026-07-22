# TradeGuard AI

**An agentic AI system for NBA salary cap & trade compliance** — real CBA rules engine, RAG over rule text, a trained risk classifier, and an LLM agent that decides which tools to call, all wired into a live API and web UI.

🔗 **Live site:** https://tradeguard-ai-xi.vercel.app
🔗 **Live API docs:** https://tradeguard-ai-ea0h.onrender.com/docs
🔗 **Code:** https://github.com/HarshaVardhanRepudi/tradeguard-ai-

> Note: the backend runs on a free tier that spins down after inactivity — the first request after idle time can take 20-30s to wake up. This is expected, not a bug.

## What this proves
- **Real data**, sourced from public Spotrac cap tables (not fabricated) for 6 NBA teams, spanning all three CBA rule tiers (standard, first apron, second apron)
- **A deterministic rules engine** implementing the actual CBA salary-matching math, verified against real team payrolls
- **RAG retrieval** over paraphrased CBA rule text, so every answer cites a real source
- **A trained ML classifier** (RandomForest, evaluated with accuracy/precision/recall/F1/ROC-AUC) predicting trade compliance risk
- **A recommendation engine** ranking real players by fit against a team's need, merged with legality checking
- **A real LLM agent** (Google Gemini, tool-use API) that decides for itself which of 6 tools to call and in what order — not a hardcoded pipeline
- **A production API** (FastAPI) and **deployed frontend**, both live on the public internet

## Build Log

## What's working right now (Phase 1-3: Data, Rules Engine, RAG, ML)

```
data/raw/*.csv --ETL--> etl/tradeguard.db --lookup--> agent/tools.py --applies--> etl/salary_matching.py
                                                              |              \
                                                      retrieve_rules()       predict_trade_risk()
                                                              |                     |
                                                  rag/retrieve.py (TF-IDF     ml/train_model.py
                                                  index over CBA rules)       (RandomForest classifier)
```

Run it yourself:
```bash
python3 etl/load.py                    # builds the database, prints payroll summary
python3 rag/retrieve.py                # builds the rule index, tests retrieval
python3 ml/generate_training_data.py   # builds the labeled trade dataset
python3 ml/train_model.py              # trains + evaluates the risk classifier
python3 agent/tools.py                 # demos the full chain: DB + rules + ML + RAG
```

### What's real vs. placeholder
- **Real:** salary cap / apron dollar figures (2025-26, verified from NBA.com/ESPN).
- **Real:** the salary-matching math (125% / 110% / aggregation-ban logic).
- **Real:** the CBA rule text in `data/raw/cba_rules_seed.csv` — paraphrased from real CBA
  provisions and Larry Coon's CBA FAQ, not fabricated.
- **Real player/contract data (6 of 11 teams):** CLE, GSW, LAL, BOS, DAL, PHX — pulled from
  Spotrac's public cap tables. All three CBA rule tiers (`over_cap`/`over_tax` standard
  matching, `first_apron` 110% matching, `second_apron` aggregation ban) have been proven
  against these real teams, including a case where a computed payroll independently matched
  a third-party apron tracker's figure within ~$300K.
- **Still placeholder:** DEN, MIL, OKC, NYK, MIA — all land at `under_cap` either way, so they
  don't currently prove a new rule branch. Lower priority than the ML/RAG work.
- **ML risk model:** RandomForest classifier trained on 400 synthetic trades (labeled by
  running them through the verified `salary_matching.py` engine — a rule-grounded
  approximation, not real-world negotiation outcomes) plus 5 real anchor trades from the
  2025-26 trade deadline (source: ESPN trade tracker, Feb 2026), labeled 'approved' since
  they were reported completed. Test accuracy 0.99 / F1 0.99 / ROC-AUC 1.00 — expected to
  be near-perfect since the bulk of labels come from a deterministic function, not noisy
  real outcomes. Feature importances correctly rank `salary_ratio` as the dominant signal,
  matching the CBA's actual matching-percentage logic. SHAP is unavailable in this offline
  sandbox — using RandomForest's built-in feature importances as a placeholder; swapping in
  SHAP later is a one-file change to `ml/train_model.py`.
- **Note on RAG implementation:** using TF-IDF (scikit-learn) instead of neural embeddings
  because this dev sandbox has no internet access to download model weights. The
  `retrieve_rules(query, k)` interface is what an agent calls either way — swapping in
  sentence-transformers or Azure OpenAI embeddings later is a change to `build_index()`
  only, nothing else in the codebase changes.

### Proven end-to-end today
`explain_trade()` chains all three layers automatically: looks up a team's real apron
status from the database, runs it through the deterministic CBA rules engine, and
retrieves the actual rule text that backs the verdict — with a relevance score. Verified
against real second-apron (CLE), first-apron (GSW), and standard-tier (BOS/LAL/DAL/PHX)
cases.

## Next steps (in order)
1. React frontend that calls this API instead of curl/docs page.
2. (Low priority) finish loading real data for the remaining 5 teams.

## API layer (FastAPI)
`api/main.py` wraps the tools as real HTTP endpoints:
- `GET  /cap-status/{team_id}` — direct DB lookup, no LLM
- `GET  /player/{player_name}` — direct DB lookup, no LLM
- `POST /evaluate-trade` — rules engine only, no LLM
- `POST /explain-trade` — rules engine + RAG + ML risk, no LLM (same as before, just as an endpoint)
- `POST /recommend` — the fit+legality merge, no LLM
- `POST /ask` — the real agentic endpoint; runs the full Gemini agent loop,
  the LLM decides which tools to call. Needs `GEMINI_API_KEY` set on the
  server. This is the one that costs free-tier quota per call; the others
  are free/instant since they skip the LLM entirely.

Run it:
```bash
pip3 install fastapi uvicorn
python3 api/main.py
```
Then open http://localhost:8000/docs — FastAPI auto-generates an interactive
page where you can test every endpoint directly in the browser, no frontend
needed yet.

## Agent layer status
- `agent/tool_schemas.py` — 6 tools described in a provider-agnostic format
- `agent/gemini_agent.py` — the REAL agent loop on Google's FREE tier (no card,
  ~1,500 req/day). Needs `pip install google-genai` + a free key from
  aistudio.google.com — see `agent/HOW_TO_RUN_REAL_AGENT.md`. **Recommended
  starting point** since it costs nothing.
- `agent/llm_agent.py` — the same architecture on Claude's API instead (paid,
  pay-as-you-go, no subscription) — functionally interchangeable, only the
  API-calling code differs
- `agent/mock_router.py` — a local keyword-based stand-in used to test the
  tool-selection *pattern* offline. This is NOT real reasoning — it's a
  substitute so the pipeline shape could be verified without internet access.
  Be upfront about this distinction in interviews.

## Files
- `etl/schema.sql` — database schema (includes `rule_chunks` table for RAG)
- `etl/load.py` — ETL: read CSVs, validate, compute payroll/apron status, write SQLite
- `etl/salary_matching.py` — the deterministic CBA rules engine (no dependencies)
- `rag/retrieve.py` — builds the TF-IDF rule index, exposes `retrieve_rules(query, k)`
- `agent/tools.py` — tool functions an LLM agent calls: cap lookup, trade evaluation,
  full explain-with-citations chain
- `data/raw/*.csv` — seed data (teams, contracts, league thresholds, CBA rule chunks)

