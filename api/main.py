"""
TradeGuard AI — FastAPI Layer
---------------------------------
Wraps the tool functions (agent/tools.py) and the agent loop as real HTTP
endpoints. This is what a frontend (or curl, or Postman) actually talks to.

Two kinds of endpoints here, deliberately:
  - Direct endpoints (/cap-status, /evaluate-trade, /recommend) call our
    Python functions directly — fast, deterministic, no LLM cost.
  - /ask is the agentic endpoint — it runs the full Gemini agent loop,
    so the LLM decides which of the above to use internally.

Run:  pip3 install fastapi uvicorn
      python3 api/main.py
Then open http://localhost:8000/docs for interactive API docs (FastAPI
generates this automatically from the type hints below).
"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import tools as t

app = FastAPI(title="TradeGuard AI API", version="0.1")

# Allow a local frontend (different port) to call this API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request/response schemas (Pydantic — FastAPI uses these for
# automatic validation AND to generate the /docs page) ----------

class TradeRequest(BaseModel):
    team_from: str
    outgoing_salaries: List[float]
    incoming_salaries: List[float]
    question: Optional[str] = None


class RecommendRequest(BaseModel):
    team_id: str
    needed_position: str
    salary_min: float
    salary_max: float
    giving_up_salary: float
    top_k: int = 5


class AskRequest(BaseModel):
    question: str


# ---------- Direct endpoints — no LLM, just our real functions ----------

@app.get("/")
def root():
    return {"status": "TradeGuard AI API is running", "docs": "/docs"}


@app.get("/cap-status/{team_id}")
def cap_status(team_id: str):
    result = t.get_team_cap_status(team_id.upper())
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/player/{player_name}")
def player_salary(player_name: str):
    result = t.get_player_salary(player_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/evaluate-trade")
def evaluate_trade(req: TradeRequest):
    result = t.evaluate_trade(req.team_from.upper(), req.outgoing_salaries, req.incoming_salaries)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/explain-trade")
def explain_trade(req: TradeRequest):
    result = t.explain_trade(req.team_from.upper(), req.outgoing_salaries,
                               req.incoming_salaries, req.question)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/recommend")
def recommend(req: RecommendRequest):
    result = t.recommend_legal_targets(
        req.team_id.upper(), req.needed_position, req.salary_min,
        req.salary_max, req.giving_up_salary, req.top_k,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ---------- Agentic endpoint — runs the real LLM loop ----------

@app.post("/ask")
def ask(req: AskRequest):
    """
    Runs the question through the real Gemini agent loop — the LLM decides
    which tools to call. Requires GEMINI_API_KEY to be set in the
    environment this server runs in.
    """
    try:
        from agent.gemini_agent import run_agent
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="google-genai not installed. Run: pip3 install google-genai",
        )
    answer = run_agent(req.question)
    return {"question": req.question, "answer": answer}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # hosting platforms (Render, etc.) assign PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
