# Running the REAL agent with Gemini (free tier) on your own machine

This sandbox has no internet access, so `agent/mock_router.py` was used to test
the tool-selection *pattern* offline (keyword matching, NOT real reasoning).
To run the actual LLM-driven agent for free:

## Setup
```bash
pip install google-genai
```
1. Go to https://aistudio.google.com/apikey — no credit card needed
2. Create a free API key
3. `export GEMINI_API_KEY=your_key_here`

## Run it
```bash
python3 agent/gemini_agent.py "Can Boston legally trade Jayson Tatum for a player making 70 million?"
```

## What to watch for
The printed `-> Agent calls: tool_name(...)` lines are Gemini's real, live
decision-making — it read the tool descriptions in `agent/tool_schemas.py`
and decided for itself which ones this question needs, in what order. That's
the difference from `mock_router.py`, which can only match questions it was
specifically coded to recognize.

## Suggested test questions
- "Is Cleveland allowed to combine two contracts in a trade right now?"
- "What's the difference in trade flexibility between Golden State and Denver?"
- "If Boston wanted to upgrade at center for around $12M, what salary range could they offer?"
- "Recommend a legal center trade target for Cleveland" — this one should
  trigger the merged fit+legality tool if you add it to tool_schemas.py

## Free tier limits to be aware of
~1,500 requests/day, no cost, no card required. Google's terms note free-tier
prompts may be used to improve their models — acceptable for this project
(all data here is public NBA salary info), but worth knowing as a general
privacy tradeoff for future projects with sensitive data.

## If you'd rather use Claude instead (paid, pay-as-you-go, no subscription)
`agent/llm_agent.py` is the same architecture built on Claude's API instead —
functionally interchangeable, just needs `pip install anthropic` and a paid
API key. Having both versions is a legitimate thing to mention in an
interview: the agent architecture (tool schemas, the loop, the tool
implementations) is provider-agnostic — only the ~15 lines that call the API
differ between them.
