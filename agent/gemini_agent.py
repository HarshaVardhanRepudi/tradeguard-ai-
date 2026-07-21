"""
TradeGuard AI — Real LLM Agent Loop, Gemini version (FREE tier, runs on YOUR machine)
------------------------------------------------------------------------------------------
Same agent loop concept as llm_agent.py (Claude version), rebuilt on Google's
Gemini API, which has a genuinely free tier (no card, ~1,500 requests/day)
suitable for building and demoing this project at zero cost.

Requires:  pip install google-genai
Get a free key: https://aistudio.google.com/apikey  (no credit card required)
Set:       export GEMINI_API_KEY=your_key_here
Run:       python3 agent/gemini_agent.py "Can BOS trade Tatum for a $70M contract?"

Note: the free tier trains on your prompts (per Google's terms). Fine for
this project (public NBA salary data), not appropriate for private/sensitive
data in other projects.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import types

from agent.tool_schemas import TOOL_SCHEMAS
from agent import tools as tool_impls

MODEL = "gemini-flash-latest"  # alias that always points to Google's current free-tier Flash model,
                                # avoids breaking when specific versions (e.g. gemini-2.5-flash) get retired

SYSTEM_PROMPT = """You are TradeGuard AI, an assistant that answers questions about NBA \
salary cap rules and trade legality. You have tools to look up real team payroll data, \
check trade compliance against CBA rules, retrieve rule text, and predict trade risk. \
Use only the tools provided — never guess at real salary or cap figures yourself. \
Chain multiple tool calls when a question requires it."""


def claude_schema_to_gemini(schema: dict) -> dict:
    """
    Claude's tool format (name/description/input_schema) and Gemini's format
    (name/description/parameters) differ slightly. Converting here means we
    define each tool ONCE in tool_schemas.py and reuse it for both providers,
    instead of maintaining two separate schema files that could drift apart.
    """
    return {
        "name": schema["name"],
        "description": schema["description"],
        "parameters": schema["input_schema"],
    }


def execute_tool(name: str, tool_input: dict):
    fn = getattr(tool_impls, name, None)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    return fn(**tool_input)


def run_agent(user_question: str, max_turns: int = 6):
    client = genai.Client()  # reads GEMINI_API_KEY from environment

    gemini_tools = [types.Tool(function_declarations=[
        claude_schema_to_gemini(s) for s in TOOL_SCHEMAS
    ])]

    contents = [types.Content(role="user", parts=[types.Part(text=user_question)])]

    for turn in range(max_turns):
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=gemini_tools,
            ),
        )

        candidate = response.candidates[0]
        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not function_calls:
            final_text = "".join(p.text for p in candidate.content.parts if p.text)
            print(f"\n[Final answer after {turn+1} turn(s)]\n{final_text}")
            return final_text

        # Model wants to call one or more tools — execute and report back.
        contents.append(candidate.content)
        function_response_parts = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            print(f"  -> Agent calls: {fc.name}({args})")
            result = execute_tool(fc.name, args)
            function_response_parts.append(
                types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": result}))
            )
        contents.append(types.Content(role="user", parts=function_response_parts))

    return "Max turns reached without a final answer."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Is Cleveland's roster over the second apron, and what does that restrict?"
    print(f"QUESTION: {question}\n")
    run_agent(question)
