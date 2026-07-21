"""
TradeGuard AI — Real LLM Agent Loop (runs on YOUR machine, needs ANTHROPIC_API_KEY)
--------------------------------------------------------------------------------------
This is the actual agentic loop: Claude receives the user's question plus the
tool schemas, decides which tool(s) to call, we execute them in Python, feed
the results back to Claude, and it either calls more tools or gives a final
answer. This repeats until Claude stops requesting tools.

Requires:  pip install anthropic
Set:       export ANTHROPIC_API_KEY=your_key_here
Run:       python3 agent/llm_agent.py "Can BOS trade Jayson Tatum's $54M for a $70M contract?"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
from agent.tool_schemas import TOOL_SCHEMAS
from agent import tools as tool_impls

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are TradeGuard AI, an assistant that answers questions about NBA \
salary cap rules and trade legality. You have tools to look up real team payroll data, \
check trade compliance against CBA rules, retrieve rule text, and predict trade risk. \
Use only the tools provided — never guess at real salary or cap figures yourself. \
Chain multiple tool calls when a question requires it (for example, look up a team's \
cap status before evaluating whether a trade is legal for them)."""


def execute_tool(name: str, tool_input: dict):
    """Dispatch a tool call by name to the real Python function in agent/tools.py."""
    fn = getattr(tool_impls, name, None)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    return fn(**tool_input)


def run_agent(user_question: str, max_turns: int = 6):
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    messages = [{"role": "user", "content": user_question}]

    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # If Claude didn't ask for any tools, it's giving its final answer.
        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            print(f"\n[Final answer after {turn+1} turn(s)]\n{final_text}")
            return final_text

        # Claude requested one or more tool calls — execute each and report back.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  -> Agent calls: {block.name}({block.input})")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})

    return "Max turns reached without a final answer."


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Is Cleveland's roster over the second apron, and what does that restrict?"
    print(f"QUESTION: {question}\n")
    run_agent(question)
