"""
TradeGuard AI — Tool Schemas
-------------------------------
These describe each Python function in agent/tools.py in the JSON format
Claude's tool-use API expects. This is the "menu" the LLM reads to decide
which function to call and with what arguments — the LLM never sees our
Python code, only these descriptions.

Why this file exists separately from tools.py: the schema (what the LLM
sees) and the implementation (what actually runs) are deliberately kept
apart. This mirrors how real production agent systems are built — the
LLM-facing contract and the backend logic evolve independently.
"""

TOOL_SCHEMAS = [
    {
        "name": "get_team_cap_status",
        "description": (
            "Look up a team's current total payroll, cap space, and apron status "
            "(under_cap, over_cap, over_tax, first_apron, or second_apron). Use this "
            "whenever the question depends on a team's current spending tier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_id": {"type": "string", "description": "Three-letter team code, e.g. 'BOS', 'CLE', 'GSW'"},
            },
            "required": ["team_id"],
        },
    },
    {
        "name": "get_player_salary",
        "description": "Look up a player's current salary and years remaining by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Player's name, e.g. 'Jayson Tatum'"},
            },
            "required": ["player_name"],
        },
    },
    {
        "name": "evaluate_trade",
        "description": (
            "Check whether a proposed trade satisfies CBA salary-matching rules for the "
            "team sending players out. Looks up that team's real apron status automatically. "
            "Use this when the question is specifically 'is this trade legal'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_from": {"type": "string", "description": "Team sending players out"},
                "outgoing_salaries": {"type": "array", "items": {"type": "number"}},
                "incoming_salaries": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["team_from", "outgoing_salaries", "incoming_salaries"],
        },
    },
    {
        "name": "retrieve_rules",
        "description": (
            "Search the CBA rulebook text for passages relevant to a natural-language "
            "question. Use this when the user asks 'why' or 'what rule applies' rather "
            "than just wanting a yes/no."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language question about a CBA rule"},
                "k": {"type": "integer", "description": "Number of rule passages to return", "default": 2},
            },
            "required": ["query"],
        },
    },
    {
        "name": "predict_trade_risk",
        "description": (
            "Get the ML model's predicted risk score (0-1) for a trade based on its "
            "structure. Use this alongside evaluate_trade when the user asks how risky "
            "or how likely-to-be-flagged a trade is, not just whether it's technically legal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "apron_status": {"type": "string"},
                "n_outgoing": {"type": "integer"},
                "outgoing_total": {"type": "number"},
                "incoming_total": {"type": "number"},
            },
            "required": ["apron_status", "n_outgoing", "outgoing_total", "incoming_total"],
        },
    },
    {
        "name": "recommend_legal_targets",
        "description": (
            "Recommend real players who would be a good fit for a team's roster need "
            "(by position and salary range) AND are legal to acquire given that team's "
            "current apron status. Use this when the user asks what a team SHOULD do, "
            "e.g. 'who should Team X target' or 'recommend a trade for Team X', not just "
            "whether one specific trade is legal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_id": {"type": "string", "description": "Team code, e.g. 'CLE'"},
                "needed_position": {"type": "string", "description": "e.g. 'PG', 'C', 'SF'"},
                "salary_min": {"type": "number"},
                "salary_max": {"type": "number"},
                "giving_up_salary": {"type": "number", "description": "Salary the team would send out"},
            },
            "required": ["team_id", "needed_position", "salary_min", "salary_max", "giving_up_salary"],
        },
    },
]
