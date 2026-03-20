from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class ContextSchema(TypedDict):
    user_id: str
    mcp_context: Optional[dict[str, Any]]
    web_mcp_state: Optional[dict[str, Any]]
    interactive_mode: Optional[Literal["interrupt", "streamlit", "cli"]]
    need_reserve_decision: Optional[Literal["需要", "不需要"]]
    reserve_form: Optional[dict[str, Any]]