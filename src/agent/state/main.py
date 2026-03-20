from typing import Any, Literal

from typing_extensions import TypedDict

from langgraph.graph import MessagesState


RouteDestination = Literal["RECOMMEND", "POLICY_RAG", "HANDOFF", "EMPATHY", "END"]


class UserProfile(TypedDict, total=False):
    budget_min: float
    budget_max: float
    city: str
    district: str
    bedrooms: int
    room_type: str
    orientation: str
    soft_preferences: list[str]
    emotion: str


class AgentOutcomes(TypedDict, total=False):
    recommend: dict[str, Any]
    policy_rag: dict[str, Any]
    handoff: dict[str, Any]


# 主图状态
class State(MessagesState):
    # 新架构核心字段
    user_profile: UserProfile
    route_destination: RouteDestination
    agent_outcomes: AgentOutcomes

    # 兼容旧字段（避免影响历史节点/调用）
    user_preferences: dict
    user_intent: str
    user_memory: dict
    emotion_label: str
    intent_level: str
    route: str


# 私有状态
class NeedReserveOutput(TypedDict):
    reserve: str  # 需要、不需要
    # 这个状态不会出现在最终状态