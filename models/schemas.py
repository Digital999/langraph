"""数据模型定义"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, TypedDict

class AgentState(TypedDict, total=False):
    """Agent 状态模型 - 用于 LangGraph"""
    user_input: str
    is_complete: bool
    user_info: Optional[Dict[str, Any]]
    query_results: Optional[Dict[str, Any]]
    report_path: Optional[str]
    error: Optional[str]
    conversation_history: List[str]
    next_step: str
    # 新增：用于询问是否生成报告
    waiting_for_report_confirmation: bool
    # 新增：性能统计
    performance_metrics: Dict[str, float]

class UserRequest(BaseModel):
    """用户请求模型"""
    user_input: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    """Agent 响应模型"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    report_url: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
