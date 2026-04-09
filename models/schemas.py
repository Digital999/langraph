"""数据模型定义"""
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel


class AgentState(TypedDict, total=False):
    """Agent状态模型 - 用于LangGraph"""
    user_input: str
    is_complete: bool
    user_info: Optional[Dict[str, Any]]
    query_results: Optional[Dict[str, Any]]
    report_path: Optional[str]
    error: Optional[str]
    conversation_history: List[str]
    next_step: str
    # 性能统计
    performance_metrics: Dict[str, float]
    # RAG 相关字段
    rag_answer: Optional[str]
    rag_sources: Optional[List[Dict[str, Any]]]
    # RAG 检索重试状态（图结构循环用）
    rag_retrieval_attempt: int
    rag_search_query: Optional[str]
    rag_attempted_queries: List[str]
    rag_retrieved_docs: Optional[List[Dict[str, Any]]]
    rag_retry_pending: bool


class UserRequest(BaseModel):
    """用户请求模型"""
    user_input: str
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent响应模型"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    report_url: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
