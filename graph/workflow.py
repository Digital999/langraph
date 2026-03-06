"""LangGraph 工作流定义"""
from typing import Optional
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from agents import (
    process_format_result,
    process_intent,
    process_query,
    process_report,
    process_rag,
)
from models.schemas import AgentState
from config import settings
from utils.logger import logger


# 全局 checkpointer 实例
_checkpointer: Optional[BaseCheckpointSaver] = None
_connection: Optional[AsyncConnection] = None


def create_workflow():
    """创建LangGraph工作流（不带 checkpointer）"""
    
    # 创建状态图
    workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)
    
    # 添加节点
    workflow.add_node("intent", process_intent)
    workflow.add_node("query", process_query)
    workflow.add_node("format_result", process_format_result)
    workflow.add_node("report", process_report)
    workflow.add_node("rag", process_rag)
    
    # 定义路由逻辑
    def route_after_intent(state: AgentState) -> str:
        """意图理解后的路由"""
        # 检查是否在等待报告确认
        if state.get("waiting_for_report_confirmation"):
            user_input = state.get("user_input", "").lower()
            # 检查用户是否确认生成报告
            reject_keywords = ["不", "取消", "拒绝", "no", "cancel", "不需要", "不用", "不要"]
            confirm_keywords = ["是", "需要", "生成", "报告", "yes", "ok", "好"]
            
            if any(keyword in user_input for keyword in reject_keywords):
                # 清理状态
                state["waiting_for_report_confirmation"] = False
                state["next_step"] = "end"
                # 推送拒绝消息
                try:
                    from langgraph.config import get_stream_writer
                    writer = get_stream_writer()
                    writer({"type": "message", "content": "好的，如果以后需要生成报告，请告诉我。"})
                except Exception:
                    pass
                return "end"
            elif any(keyword in user_input for keyword in confirm_keywords):
                # 清理状态并生成报告
                state["waiting_for_report_confirmation"] = False
                state["next_step"] = "report"
                return "report"
            else:
                # 用户回复不明确，但不清理状态，继续等待明确回复
                try:
                    from langgraph.config import get_stream_writer
                    writer = get_stream_writer()
                    writer({"type": "message", "content": '请明确回复"是"或"不需要"确认是否需要生成报告。'})
                except Exception:
                    pass
                # 保持 waiting_for_report_confirmation 为 True
                state["next_step"] = "end"
                return "end"
        
        # 根据查询类型路由
        next_step = state.get("next_step", "end")
        
        # 如果是知识问答，路由到 RAG
        user_info = state.get("user_info")
        if state.get("is_complete") and user_info and user_info.get("query_type") == "knowledge_qa":
            return "rag"
        
        return next_step
    
    def route_after_query(state: AgentState) -> str:
        """查询后的路由"""
        return state.get("next_step", "end")
    
    def route_after_format(state: AgentState) -> str:
        """格式化后的路由"""
        return state.get("next_step", "end")
    
    def route_after_report(state: AgentState) -> str:
        """报告生成后的路由"""
        return state.get("next_step", "end")
    
    # 设置入口点
    workflow.set_entry_point("intent")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "query": "query",
            "report": "report",
            "rag": "rag",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "query",
        route_after_query,
        {
            "format_result": "format_result",
            "report": "report",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "format_result",
        route_after_format,
        {
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "report",
        route_after_report,
        {
            "end": END
        }
    )
    
    # RAG 节点后直接结束
    workflow.add_edge("rag", END)
    
    # 编译工作流（使用全局 checkpointer）
    if _checkpointer:
        return workflow.compile(checkpointer=_checkpointer)  # type: ignore
    return workflow.compile()  # type: ignore


async def init_checkpointer() -> BaseCheckpointSaver:
    """初始化 PostgreSQL checkpointer（应用启动时调用）"""
    global _checkpointer, _connection, app
    
    if _checkpointer is not None:
        return _checkpointer
    
    try:
        connection_string = settings.POSTGRES_URI
        logger.info(f"初始化 PostgreSQL checkpointer: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
        
        # 创建异步连接
        _connection = await AsyncConnection.connect(
            connection_string,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row  # type: ignore
        )
        
        # 创建 checkpointer
        _checkpointer = AsyncPostgresSaver(_connection)  # type: ignore
        
        # 初始化表结构
        await _checkpointer.setup()
        
        logger.info("PostgreSQL checkpointer 初始化成功")
        
        # 重新创建带 checkpointer 的 app
        app = create_workflow()
        
        return _checkpointer
        
    except Exception as e:
        logger.error(f"PostgreSQL checkpointer 初始化失败: {e}")
        logger.warning("回退到内存存储")
        from langgraph.checkpoint.memory import MemorySaver
        _checkpointer = MemorySaver()
        
        # 重新创建带 checkpointer 的 app
        app = create_workflow()
        
        return _checkpointer


async def cleanup_checkpointer() -> None:
    """清理 PostgreSQL 连接（应用关闭时调用）"""
    global _checkpointer, _connection, app
    
    if _connection is not None:
        try:
            await _connection.close()
            logger.info("PostgreSQL 连接已关闭")
        except Exception as e:
            logger.error(f"关闭 PostgreSQL 连接失败: {e}")
        finally:
            _connection = None
            _checkpointer = None
            # 重新创建不带 checkpointer 的 app
            app = create_workflow()


def get_app():
    """获取当前的 workflow app 实例"""
    return app


# 全局应用实例（初始不带 checkpointer，需要在启动时调用 init_checkpointer）
app = create_workflow()
