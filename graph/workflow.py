"""LangGraph 工作流定义"""
from typing import Optional
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from agents import (
    process_confirm_report,
    process_format_result,
    process_intent,
    process_query,
    process_report,
    process_rag_retrieve,
    process_rag_rewrite,
    process_rag_generate,
)
from models.schemas import AgentState
from config import settings
from utils.constants import RAG_MAX_RETRIEVAL_RETRIES
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
    workflow.add_node("confirm_report", process_confirm_report)
    workflow.add_node("report", process_report)
    workflow.add_node("rag_retrieve", process_rag_retrieve)
    workflow.add_node("rag_rewrite", process_rag_rewrite)
    workflow.add_node("rag_generate", process_rag_generate)
    
    # ── 路由函数 ──

    def route_after_intent(state: AgentState) -> str:
        next_step = state.get("next_step", "end")
        user_info = state.get("user_info")
        if state.get("is_complete") and user_info and user_info.get("query_type") == "knowledge_qa":
            return "rag"
        return next_step

    def route_after_query(state: AgentState) -> str:
        return state.get("next_step", "end")

    def route_after_format(state: AgentState) -> str:
        return state.get("next_step", "end")

    def route_after_confirm(state: AgentState) -> str:
        return state.get("next_step", "end")

    # ── 入口 ──
    workflow.set_entry_point("intent")
    
    # ── 边 ──
    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {"query": "query", "report": "report", "rag": "rag_retrieve", "end": END},
    )
    
    workflow.add_conditional_edges(
        "query",
        route_after_query,
        {"format_result": "format_result", "report": "report", "end": END},
    )
    
    workflow.add_conditional_edges(
        "format_result",
        route_after_format,
        {"confirm_report": "confirm_report", "end": END},
    )

    workflow.add_conditional_edges(
        "confirm_report",
        route_after_confirm,
        {"report": "report", "end": END},
    )

    workflow.add_edge("report", END)
    
    # RAG 子流程
    def route_after_rag_retrieve(state: AgentState) -> str:
        if state.get("error"):
            return "end"
        if state.get("rag_retrieved_docs"):
            return "rag_generate"
        attempt = state.get("rag_retrieval_attempt", 1)
        if attempt >= RAG_MAX_RETRIEVAL_RETRIES:
            return "end"
        return "rag_rewrite"

    workflow.add_conditional_edges(
        "rag_retrieve",
        route_after_rag_retrieve,
        {"rag_generate": "rag_generate", "rag_rewrite": "rag_rewrite", "end": END},
    )
    workflow.add_edge("rag_rewrite", "rag_retrieve")
    workflow.add_edge("rag_generate", END)
    
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


if __name__ == "__main__":
    import sys
    from pathlib import Path

    output_dir = Path(__file__).resolve().parent.parent / "graph_images"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"

    compiled = create_workflow()
    png_data = compiled.get_graph().draw_mermaid_png()
    output_path.write_bytes(png_data)
    print(f"Graph saved: {output_path}")

    if "--open" in sys.argv:
        import webbrowser
        webbrowser.open(str(output_path))
