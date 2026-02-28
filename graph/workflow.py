"""LangGraph 工作流定义"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents import (
    process_format_result,
    process_intent,
    process_query,
    process_report,
)
from models.schemas import AgentState


def create_workflow():
    """创建LangGraph工作流"""
    
    # 创建状态图
    workflow = StateGraph[AgentState, None, AgentState, AgentState](AgentState)
    
    # 添加节点
    workflow.add_node("intent", process_intent)
    workflow.add_node("query", process_query)
    workflow.add_node("format_result", process_format_result)
    workflow.add_node("report", process_report)
    
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
        
        return state.get("next_step", "end")
    
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
    
    # 创建内存checkpointer
    memory = MemorySaver()
    
    # 编译工作流,传入checkpointer
    return workflow.compile(checkpointer=memory)


# 创建全局工作流实例
app = create_workflow()
