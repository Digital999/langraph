"""LangGraph 工作流定义"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models.schemas import AgentState
from agents import process_intent, process_query, process_report, process_format_result

def create_workflow():
    """创建 LangGraph 工作流"""
    
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
            if any(keyword in user_input for keyword in ["不", "取消", "拒绝", "no", "cancel", "不需要", "不用"]):
                state["waiting_for_report_confirmation"] = False
                return "end"
            if any(keyword in user_input for keyword in ["是", "需要", "生成", "报告", "yes", "ok"]):
                state["waiting_for_report_confirmation"] = False
                return "report"
        
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
    
    # 创建内存 checkpointer
    memory = MemorySaver()
    
    # 编译工作流，传入 checkpointer
    return workflow.compile(checkpointer=memory)

# 创建全局工作流实例
app = create_workflow()
