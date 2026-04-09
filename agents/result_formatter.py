"""结果格式化 Agent - 将查询结果格式化为用户友好的文本"""
import traceback
from typing import Any, Dict, List

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from models.schemas import AgentState
from utils.constants import MEAL_STATUS_MAP
from utils.decorators import PerformanceTimer
from utils.logger import logger

def format_query_results(query_results: Dict[str, Any]) -> str:
    """
    将查询结果格式化为用户友好的文本
    
    Args:
        query_results: 查询结果字典
        
    Returns:
        格式化后的文本
    """
    lines: List[str] = []
    lines.append("查询结果：")
    lines.append("")
    
    # 调试：打印所有工具结果
    logger.debug(f"format_query_results 接收到的工具: {list(query_results.keys())}")
    
    # 用户基本信息
    if "query_user_by_phone" in query_results:
        user_data = query_results["query_user_by_phone"]
        logger.debug(f"处理 query_user_by_phone: type={type(user_data)}, data={user_data}")
        if isinstance(user_data, dict) and user_data.get("success"):
            data = user_data.get("data", {})
            lines.append("【用户信息】")
            lines.append(f"  用户ID: {data.get('id')}")
            lines.append(f"  手机号: {data.get('phone')}")
            lines.append(f"  用户名: {data.get('username')}")
            lines.append("")
    
    # 套餐信息
    if "query_user_package" in query_results:
        package_data = query_results["query_user_package"]
        logger.debug(f"处理 query_user_package: type={type(package_data)}, data={package_data}")
        if isinstance(package_data, dict) and package_data.get("success"):
            data = package_data.get("data", {})
            lines.append("【套餐信息】")
            lines.append(f"  套餐ID: {data.get('id')}")
            
            # 套餐状态
            meal_status = data.get('meal_status')
            status_text = MEAL_STATUS_MAP.get(meal_status, f"状态码{meal_status}")
            lines.append(f"  套餐状态: {status_text}")
            
            lines.append(f"  电压: {data.get('voltage')}V")
            lines.append(f"  容量: {data.get('ah')}Ah")
            lines.append(f"  套餐类型: {data.get('meal')}")
            
            if data.get('start_date'):
                lines.append(f"  开始日期: {data.get('start_date')}")
            if data.get('end_date'):
                lines.append(f"  结束日期: {data.get('end_date')}")
            
            lines.append(f"  保险金额: {data.get('insurance_money')}元")
            lines.append(f"  押金金额: {data.get('deposit_money')}元")
            lines.append(f"  保险期限: {data.get('insurance_duration')}个月")
            lines.append(f"  创建时间: {data.get('create_time')}")
            lines.append("")
        else:
            logger.warning(f"query_user_package 检查失败: isinstance={isinstance(package_data, dict)}, success={package_data.get('success') if isinstance(package_data, dict) else 'N/A'}")
    else:
        logger.warning("query_user_package 不在查询结果中")
    
    # 实名信息
    if "query_realname_info" in query_results:
        realname_data = query_results["query_realname_info"]
        logger.debug(f"处理 query_realname_info: type={type(realname_data)}")
        if isinstance(realname_data, dict) and realname_data.get("success"):
            data = realname_data.get("data", {})
            lines.append("【实名信息】")
            auth_status = "已认证" if data.get('is_authentication') else "未认证"
            lines.append(f"  实名认证状态: {auth_status}")
            lines.append("")
    
    # 身份验证
    if "query_identity_info" in query_results:
        identity_data = query_results["query_identity_info"]
        logger.debug(f"处理 query_identity_info: type={type(identity_data)}")
        if isinstance(identity_data, dict) and identity_data.get("success"):
            data = identity_data.get("data", {})
            lines.append("【身份验证】")
            lines.append(f"  验证结果: {data}")
            lines.append("")
    
    result = "\n".join(lines)
    logger.debug(f"格式化后的结果长度: {len(result)} 字符")
    return result

def process_format_result(state: AgentState) -> AgentState:
    """格式化查询结果并询问是否生成报告"""
    logger.separator()
    logger.info("开始格式化查询结果")
    
    # 获取流式写入器
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    
    if not state.get("query_results"):
        logger.error("缺少查询结果")
        state["error"] = "缺少查询结果"
        state["next_step"] = "end"
        if writer:
            writer({"type": "content", "content": "\n\n缺少查询结果"})
        return state
    
    try:
        query_results = state["query_results"]
        
        logger.debug(f"查询结果包含的工具: {list(query_results.keys())}")
        for tool_name in query_results.keys():
            logger.debug(f"  - {tool_name}: {type(query_results[tool_name])}")
        
        if not isinstance(query_results, dict):
            logger.error(f"查询结果类型错误: {type(query_results)}")
            state["error"] = "查询结果格式错误"
            state["next_step"] = "end"
            if writer:
                writer({"type": "content", "content": "\n\n查询结果格式错误"})
            return state
        
        with PerformanceTimer("format_result", state.get("performance_metrics", {})):
            formatted_result = format_query_results(query_results)
            logger.debug(f"格式化结果:\n{formatted_result}")
            
            if writer:
                writer({"type": "result", "content": formatted_result})
                writer({"type": "message", "content": '是否需要生成详细报告？（回复"是"或"不需要"）'})
            
            state["error"] = formatted_result
            state["next_step"] = "confirm_report"
        
        logger.info("✓ 结果格式化完成")
        return state
    
    except Exception:
        logger.error(f"格式化失败: {traceback.format_exc()}")
        state["error"] = "结果格式化失败"
        state["next_step"] = "end"
        if writer:
            writer({"type": "content", "content": "\n\n结果格式化失败"})
        return state


def process_confirm_report(state: AgentState) -> AgentState:
    """通过 interrupt 等待用户确认是否生成报告（human-in-the-loop）

    注意：interrupt() 恢复时节点会从头重新执行，
    因此提问推送放在 format_result 中（只执行一次），此处不做 writer push。
    """
    answer = interrupt("等待用户确认是否生成报告")

    try:
        writer = get_stream_writer()
    except Exception:
        writer = None

    user_answer = str(answer).strip().lower()
    reject_keywords = ["不", "否", "算了", "取消", "no", "cancel"]
    confirm_keywords = ["是", "需要", "生成", "报告", "yes", "ok", "好"]

    # 拒绝关键词优先，避免 "不需要" 被 "需要" 误匹配
    if any(kw in user_answer for kw in reject_keywords):
        logger.info("用户拒绝生成报告")
        state["next_step"] = "end"
        if writer:
            writer({"type": "message", "content": "好的，如果以后需要生成报告，请告诉我。"})
    elif any(kw in user_answer for kw in confirm_keywords):
        logger.info("用户确认生成报告")
        state["next_step"] = "report"
    else:
        logger.info("用户回复不明确，默认不生成报告")
        state["next_step"] = "end"
        if writer:
            writer({"type": "message", "content": "好的，如果以后需要生成报告，请告诉我。"})

    return state
