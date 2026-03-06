"""结果格式化 Agent - 将查询结果格式化为用户友好的文本"""
import traceback
from typing import Any, Dict, List

from langgraph.config import get_stream_writer
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
        
        # 调试：打印查询结果的键
        logger.debug(f"查询结果包含的工具: {list(query_results.keys())}")
        for tool_name in query_results.keys():
            logger.debug(f"  - {tool_name}: {type(query_results[tool_name])}")
        
        # 检查 query_results 是否是字典
        if not isinstance(query_results, dict):
            logger.error(f"查询结果类型错误: {type(query_results)}")
            state["error"] = "查询结果格式错误"
            state["next_step"] = "end"
            if writer:
                writer({"type": "content", "content": "\n\n查询结果格式错误"})
            return state
        
        # 使用性能计时器
        with PerformanceTimer("format_result", state.get("performance_metrics", {})):
            # 格式化结果
            formatted_result = format_query_results(query_results)
            logger.debug(f"格式化结果:\n{formatted_result}")
            
            # 推送格式化后的结果到前端（保留换行符）
            if writer:
                writer({"type": "result", "content": formatted_result})
            
            # 添加询问是否生成报告
            report_prompt = "\n" + "="*50 + "\n" + '是否需要生成详细报告？（回复"是"、"需要"、"生成报告"等即可生成Word报告）'
            formatted_result += report_prompt
            
            # 推送报告询问
            if writer:
                writer({"type": "message", "content": report_prompt})
            
            state["error"] = formatted_result
            state["waiting_for_report_confirmation"] = True
            state["next_step"] = "end"
        
        logger.info("✓ 结果格式化完成")
        return state
    
    except Exception:
        logger.error(f"格式化失败: {traceback.format_exc()}")
        state["error"] = "结果格式化失败"
        state["next_step"] = "end"
        if writer:
            writer({"type": "content", "content": "\n\n结果格式化失败"})
        return state
