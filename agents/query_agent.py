"""查库 Agent - 使用 MCP 工具通过大模型自主决策查询数据"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from config import settings
from models.schemas import AgentState
from tools import ALL_TOOLS
from utils.logger import logger
from utils.constants import LLM_TEMPERATURE, LLM_MAX_RETRIES, LLM_TIMEOUT
from utils.decorators import PerformanceTimer
import traceback

QUERY_SYSTEM_PROMPT = """你是一个专业的数据查询助手，负责根据用户需求调用相应的数据库查询工具。

你的职责：
1. 根据用户提供的信息和查询意图，自主选择合适的工具进行查询
2. 必须先调用 query_user_by_phone 工具获取用户基本信息（user_id等）
3. 如果用户不存在，直接返回友好提示
4. 如果用户存在，根据查询意图选择对应的工具（只调用一个）：
   - 查询套餐信息：调用 query_user_package(user_id)
   - 验证身份：调用 query_identity_info(name, id_card, phone)
   - 查询实名信息：调用 query_realname_info(user_id)

可用工具说明：
- query_user_by_phone(phone): 根据手机号查询用户基本信息（必须首先调用）
- query_user_package(user_id): 查询用户套餐信息
- query_identity_info(name, id_card, phone): 验证身份三要素（需要姓名、身份证号、手机号）
- query_realname_info(user_id): 查询用户实名信息

工作流程：
1. 首先调用 query_user_by_phone 获取用户信息
2. 检查返回结果的 success 字段，如果为 false 说明用户不存在
3. 如果用户存在，从返回的 data 字段中提取 user_id, real_name, id_card 等信息
4. 根据查询意图调用对应的工具（只调用一个），传入正确的参数

特别注意：
- 身份验证（identity）需要三个参数：name, id_card, phone
  * 如果用户提供了这三个参数，直接使用用户提供的值
  * 如果用户只提供了 phone，从 query_user_by_phone 的结果中获取 name 和 id_card
- 套餐查询和实名查询需要 user_id，从 query_user_by_phone 的结果中获取
- 所有工具返回格式为 {{"success": true/false, "data": {{...}} 或 "error": "..."}}
- 必须检查每个工具调用的返回结果
- 按照正确的顺序调用工具（先查用户，再查详细信息）
- 用户一次只能查询一种信息，不要调用多个查询工具

完成查询后，请返回一个总结，格式为：
查询完成。结果：[简要描述查询到的信息]
"""

# 创建带工具的 LLM
llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    max_retries=LLM_MAX_RETRIES,
    timeout=LLM_TIMEOUT
)

# 创建 ReAct Agent
query_agent = create_react_agent(llm, ALL_TOOLS)

def process_query(state: AgentState) -> AgentState:
    """执行数据库查询 - 通过大模型自主选择工具"""
    logger.separator()
    logger.info("开始数据查询")
    
    # 初始化性能指标
    if "performance_metrics" not in state:
        state["performance_metrics"] = {}
    
    if not state.get("user_info"):
        logger.error("缺少用户信息")
        state["error"] = "缺少用户信息"
        state["next_step"] = "end"
        return state
    
    params = state["user_info"].get("params", {})
    query_type = state["user_info"].get("query_type", "all")
    
    logger.info(f"查询类型: {query_type}")
    logger.info(f"查询参数: {params}")
    
    try:
        # 构建查询消息
        query_message = f"""
用户输入：{state["user_input"]}
查询类型：{query_type}
查询参数：{params}

请严格按照以下步骤执行查询：

第一步：使用 query_user_by_phone 查询用户基本信息
- 参数：phone = {params.get('phone')}

第二步：根据查询类型调用对应工具（只调用一个）
"""
        
        if query_type == "package":
            query_message += """- 查询类型是 package，必须调用 query_user_package(user_id)
- 从第一步的结果中获取 user_id（即 data.id 字段）
- 调用 query_user_package 获取套餐信息
"""
        elif query_type == "realname":
            query_message += """- 查询类型是 realname，必须调用 query_realname_info(user_id)
- 从第一步的结果中获取 user_id（即 data.id 字段）
- 调用 query_realname_info 获取实名信息
"""
        elif query_type == "identity":
            query_message += f"""- 查询类型是 identity，必须调用 query_identity_info(name, id_card, phone)
- 参数：name = {params.get('name')}, id_card = {params.get('id_card')}, phone = {params.get('phone')}
- 如果参数不完整，从第一步的结果中获取
"""
        
        query_message += """
重要提醒：
- 必须完成所有步骤，不要只查询用户信息就停止
- 只调用一个查询工具，不要调用多个
- 每个工具调用后检查返回的 success 字段
- 完成查询后，返回总结
"""
        
        logger.debug("调用 ReAct Agent 执行工具调用...")
        
        # 使用性能计时器
        with PerformanceTimer("query_total", state["performance_metrics"]) as query_timer:
            with PerformanceTimer("query_agent_total", state["performance_metrics"]):
                # 调用 Agent 执行查询
                messages = [
                    SystemMessage(content=QUERY_SYSTEM_PROMPT),
                    HumanMessage(content=query_message)
                ]
                
                result = query_agent.invoke({"messages": messages})
            
            # 提取工具调用结果和性能数据
            tool_results = {}
            has_error = False
            
            if "messages" in result:
                logger.debug(f"收到 {len(result['messages'])} 条消息")
                for msg in result["messages"]:
                    # 只处理工具消息（ToolMessage），忽略 AI 的总结消息
                    if hasattr(msg, 'name') and hasattr(msg, 'content') and msg.name:
                        tool_name = msg.name
                        logger.info(f"✓ 工具调用: {tool_name}")
                        logger.debug(f"消息类型: {type(msg)}, content类型: {type(msg.content)}")
                        
                        try:
                            tool_content = None
                            
                            # 如果已经是字典，直接使用
                            if isinstance(msg.content, dict):
                                tool_content = msg.content
                                logger.debug("内容已经是字典类型")
                            # 如果是字符串，尝试解析
                            elif isinstance(msg.content, str):
                                import json
                                try:
                                    tool_content = json.loads(msg.content)
                                    logger.debug("JSON 解析成功")
                                except json.JSONDecodeError:
                                    logger.warning(f"工具 {tool_name} 返回的不是有效 JSON，尝试其他方式")
                                    logger.debug(f"原始内容前200字符: {msg.content[:200]}")
                            
                            # 只保存字典类型的结果
                            if isinstance(tool_content, dict):
                                tool_results[tool_name] = tool_content
                                
                                # 提取性能数据
                                if "_perf" in tool_content:
                                    perf_data = tool_content.pop("_perf")
                                    state["performance_metrics"].update(perf_data)
                                    logger.debug(f"工具耗时: {perf_data}")
                                
                                logger.debug(f"✓ 工具结果已保存: success={tool_content.get('success')}, has_data={bool(tool_content.get('data'))}")
                                
                                # 检查工具调用是否失败
                                if not tool_content.get("success", True):
                                    has_error = True
                                    error_msg = tool_content.get("error", "查询失败")
                                    logger.warning(f"工具调用失败: {error_msg}")
                            else:
                                logger.warning(f"✗ 工具 {tool_name} 结果未保存: tool_content类型={type(tool_content)}")
                        except Exception as e:
                            logger.error(f"处理工具结果时出错:\n{traceback.format_exc()}")
            
            # 如果没有工具结果，说明查询失败
            if not tool_results:
                logger.error("未能执行查询工具")
                state["error"] = "抱歉，未能查询到相关信息。请检查手机号码是否正确。"
                state["next_step"] = "end"
                return state
            
            # 检查是否有工具调用失败（特别是 query_user_by_phone）
            if has_error or (tool_results.get("query_user_by_phone") and 
                            not tool_results["query_user_by_phone"].get("success", True)):
                logger.warning("用户信息查询失败")
                state["error"] = f"抱歉，未找到手机号 {params.get('phone')} 对应的用户信息。请确认手机号码是否正确。"
                state["next_step"] = "end"
                return state
            
            # 保存查询结果
            state["query_results"] = tool_results
            # 不再自动生成报告，而是返回结果并询问
            state["next_step"] = "format_result"
            
            logger.info(f"✓ 查询完成，共调用 {len(tool_results)} 个工具，总耗时: {query_timer.elapsed_ms:.0f}ms")
            return state
    
    except Exception as e:
        logger.error(f"查询失败:\n{traceback.format_exc()}")
        state["error"] = "抱歉，查询服务暂时不可用，请稍后重试。"
        state["next_step"] = "end"
        return state
