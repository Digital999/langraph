"""意图理解 Agent - 统一的流式/非流式处理"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer
from config import settings
from models.schemas import AgentState
from utils.constants import LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_MAX_RETRIES, MAX_HISTORY_DISPLAY_TURNS
from utils.decorators import build_history_string
from utils.logger import logger
import traceback
import json

INTENT_SYSTEM_PROMPT = """你是智能查询助手。用户可能分多次提供信息，你必须结合对话历史理解完整意图。

【核心规则 - 必须严格执行】：

1. 查询类型识别（按优先级）：
   第一步：检查对话历史和当前输入中是否包含关键词
   - 包含"套餐" → query_type = "package"（需要：手机号）
   - 包含"实名"或"认证" → query_type = "realname"（需要：手机号）
   - 包含"验证"或"三要素"或"身份" → query_type = "identity"（需要：手机号 + 姓名 + 身份证号）
   - 都不包含 → 询问用户想查什么

2. 参数提取：
   - package/realname：只需要手机号（11位）
   - identity：需要手机号（11位）+ 姓名 + 身份证号（18位）

3. 信息完整性判断：
   - package/realname：有查询类型 + 手机号 → complete: true
   - identity：有查询类型 + 手机号 + 姓名 + 身份证号 → complete: true
   - 否则 → complete: false，询问缺少的信息

【输出格式】：
- 信息完整时，返回JSON：
  * package/realname：{{"complete": true, "query_type": "package/realname", "params": {{"phone": "手机号"}}}}
  * identity：{{"complete": true, "query_type": "identity", "params": {{"phone": "手机号", "name": "姓名", "id_card": "身份证号"}}}}
- 信息不完整时，返回纯文本（不要JSON）：
  * 直接用自然语言询问缺少的信息，例如："好的，请提供您的11位手机号码。"

【关键示例】：

示例1 - 套餐查询（完整）：
对话历史："用户: 帮我查下套餐信息\n助手: 好的，请提供您的11位手机号码。"
当前输入："17775711190"
→ 有"套餐" + 手机号 → 完整
→ 返回：{{"complete": true, "query_type": "package", "params": {{"phone": "17775711190"}}}}

示例2 - 身份验证（不完整，只有手机号）：
对话历史："用户: 17775711190"
当前输入："帮我查下身份三要素"
→ 有"三要素" + 手机号，但缺少姓名和身份证号 → 不完整
→ 返回："好的，除了手机号，我还需要您的姓名和身份证号码来验证身份三要素。请提供：\n1. 姓名\n2. 身份证号（18位）"

示例3 - 身份验证（完整）：
当前输入："验证17775711190，张三，110101199001011234"
→ 有"验证" + 手机号 + 姓名 + 身份证号 → 完整
→ 返回：{{"complete": true, "query_type": "identity", "params": {{"phone": "17775711190", "name": "张三", "id_card": "110101199001011234"}}}}

示例4 - 身份验证（完整，分多次提供）：
对话历史："用户: 验证身份\n助手: 请提供手机号、姓名和身份证号\n用户: 17775711190\n用户: 张三"
当前输入："110101199001011234"
→ 对话历史中有"验证" + 手机号 + 姓名，当前输入有身份证号 → 完整
→ 返回：{{"complete": true, "query_type": "identity", "params": {{"phone": "17775711190", "name": "张三", "id_card": "110101199001011234"}}}}

【严禁错误】：
❌ 身份验证只有手机号，却返回 complete: true（错误！必须有姓名和身份证号）
❌ 对话历史有"套餐"，却返回 query_type: "all"
❌ 信息已完整，却继续询问
"""

# 创建支持流式的 LLM - 优化配置减少延迟
llm_stream = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    streaming=True,  # 启用流式输出
    max_tokens=LLM_MAX_TOKENS,  # 限制最大 token 数，加快首字响应
)

llm_normal = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=LLM_TEMPERATURE,
    max_retries=LLM_MAX_RETRIES,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_SYSTEM_PROMPT),
    ("human", "对话历史：\n{history}\n\n当前用户输入：{user_input}")
])

chain_stream = prompt | llm_stream
chain_normal = prompt | llm_normal

def process_intent(state: AgentState) -> AgentState:
    """
    处理用户意图 - 统一入口（支持LangGraph流式输出）
    
    使用 get_stream_writer() 自动推送流式消息到前端
    """
    try:
        # 检查是否在等待报告确认
        if state.get("waiting_for_report_confirmation"):
            user_input = state.get("user_input", "").lower()
            # 检查用户输入是否是报告确认相关的回复
            report_related_keywords = ["是", "需要", "生成", "报告", "不", "取消", "拒绝", "不需要", "不用", "不要", "yes", "no", "ok", "好", "cancel"]
            
            # 如果用户输入包含报告相关关键词，说明是在回答报告确认问题
            if any(keyword in user_input for keyword in report_related_keywords):
                logger.info("检测到报告确认等待状态，跳过意图理解")
                # 直接返回，让workflow的路由逻辑处理
                state["is_complete"] = False
                state["next_step"] = "end"
                return state
            else:
                # 用户输入不是报告确认相关，说明是新的查询，清理报告确认状态
                logger.info("用户开始新的查询，清理报告确认状态")
                state["waiting_for_report_confirmation"] = False
                # 继续正常的意图理解流程
        
        full_content = ""
        
        # 获取流式写入器（如果在流式上下文中）
        try:
            writer = get_stream_writer()
        except Exception:
            writer = None  # 非流式上下文
        
        # 构建对话历史字符串
        history = build_history_string(
            state.get("conversation_history", []),
            max_turns=MAX_HISTORY_DISPLAY_TURNS
        )
        
        # 添加调试日志
        logger.debug(f"对话历史内容: {history}")
        logger.debug(f"当前用户输入: {state['user_input']}")
        
        # 使用invoke调用LLM
        # LangGraph的stream_mode="messages"会自动捕获流式输出
        response = chain_stream.invoke({
            "user_input": state["user_input"],
            "history": history
        })
        
        # 提取完整内容
        full_content = response.content
        if isinstance(full_content, list):
            full_content = full_content[0] if full_content else ""
            if hasattr(full_content, 'text'):
                full_content = full_content.text
        full_content = str(full_content).strip()
        
        logger.debug(f"LLM完整响应: {full_content[:100]}...")
        
        # 清理markdown代码块标记
        content_to_parse = full_content.strip()
        if content_to_parse.startswith("```json"):
            content_to_parse = content_to_parse[7:]
        if content_to_parse.startswith("```"):
            content_to_parse = content_to_parse[3:]
        if content_to_parse.endswith("```"):
            content_to_parse = content_to_parse[:-3]
        content_to_parse = content_to_parse.strip()
        
        # 尝试解析为 JSON
        try:
            result = json.loads(content_to_parse)
            
            # 如果是完整的查询请求
            if result.get("complete"):
                state["is_complete"] = True
                state["user_info"] = result
                state["next_step"] = "query"
                # JSON 格式不输出到前端，只用于内部处理
                # 更新对话历史（不包含JSON）
                if "conversation_history" not in state:
                    state["conversation_history"] = []
                state["conversation_history"].append(f"用户: {state['user_input']}")
                
                # 推送状态更新（不推送JSON内容）
                if writer:
                    writer({"type": "intent_complete", "query_type": result.get("query_type")})
            else:
                state["is_complete"] = False
                state["error"] = result.get("response") or result.get("suggestion") or content_to_parse
                state["next_step"] = "end"
                # 非完整请求，推送提示信息
                if writer and state["error"]:
                    writer({"type": "message", "content": state["error"]})
                # 更新对话历史
                if "conversation_history" not in state:
                    state["conversation_history"] = []
                state["conversation_history"].append(f"用户: {state['user_input']}")
                state["conversation_history"].append(f"助手: {state['error']}")
        
        except json.JSONDecodeError:
            # 不是 JSON，说明是自然语言回复，输出到前端
            state["is_complete"] = False
            state["error"] = full_content
            state["next_step"] = "end"
            if writer:
                writer({"type": "message", "content": full_content})
            # 更新对话历史
            if "conversation_history" not in state:
                state["conversation_history"] = []
            state["conversation_history"].append(f"用户: {state['user_input']}")
            state["conversation_history"].append(f"助手: {full_content}")
        
        return state
        
    except Exception as e:
        logger.error(f"意图理解错误:\n{traceback.format_exc()}")
        error_msg = str(e)
        
        if "500" in error_msg or "InternalServerError" in error_msg:
            state["error"] = "抱歉，服务暂时不可用，请稍后重试。"
        else:
            state["error"] = "您好！我可以帮您查询用户信息。请提供您的11位手机号码。"
        
        state["is_complete"] = False
        state["next_step"] = "end"
        
        # 推送错误消息
        try:
            writer = get_stream_writer()
            writer({"type": "error", "content": state["error"]})
        except Exception:
            pass
        
        return state
