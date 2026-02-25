"""意图理解 Agent - 判断用户输入是否完整"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from models.schemas import AgentState
from utils.logger import logger
import time
import traceback

INTENT_SYSTEM_PROMPT = """你是一个智能查询助手，可以帮助用户查询套餐信息、实名信息、身份验证等。

你的任务：
1. 如果用户只是打招呼或闲聊，自然地回应并简单介绍你的功能
2. 识别用户想要查询的类型，并检查是否提供了所需的全部信息
3. 根据查询类型要求不同的参数

查询类型和所需参数：
- package（查询套餐）: 需要手机号（11位）
- realname（查询实名信息）: 需要手机号（11位）
- identity（验证身份三要素）: 需要手机号（11位）+ 姓名 + 身份证号（18位）
- all（查询所有信息）: 需要手机号（11位）

【重要】查询类型识别规则（必须严格执行）：
第一步：检查用户输入中是否包含关键词
- 包含"套餐"或"package"或"meal" → 类型是 package
- 包含"实名"或"realname"或"认证" → 类型是 realname
- 包含"验证"或"identity"或"三要素" → 类型是 identity
- 不包含以上任何关键词，只有手机号 → 类型是 all

第二步：检查是否有手机号
- 有手机号 → complete: true
- 没有手机号 → 返回文字提示

【错误示例】：
❌ 用户输入"查询用户套餐信息" → 识别为 all（错误！应该是 package）
❌ 用户输入"17775711190的套餐" → 识别为 all（错误！应该是 package）

【正确示例】：
✓ 用户输入"查询用户套餐信息" → 包含"套餐" → package 类型，但缺少手机号 → 返回文字提示
✓ 用户输入"17775711190的套餐" → 包含"套餐" → package 类型，有手机号 → {{"complete": true, "query_type": "package", ...}}
✓ 用户输入"17775711190" → 不包含关键词 → all 类型，有手机号 → {{"complete": true, "query_type": "all", ...}}

当用户提供了完整信息时，返回JSON格式：
{{"complete": true, "query_type": "类型", "params": {{"phone": "手机号", "name": "姓名", "id_card": "身份证号"}}}}

当信息不完整时，自然地用文字回复，告知缺少什么信息。

示例（严格遵守）：
用户："你好" 
→ 回复："您好！我可以帮您查询用户套餐、实名信息、身份验证等。请告诉我您的手机号码。"

用户："查询我的套餐" 
→ 回复："好的，请提供您的11位手机号码。"

用户："查询13800138000的套餐" 
→ 回复：{{"complete": true, "query_type": "package", "params": {{"phone": "13800138000"}}}}

用户："查询用户套餐信息"
→ 回复："好的，请提供您的11位手机号码。"

用户："17775711190 查询套餐"
→ 回复：{{"complete": true, "query_type": "package", "params": {{"phone": "17775711190"}}}}

用户："查询17775711190的套餐信息"
→ 回复：{{"complete": true, "query_type": "package", "params": {{"phone": "17775711190"}}}}

用户："17775711190"（只有手机号，没有任何关键词）
→ 回复：{{"complete": true, "query_type": "all", "params": {{"phone": "17775711190"}}}}

用户："17775711190的套餐"（包含"套餐"关键词）
→ 回复：{{"complete": true, "query_type": "package", "params": {{"phone": "17775711190"}}}}

用户："查询17775711190的实名信息"
→ 回复：{{"complete": true, "query_type": "realname", "params": {{"phone": "17775711190"}}}}

用户："验证13800138000的身份" 
→ 回复："好的，除了手机号，我还需要您的姓名和身份证号码来验证身份三要素。请提供：\\n1. 姓名\\n2. 身份证号（18位）"

用户："验证13800138000，张三，110101199001011234" 
→ 回复：{{"complete": true, "query_type": "identity", "params": {{"phone": "13800138000", "name": "张三", "id_card": "110101199001011234"}}}}

注意事项（必须严格遵守）：
- 文本中只要出现"套餐"就必须是 package 类型，不能是 all
- 文本中只要出现"实名"就必须是 realname 类型，不能是 all  
- 文本中只要出现"验证"就必须是 identity 类型，不能是 all
- 只有纯手机号（没有任何关键词）才使用 all 类型
- 身份验证必须同时提供手机号、姓名、身份证号三项信息
- 其他查询类型只需要手机号
"""

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=0,
    max_retries=3,
    request_timeout=30
)

prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_SYSTEM_PROMPT),
    ("human", "{user_input}")
])

chain = prompt | llm

def process_intent(state: AgentState) -> AgentState:
    """处理用户意图"""
    logger.separator()
    logger.info("开始意图理解")
    logger.info(f"用户输入: {state['user_input']}")
    
    try:
        logger.debug("调用 LLM 进行意图分析...")
        start_time = time.time()
        
        response = chain.invoke({"user_input": state["user_input"]})
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"LLM 响应耗时: {elapsed:.0f}ms")
        
        import json
        # 处理响应内容
        content = response.content
        if isinstance(content, list):
            content = content[0] if content else ""
            if hasattr(content, 'text'):
                content = content.text
        
        content_str = str(content).strip()
        logger.llm_output(content_str)
        
        # 尝试解析为 JSON
        try:
            result = json.loads(content_str)
            logger.debug(f"解析结果: {result}")
            
            # 如果是完整的查询请求
            if result.get("complete"):
                state["is_complete"] = True
                state["user_info"] = result
                state["next_step"] = "query"
                logger.info("✓ 意图完整，准备查询数据")
                logger.debug(f"查询类型: {result.get('query_type')}, 参数: {result.get('params')}")
            else:
                # 不完整，返回提示
                state["is_complete"] = False
                state["error"] = result.get("response") or result.get("suggestion") or content_str
                state["next_step"] = "end"
                logger.info("✓ 意图不完整，返回提示")
                logger.debug(f"提示内容: {state['error'][:100]}")
        
        except json.JSONDecodeError:
            # 不是 JSON，说明是自然语言回复（问候、闲聊等）
            state["is_complete"] = False
            state["error"] = content_str
            state["next_step"] = "end"
            logger.info("✓ 自然语言回复（非查询）")
            logger.debug(f"回复内容: {content_str[:100]}")
        
        return state
        
    except Exception as e:
        logger.error(f"意图理解失败:\n{traceback.format_exc()}")
        
        # 如果是 API 错误
        error_msg = str(e)
        if "500" in error_msg or "InternalServerError" in error_msg:
            state["error"] = "抱歉，服务暂时不可用，请稍后重试。"
        else:
            state["error"] = "您好！我可以帮您查询用户信息。请提供您的11位手机号码。"
        
        state["is_complete"] = False
        state["next_step"] = "end"
        return state
