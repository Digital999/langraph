"""公共装饰器和工具函数"""
import time
from typing import Dict, List, Optional

from utils.logger import logger


class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, name: str, metrics_dict: Dict[str, float] | None = None):
        """
        初始化计时器
        
        Args:
            name: 计时器名称
            metrics_dict: 用于存储性能指标的字典
        """
        self.name = name
        self.metrics_dict = metrics_dict
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.time() - self.start_time) * 1000
        logger.debug(f"{self.name} 耗时: {self.elapsed_ms:.0f}ms")
        
        if self.metrics_dict is not None:
            self.metrics_dict[self.name] = self.elapsed_ms
        
        return False


def build_history_string(history: List[str], max_turns: int = 3) -> str:
    """
    构建对话历史字符串
    
    Args:
        history: 对话历史列表
        max_turns: 最大显示轮数
        
    Returns:
        格式化的对话历史字符串
    """
    if not history:
        return "（无历史对话）"
    
    # 只取最近的对话
    recent_history = history[-(max_turns * 2):]
    return "\n".join(recent_history)


_query_llm = None


def _get_query_llm():
    """延迟初始化查询改写用的 LLM"""
    global _query_llm
    if _query_llm is None:
        from langchain_openai import ChatOpenAI
        from config import settings
        from utils.constants import LLM_TEMPERATURE

        _query_llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            temperature=LLM_TEMPERATURE,
        )
    return _query_llm


def rewrite_query_with_history(
    query: str,
    history: List[str],
    max_turns: int = 3,
) -> str:
    """结合对话历史改写查询（第一轮，补全指代和省略）

    历史截取方式与 intent_agent / MCP 工具调用保持一致。
    如果没有历史对话，直接返回原始 query。
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    history_str = build_history_string(history, max_turns=max_turns)
    if history_str == "（无历史对话）":
        return query

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是查询改写助手。根据对话历史，将用户最新的问题改写为一个独立完整的查询语句，"
         "用于在知识库中检索。\n"
         "要求：只输出改写后的一句查询，不要任何解释。"
         "补全代词指代和省略的主语。如果问题本身已完整则原样输出。"),
        ("human", "对话历史：\n{history}\n\n最新问题：{question}"),
    ])
    chain = prompt | _get_query_llm() | StrOutputParser()

    try:
        rewritten = chain.invoke({
            "history": history_str,
            "question": query,
        }).strip()
        if rewritten:
            logger.info(f"查询改写[初始]: '{query}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"查询改写失败，使用原始查询: {e}")

    return query


def reformulate_failed_query(
    original_question: str,
    history: List[str],
    attempted_queries: List[str],
    max_turns: int = 3,
) -> str:
    """检索失败后重新改写查询（自适应重试）

    LLM 根据原始问题、对话历史和已失败的查询记录，
    从不同角度 / 关键词 / 粒度生成新的检索语句。

    Args:
        original_question: 用户原始问题
        history: 对话历史
        attempted_queries: 已尝试过但未命中的查询列表
        max_turns: 历史截取轮数（与其他模块保持一致）
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    history_str = build_history_string(history, max_turns=max_turns)
    attempts_str = "\n".join(
        f"  第{i}次: {q}" for i, q in enumerate(attempted_queries, 1)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是知识库检索优化助手。用户的问题在知识库中未检索到相关内容，"
         "你需要根据以下信息生成一个全新的检索查询语句。\n\n"
         "改写策略（按优先级尝试）：\n"
         "1. 提取问题的核心关键词，去掉修饰语\n"
         "2. 使用同义词或近义词替换关键术语\n"
         "3. 将具体问题泛化为更宽泛的主题\n"
         "4. 从不同业务角度重新描述\n\n"
         "要求：\n"
         "- 只输出一句新的检索查询，不要任何解释\n"
         "- 必须与已失败的查询不同\n"
         "- 保持与原始问题的语义相关性"),
        ("human",
         "原始问题：{original_question}\n\n"
         "对话历史：\n{history}\n\n"
         "已尝试过的查询（均未命中）：\n{attempts}\n\n"
         "请生成一个新的检索查询："),
    ])
    chain = prompt | _get_query_llm() | StrOutputParser()

    try:
        new_query = chain.invoke({
            "original_question": original_question,
            "history": history_str,
            "attempts": attempts_str,
        }).strip()
        if new_query and new_query not in attempted_queries:
            logger.info(
                f"查询改写[重试第{len(attempted_queries)+1}轮]: "
                f"'{attempted_queries[-1]}' → '{new_query}'"
            )
            return new_query
    except Exception as e:
        logger.warning(f"重试改写失败: {e}")

    return original_question
