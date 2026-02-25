"""报告生成 Agent - 生成 Word 报告"""
from docx import Document
from datetime import datetime
import os
from models.schemas import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import settings
from utils.logger import logger
from utils.constants import REPORT_LLM_TEMPERATURE, LLM_MAX_RETRIES, LLM_TIMEOUT
import traceback

REPORT_SYSTEM_PROMPT = """你是一个专业的报告生成助手，负责将查询结果整理成结构化的报告内容。

你的职责：
1. 分析查询结果，提取关键信息
2. 组织报告结构，包括：标题、摘要、详细信息、结论
3. 使用清晰、专业的语言描述数据
4. 对敏感信息进行适当处理
5. 生成易读的报告文本

报告结构要求：
- 报告标题：简洁明了
- 查询摘要：概述查询目的和范围
- 详细信息：分类展示查询结果
- 数据分析：对结果进行简要分析
- 结论建议：给出相关建议（如适用）

输出格式：
返回JSON格式：{{"title": "标题", "summary": "摘要", "details": "详细内容", "conclusion": "结论"}}
"""

llm = ChatOpenAI(
    model=settings.MODEL_NAME,
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,
    temperature=REPORT_LLM_TEMPERATURE,
    max_retries=LLM_MAX_RETRIES,
    timeout=LLM_TIMEOUT
)

prompt = ChatPromptTemplate.from_messages([
    ("system", REPORT_SYSTEM_PROMPT),
    ("human", "用户查询：{user_input}\n\n查询结果：{query_results}\n\n请生成报告内容。")
])

chain = prompt | llm

def generate_word_report(report_content: dict, filename: str) -> str:
    """生成 Word 报告文件"""
    doc = Document()
    
    # 标题
    title = doc.add_heading(report_content.get("title", "用户信息查询报告"), 0)
    title.alignment = 1  # 居中
    
    # 生成时间
    doc.add_paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph("")
    
    # 摘要
    doc.add_heading("查询摘要", 1)
    doc.add_paragraph(report_content.get("summary", ""))
    doc.add_paragraph("")
    
    # 详细信息
    doc.add_heading("详细信息", 1)
    details = report_content.get("details", "")
    
    # 如果 details 是字典，格式化输出
    if isinstance(details, dict):
        for section, content in details.items():
            doc.add_heading(section, 2)
            if isinstance(content, dict):
                for key, value in content.items():
                    doc.add_paragraph(f"{key}: {value}", style='List Bullet')
            elif isinstance(content, list):
                for item in content:
                    doc.add_paragraph(str(item), style='List Bullet')
            else:
                doc.add_paragraph(str(content))
            doc.add_paragraph("")
    else:
        doc.add_paragraph(str(details))
        doc.add_paragraph("")
    
    # 结论
    doc.add_heading("结论", 1)
    conclusion = report_content.get("conclusion", "")
    if isinstance(conclusion, dict):
        for key, value in conclusion.items():
            doc.add_paragraph(f"{key}: {value}")
    else:
        doc.add_paragraph(str(conclusion))
    
    # 保存文件
    os.makedirs("reports", exist_ok=True)
    filepath = f"reports/{filename}"
    doc.save(filepath)
    return filepath

def process_report(state: AgentState) -> AgentState:
    """生成报告"""
    logger.separator()
    logger.info("开始生成报告")
    
    if not state.get("query_results"):
        logger.error("缺少查询结果")
        state["error"] = "缺少查询结果"
        state["next_step"] = "end"
        return state
    
    logger.debug(f"查询结果: {state['query_results']}")
    
    try:
        # 使用 LLM 生成报告内容
        logger.debug("调用 LLM 生成报告内容...")
        start_time = datetime.now()
        
        response = chain.invoke({
            "user_input": state["user_input"],
            "query_results": str(state["query_results"])
        })
        
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"LLM 响应耗时: {elapsed:.0f}ms")
        
        import json
        # 处理响应内容，确保是字符串
        content = response.content
        if isinstance(content, list):
            content = content[0] if content else "{}"
            if hasattr(content, 'text'):
                content = content.text
        
        content_str = str(content).strip()
        logger.llm_output(content_str)
        
        # 清理 markdown 代码块标记
        if content_str.startswith("```json"):
            content_str = content_str[7:]  # 移除 ```json
        elif content_str.startswith("```"):
            content_str = content_str[3:]  # 移除 ```
        
        if content_str.endswith("```"):
            content_str = content_str[:-3]  # 移除结尾的 ```
        
        content_str = content_str.strip()
        
        report_content = json.loads(content_str)
        logger.debug(f"报告结构: {list(report_content.keys())}")
        
        # 生成 Word 文件
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        logger.info(f"生成 Word 文件: {filename}")
        
        filepath = generate_word_report(report_content, filename)
        
        state["report_path"] = filepath
        state["next_step"] = "end"
        logger.info(f"✓ 报告生成成功: {filepath}")
    
    except Exception as e:
        logger.error(f"报告生成失败:\n{traceback.format_exc()}")
        state["error"] = f"报告生成失败: {str(e)}"
        state["next_step"] = "end"
    
    return state
