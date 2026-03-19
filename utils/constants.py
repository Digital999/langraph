"""项目常量定义"""

# 对话历史配置
MAX_CONVERSATION_TURNS = 10  # 最大保留对话轮数
MAX_HISTORY_DISPLAY_TURNS = 3  # 显示给 LLM 的最大历史轮数

# 报告确认关键词
CONFIRM_KEYWORDS = ["是", "需要", "生成", "报告", "yes", "ok", "好"]
REJECT_KEYWORDS = ["不", "不需要", "不用", "取消", "no", "cancel"]

# 快速响应缓存
QUICK_RESPONSES = {
    "你好": "您好！我是智能查询助手，可以帮您查询用户套餐、实名信息、身份验证等。请提供您的11位手机号码开始查询。",
    "hi": "Hello! I'm an intelligent query assistant. Please provide your 11-digit phone number to start the query.",
    "hello": "Hello! I'm an intelligent query assistant. Please provide your 11-digit phone number to start the query.",
    "您好": "您好！我是智能查询助手，可以帮您查询用户套餐、实名信息、身份验证等。请提供您的11位手机号码开始查询。",
}

# 套餐状态映射
MEAL_STATUS_MAP = {
    0: "未激活",
    2: "使用中",
    6: "已退租"
}

# RAG 配置
RAG_TOP_K = 3                       # 检索 Top-K 文档
RAG_SCORE_THRESHOLD = 1.0           # L2 距离阈值（越小越相关，超过此值视为不相关）
RAG_LLM_TEMPERATURE = 0.2          # RAG 回答生成温度
RAG_MAX_RETRIEVAL_RETRIES = 3       # 检索失败后最大重试轮数

# LLM 配置
LLM_TEMPERATURE = 0  # LLM 温度
LLM_MAX_RETRIES = 3  # LLM 最大重试次数
LLM_TIMEOUT = 60  # LLM 超时时间（秒）

# 报告生成配置
REPORT_LLM_TEMPERATURE = 0.3  # 报告生成 LLM 温度
