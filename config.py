"""配置文件"""
import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""
    # OpenAI配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4")
    
    # Embedding 模型配置（留空则复用 OPENAI 配置）
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    @property
    def effective_embedding_api_key(self) -> str:
        return self.EMBEDDING_API_KEY or self.OPENAI_API_KEY
    
    @property
    def effective_embedding_base_url(self) -> str:
        return self.EMBEDDING_BASE_URL or self.OPENAI_BASE_URL
    
    # 数据库配置
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_NAME: str = os.getenv("DB_NAME", "user_db")
    
    # Milvus 向量数据库配置
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER: str = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "knowledge_base")
    
    # PostgreSQL 配置（用于 LangGraph 长期记忆）
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "langgraph_memory")
    
    @property
    def POSTGRES_URI(self) -> str:
        """构建 PostgreSQL 连接 URI"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # 第三方接口配置
    IDENTITY_API_URL: str = os.getenv("IDENTITY_API_URL", "")
    IDENTITY_API_KEY: str = os.getenv("IDENTITY_API_KEY", "")
    
    class Config:
        """Pydantic配置"""
        env_file = ".env"
        case_sensitive = True


settings = Settings()
