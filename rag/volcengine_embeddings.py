"""火山引擎多模态 Embedding 封装（基于 volcenginesdkarkruntime）

将 Ark SDK 的 multimodal_embeddings 接口封装为 LangChain Embeddings 接口，
供 Milvus 向量数据库使用。
"""
from typing import List

from langchain_core.embeddings import Embeddings
from volcenginesdkarkruntime import Ark

from config import settings
from utils.logger import logger


class VolcEngineMultimodalEmbeddings(Embeddings):
    """火山引擎多模态 Embedding（仅使用文本模态用于 RAG）"""

    def __init__(
        self,
        model: str = "",
        api_key: str = "",
    ):
        self.model = model or settings.EMBEDDING_MODEL
        self.api_key = api_key or settings.effective_embedding_api_key
        self.client = Ark(api_key=self.api_key)
        logger.info(f"VolcEngine Embedding 初始化: model={self.model}")

    def _embed_text(self, text: str) -> List[float]:
        """单条文本向量化"""
        resp = self.client.multimodal_embeddings.create(
            model=self.model,
            input=[{"type": "text", "text": text}],
        )
        data = resp.data
        # 火山引擎返回单个 MultimodalEmbedding 对象，非列表
        if isinstance(data, list):
            return data[0].embedding
        return data.embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档"""
        embeddings = []
        total = len(texts)
        for i, text in enumerate(texts):
            if i == 0 or (i + 1) % 10 == 0 or i == total - 1:
                logger.info(f"Embedding 进度: {i + 1}/{total}")
            embeddings.append(self._embed_text(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """向量化查询文本"""
        return self._embed_text(text)
