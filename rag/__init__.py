"""RAG 模块"""
# 使用 Milvus 向量数据库
from rag.vector_store_milvus import (
    MilvusVectorStoreManager,
    get_milvus_store,
    init_milvus_knowledge_base
)

# 为了保持向后兼容，提供别名
VectorStoreManager = MilvusVectorStoreManager
get_vector_store = get_milvus_store
init_knowledge_base = init_milvus_knowledge_base

__all__ = [
    "VectorStoreManager",
    "get_vector_store",
    "init_knowledge_base",
    "MilvusVectorStoreManager",
    "get_milvus_store",
    "init_milvus_knowledge_base"
]
