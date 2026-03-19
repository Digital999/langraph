"""RAG 模块"""
from rag.volcengine_embeddings import VolcEngineMultimodalEmbeddings
from rag.vector_store_milvus import (
    MilvusVectorStoreManager,
    get_milvus_store,
    init_milvus_knowledge_base,
)

VectorStoreManager = MilvusVectorStoreManager
get_vector_store = get_milvus_store
init_knowledge_base = init_milvus_knowledge_base

__all__ = [
    "VolcEngineMultimodalEmbeddings",
    "VectorStoreManager",
    "get_vector_store",
    "init_knowledge_base",
    "MilvusVectorStoreManager",
    "get_milvus_store",
    "init_milvus_knowledge_base",
]
