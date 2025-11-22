"""
ChromaDB向量数据库存储实现
使用ChromaDB本地向量数据库替代Qdrant
"""

import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from chromadb import Client
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

logger = logging.getLogger(__name__)


class ChromaConnectionManager:
    """Chroma连接管理器 - 防止重复初始化"""
    _instances = {}

    @classmethod
    def get_instance(
            cls,
            collection_name: str = "hello_agents_vectors",
            persist_path: str = "./memory_data/chroma_store",
            vector_size: int = 384,
            **kwargs
    ) -> 'ChromaVectorStore':
        key = collection_name
        if key not in cls._instances:
            cls._instances[key] = ChromaVectorStore(
                collection_name=collection_name,
                persist_path=persist_path,
                vector_size=vector_size,
                **kwargs
            )
        return cls._instances[key]


class ChromaVectorStore:
    """ChromaDB向量数据库存储实现"""

    def __init__(
            self,
            collection_name: str = "hello_agents_vectors",
            persist_path: str = "./memory_data/chroma_store",
            vector_size: int = 384,
    ):
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "chromadb未安装。请运行: pip install chromadb"
            )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.persist_path = persist_path

        self.client = Client(
            Settings(
                persist_directory=self.persist_path,
                is_persistent=True,
                anonymized_telemetry=False  # 关闭遥测
            )
        )

        # 尝试获取已有集合，否则创建新集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"✅ 使用现有Chroma集合: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"vector_size": vector_size}
            )
            logger.info(f"✅ 创建新的Chroma集合: {collection_name}")

    def add_vectors(
            self,
            vectors: List[List[float]],
            metadata: List[Dict[str, Any]],
            ids: Optional[List[str]] = None
    ) -> bool:
        """添加向量到Chroma"""
        try:
            if not vectors:
                logger.warning("⚠️ 向量列表为空")
                return False

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(vectors))]

            # 确保metadata带时间戳
            for meta in metadata:
                meta["timestamp"] = int(datetime.now().timestamp())
                meta["added_at"] = int(datetime.now().timestamp())
                if "external" in meta and not isinstance(meta.get("external"), bool):
                    val = meta.get("external")
                    meta["external"] = str(val).lower() in ("1", "true", "yes")

            self.collection.add(
                embeddings=vectors,
                metadatas=metadata,
                ids=ids
            )
            logger.info(f"✅ 成功添加 {len(vectors)} 个向量到Chroma")
            return True
        except Exception as e:
            logger.error(f"❌ 添加向量失败: {e}")
            return False

    def search_similar(
            self,
            query_vector: List[float],
            limit: int = 10,
            score_threshold: Optional[float] = None,
            where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            query_kwargs = {}
            if where:
                query_kwargs["where"] = where

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                **query_kwargs
            )

            hits = []
            for i, id_ in enumerate(results.get("ids", [[]])[0]):
                score = results.get("distances", [[]])[0][i] if "distances" in results else None
                metadata = results.get("metadatas", [[]])[0][i] if "metadatas" in results else {}
                if score_threshold is not None and score is not None and score < score_threshold:
                    continue
                hits.append({"id": id_, "score": score, "metadata": metadata})

            return hits
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def delete_vectors(self, ids: List[str]) -> bool:
        """删除向量"""
        try:
            if not ids:
                return True
            self.collection.delete(ids=ids)
            logger.info(f"✅ 成功删除 {len(ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"❌ 删除向量失败: {e}")
            return False

    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"vector_size": self.vector_size}
            )
            logger.info(f"✅ 成功清空Chroma集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ 清空集合失败: {e}")
            return False

    def delete_memories(self, memory_ids: List[str]):
        """按memory_id删除"""
        try:
            for mid in memory_ids:
                results = self.collection.query(where={"memory_id": mid}, n_results=1000)
                ids_to_delete = results.get("ids", [[]])[0]
                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
            logger.info(f"✅ 成功按memory_id删除 {len(memory_ids)} 个向量")
        except Exception as e:
            logger.error(f"❌ 删除记忆失败: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "vectors_count": count,
                "config": {"vector_size": self.vector_size}
            }
        except Exception as e:
            logger.error(f"❌ 获取集合信息失败: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.get_collection_info()
        info["store_type"] = "chroma"
        return info

    def health_check(self) -> bool:
        """健康检查"""
        try:
            _ = self.client.list_collections()
            return True
        except Exception as e:
            logger.error(f"❌ Chroma健康检查失败: {e}")
            return False
