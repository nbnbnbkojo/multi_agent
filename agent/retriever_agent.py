import os
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from pymilvus import MilvusException
from dotenv import load_dotenv
from milvus_agent import MilvusConnector

# -------------------------- 配置项 --------------------------
load_dotenv()
# 向量检索配置
VECTOR_DIM = 512
TOP_K_VECTOR = 10  # 向量检索召回数
TOP_K_KEYWORD = 10  # 关键词检索召回数
MILVUS_EMBEDDING_FIELD = os.getenv("MILVUS_EMBEDDING_FIELD", "embedding")  # 向量字段名
MILVUS_TEXT_FIELD = os.getenv("MILVUS_TEXT_FIELD", "text")  # 文本字段名
MILVUS_ID_FIELD = os.getenv("MILVUS_ID_FIELD", "id")  # ID 字段名
# 重排序模型配置（本地模型路径）
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "bge-reranker-large")  # 重排序模型
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", 0.7))  # 重排序分数阈值

# -------------------------- 日志配置 --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("HybridRetriever")


# -------------------------- 混合检索核心类 --------------------------
class HybridRetriever:
    """混合检索器：向量检索 + 关键词检索 + 本地重排序"""

    def __init__(self):
        # 初始化 Milvus 连接器
        self.milvus_connector = MilvusConnector()
        # 连接 Milvus
        self._init_milvus_connection()
        # 初始化重排序模型（本地加载）
        self.rerank_model = self._load_rerank_model()
        self.logger = logging.getLogger(__name__)
        self.top_k_hybrid = 10

    def _init_milvus_connection(self) -> None:
        """初始化 Milvus 连接"""
        try:
            success, msg = self.milvus_connector.connect()
            if not success:
                raise Exception(msg)
            logger.info(f"Milvus 初始化成功: {msg}")
        except Exception as e:
            logger.error(f"❌ Milvus 初始化失败: {str(e)}")
            raise

    def _load_rerank_model(self):
        """加载本地重排序模型（使用 FlagEmbedding）"""
        try:
            from FlagEmbedding import FlagReranker
            # 加载本地重排序模型
            rerank_model = FlagReranker(
                RERANK_MODEL_PATH,
                use_fp16=True  # 半精度加速（可选）
            )
            logger.info(f"✅ 重排序模型加载成功: {RERANK_MODEL_PATH}")
            return rerank_model
        except ImportError:
            logger.error("❌ 缺少 FlagEmbedding 依赖，请执行：pip install FlagEmbedding")
            raise
        except Exception as e:
            logger.error(f"❌ 重排序模型加载失败: {str(e)}")
            raise

    def _vector_search(self, query_emb: List[float]) -> List[Dict]:
        """向量检索"""
        try:
            # 1. 校验512维度
            if len(query_emb) != 512:
                self.logger.error(f"❌ 向量维度错误：要求512维，实际{len(query_emb)}维")
                return []

            # 2. 执行Milvus检索
            search_res = self.milvus_connector.collection.search(
                data=[query_emb],
                anns_field=MILVUS_EMBEDDING_FIELD,
                param={"metric_type": "L2", "params": {"nprobe": 10}},
                limit=self.top_k_hybrid,
                output_fields=[MILVUS_TEXT_FIELD]
            )

            # 3. 解析结果
            results = []
            if search_res and len(search_res) > 0:
                for hit in search_res[0]:
                    # 核心修复：转成字典后取值，彻底避开 Hit.get()
                    hit_dict = hit.to_dict()
                    text = hit_dict.get("entity", {}).get(MILVUS_TEXT_FIELD, "").strip()

                    results.append({
                        "id": hit_dict["id"],
                        "text": text,
                        "distance": hit_dict["distance"],
                        "type": "vector"
                    })

            self.logger.info(f"✅ 向量检索完成，召回 {len(results)} 条结果")
            return results

        except Exception as e:
            self.logger.error(f"❌ 向量检索异常: {str(e)}")
            # 打印完整错误堆栈（方便排查）
            import traceback
            self.logger.error(f"❌ 错误详情：\n{traceback.format_exc()}")
            return []

    def _keyword_search(self, query_text: str) -> List[Dict]:
        """关键词检索"""
        try:
            # 步骤1：查询所有非空文本（避开Milvus的like限制）
            # 先查100条兜底，足够覆盖你的召回需求
            expr = f"{MILVUS_TEXT_FIELD} != ''"  # 只查非空文本
            search_res = self.milvus_connector.collection.query(
                expr=expr,
                output_fields=[MILVUS_TEXT_FIELD, MILVUS_ID_FIELD],
                limit=100  # 按需调整，小数据集直接查全量
            )

            # 步骤2：本地模糊匹配（支持任意位置的关键词）
            results = []
            for item in search_res:
                doc_text = item.get(MILVUS_TEXT_FIELD, "").strip()
                doc_id = item.get(MILVUS_ID_FIELD, "")

                # 核心：只要文档包含关键词（任意位置），就纳入结果
                if query_text in doc_text and len(results) < self.top_k_hybrid:
                    results.append({
                        "id": doc_id,
                        "text": doc_text,
                        "distance": 0.0,  # 关键词检索无距离，默认0
                        "type": "keyword"
                    })

            self.logger.info(f"✅ 关键词检索完成，召回 {len(results)} 条结果")
            return results

        except Exception as e:
            self.logger.error(f"❌ 关键词检索失败: {str(e)}")
            return []

    def _rerank_results(self, query_text: str, candidate_docs: List[Dict]) -> Optional[Dict]:
        """
        重排序核心逻辑：使用本地模型对候选文档重排序，返回最准确的 1 条
        :param query_text: 查询文本
        :param candidate_docs: 候选文档列表
        :return: 最准确的文档（None 表示无符合阈值的结果）
        """
        if not candidate_docs:
            logger.warning("⚠️ 无候选文档，跳过重排序")
            return None

        try:
            # 构建重排序输入：[(查询文本, 文档文本), ...]
            rerank_pairs = [(query_text, doc["text"]) for doc in candidate_docs if doc["text"]]
            if not rerank_pairs:
                logger.warning("⚠️ 无有效文本文档，跳过重排序")
                return None

            # 执行重排序
            scores = self.rerank_model.compute_score(rerank_pairs)

            # 合并分数到候选文档
            scored_docs = []
            for idx, doc in enumerate(candidate_docs):
                if doc["text"]:  # 过滤空文本
                    scored_docs.append({
                        **doc,
                        "rerank_score": scores[idx]
                    })

            # 按重排序分数降序排列
            scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

            # 筛选符合阈值的最优文档
            best_doc = None
            for doc in scored_docs:
                if doc["rerank_score"] >= RERANK_THRESHOLD:
                    best_doc = doc
                    break

            if best_doc:
                logger.info(
                    f"✅ 重排序完成 | 最优文档 ID: {best_doc['id']} | 分数: {best_doc['rerank_score']:.4f}"
                )
            else:
                logger.warning(f"⚠️ 无符合阈值（{RERANK_THRESHOLD}）的文档")

            return best_doc

        except Exception as e:
            logger.error(f"❌ 重排序失败: {str(e)}")
            # 降级策略：返回候选文档中第一条
            return candidate_docs[0] if candidate_docs else None

    def hybrid_search(self, query_text: str, query_embedding: List[float]) -> Optional[Dict]:
        """
        混合检索主入口
        :param query_text: 查询文本（用于关键词检索 + 重排序）
        :param query_embedding: 查询向量（用于向量检索）
        :return: 最准确的文档（None 表示检索失败）
        """
        try:
            # 1. 多路召回：向量检索 + 关键词检索
            vector_docs = self._vector_search(query_embedding)
            keyword_docs = self._keyword_search(query_text)

            # 2. 合并去重（基于 ID）
            all_docs = []
            doc_ids = set()
            # 先加向量检索结果
            for doc in vector_docs:
                if doc["id"] not in doc_ids and doc["text"]:
                    doc_ids.add(doc["id"])
                    all_docs.append(doc)
            # 再加关键词检索结果（去重）
            for doc in keyword_docs:
                if doc["id"] not in doc_ids and doc["text"]:
                    doc_ids.add(doc["id"])
                    all_docs.append(doc)

            if not all_docs:
                logger.warning("❌ 多路召回无有效文档")
                return None

            # 3. 重排序：返回最准确的 1 条
            best_doc = self._rerank_results(query_text, all_docs)
            return best_doc

        except Exception as e:
            logger.error(f"❌ 混合检索失败: {str(e)}")
            return None

    def close(self) -> None:
        """释放资源：断开 Milvus 连接"""
        try:
            success, msg = self.milvus_connector.disconnect()
            if success:
                logger.info(msg)
            else:
                logger.error(msg)
        except Exception as e:
            logger.error(f"❌ 断开 Milvus 连接失败: {str(e)}")