import os

from sentence_transformers import SentenceTransformer

from Multi_Agent.agent.milvus_agent import MilvusConnector
from Multi_Agent.models.qwen_model import QwenModel
from Multi_Agent.prompts.rag_prompt import RAG_PROMPT
from Multi_Agent.agent.retriever_agent import HybridRetriever


class RAGAgent:
    def __init__(self):
        self.milvus_agent = MilvusConnector()
        self.RetrieverAgent = HybridRetriever()
        self.qwen = QwenModel()
        self.LOCAL_MODEL_PATH = "bge-small-zh-v1.5"
        self.embedding_model = self.load_local_embedding_model()
        connect_success, connect_msg = self.milvus_agent.connect()
        print(f"Milvus 初始化状态: {connect_msg}")

    def load_local_embedding_model(self):
        """
        本地加载Embedding模型（避免联网下载，提升加载速度）
        """
        try:
            # 检查本地模型路径是否存在
            if os.path.exists(self.LOCAL_MODEL_PATH):
                print(f"✅ 从本地加载Embedding模型: {self.LOCAL_MODEL_PATH}")
                # 本地加载模式（不检查更新、不联网）
                model = SentenceTransformer(
                    self.LOCAL_MODEL_PATH,
                    device="cpu"  # 可选：指定设备，如"cuda"（有GPU时）/ "cpu"
                )
            else:
                # 兜底：如果本地没有，先下载到指定路径（仅首次）
                print(f"⚠️ 本地模型路径不存在，先下载模型到: {self.LOCAL_MODEL_PATH}")
                model = SentenceTransformer(
                    'bge-small-zh-v1.5',
                    cache_folder=self.LOCAL_MODEL_PATH  # 下载到指定本地路径
                )
            return model
        except Exception as e:
            raise RuntimeError(f"❌ 加载Embedding模型失败: {str(e)}")

    def get_query_embedding(self, question):
        """生成查询向量（本地模型版）"""
        try:
            # 使用本地加载的模型生成embedding
            embedding = self.embedding_model.encode(
                question,
                normalize_embeddings=True  # 归一化，提升检索效果
            )
            return embedding
        except Exception as e:
            raise RuntimeError(f"❌ 生成查询向量失败: {str(e)}")

    # RAG问答（结合知识库）
    def rag_answer(self, question: str) -> str:
        """
        RAG 核心问答逻辑
        :param question: 用户问题
        :return: 结合知识库的回答
        """
        query_embedding = self.get_query_embedding(question)
        context = self.RetrieverAgent.hybrid_search(question, query_embedding)

        if not context:
            context = "未找到《食品安全法》相关参考资料"

        # 2. 构造 RAG 提示词
        prompt = RAG_PROMPT.format(context=context, question=question)

        # 3. 调用 Qwen 模型生成回答
        try:
            answer = self.qwen.invoke(prompt)
            return answer
        except Exception as e:
            return f"❌ 回答生成失败: {str(e)}"


