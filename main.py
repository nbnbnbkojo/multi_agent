import logging
from datetime import datetime
from typing import Optional, Tuple

# 本地模块导入（保持路径正确）
from prompts.rag_prompt import RAG_PROMPT
from agent.domain_agent import DomainAgent
from agent.rag_agent import RAGAgent
from agent.bocha_agent import BoChaAgent
from models.qwen_model import QwenModel
from agent.milvus_agent import MilvusConnector

# -------------------------- 全局初始化 --------------------------
domain_agent = DomainAgent()
rag_agent = RAGAgent()
bocha_agent = BoChaAgent()
qwen = QwenModel()
milvus_agent = MilvusConnector()

# 基础日志配置（仅记录关键错误）
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qa_error.log"), logging.StreamHandler()]
)
logger = logging.getLogger("smart_qa")


# -------------------------- 核心问答功能 --------------------------
def chat(question: str, return_domain: bool = False) -> str | Tuple[str, str]:
    """
    非流式问答核心功能 - 常规返回格式
    :param question: 用户问题
    :param return_domain: 是否返回领域标识，默认False（仅返回回答文本）
    :return: 仅返回纯回答文本 | 若return_domain=True，返回(回答文本, 领域)元组
    """
    try:
        # 领域识别
        is_food_safety = domain_agent.is_food_safety_domain(question)
        domain = "食品安全法" if is_food_safety else "通用领域"

        # 分支处理
        if is_food_safety:
            context = rag_agent.milvus_agent.similarity_search(question)
            prompt = RAG_PROMPT.format(context=context, question=question)
            answer = qwen.invoke(prompt)
        else:
            search_result = bocha_agent.bocha(question)
            answer = qwen.invoke(
                f"根据以下搜索结果，用动漫风格的轻松语气回答用户问题：\n"
                f"搜索结果：{search_result}\n用户问题：{question}"
            )

        # 结果兜底
        final_answer = answer.strip() if answer and answer.strip() else "抱歉，暂时无法回答这个问题哦😜"
        # 常规返回：纯文本 / 按需返回(文本, 领域)
        return (final_answer, domain) if return_domain else final_answer
    except Exception as e:
        error_info = f"😥 问答出错啦：{str(e)}"
        logger.error(f"问答失败 | 问题：{question} | 错误：{e}", exc_info=True)
        return error_info


def chat_stream(question: str) -> Optional[str]:
    """
    流式问答功能 - 控制台实时打印，最终返回纯文本
    :param question: 用户问题
    :return: 完整纯回答文本（失败返回None）
    """
    try:
        is_food_safety = domain_agent.is_food_safety_domain(question)
        print(f"\n[问答领域]：{'食品安全法' if is_food_safety else '通用领域'} | [时间]：{datetime.now().strftime('%H:%M:%S')}")
        print("回答：", end="", flush=True)

        # 拼接Prompt
        if is_food_safety:
            context = rag_agent.milvus_agent.similarity_search(question)
            prompt = f"""你是《食品安全法》专业问答助手，根据以下参考资料用动漫风格的轻松语气回答：
参考资料：{context}
用户问题：{question}"""
        else:
            search_result = bocha_agent.bocha(question)
            prompt = f"""根据以下搜索结果，用动漫风格的轻松语气回答：
搜索结果：{search_result}
用户问题：{question}"""

        # 流式打印+拼接完整回答
        full_answer = ""
        for chunk in qwen.stream(prompt):
            print(chunk, end="", flush=True)
            full_answer += chunk

        print()  # 换行收尾
        return full_answer.strip() if full_answer else None
    except Exception as e:
        error_info = f"\n😥 问答出错啦：{str(e)}"
        print(error_info)
        logger.error(f"流式问答失败 | 问题：{question} | 错误：{e}", exc_info=True)
        return None


# -------------------------- 测试入口 --------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("🍻 智能问答助手".center(80))
    print("=" * 80)
    print("📌 功能：1-非流式问答（直接返回文本）  2-流式问答（控制台实时打印）")
    print("📌 输入q退出，支持食品安全法专业问答+通用领域问答")
    print("=" * 80)

    # 测试示例
    test_food_question = "《食品安全法》中关于举报食品安全违法行为有哪些规定？"
    test_general_question = "什么是人工智能？"

    while True:
        choice = input("\n请输入功能编号（1/2，输入q退出）：").strip()
        if choice.lower() == "q":
            print("👋 退出程序，感谢使用！")
            break
        elif choice == "1":
            # 非流式问答 - 直接返回纯文本，调用更直观
            q = input(f"\n请输入问题（示例：{test_food_question[:30]}...）：").strip()
            if not q:
                q = test_food_question
            # 极简调用：直接获取回答文本
            answer = chat(q)
            print(f"\n✅ 回答：{answer}")

        elif choice == "2":
            # 流式问答 - 控制台实时打印，最终返回纯文本
            q = input(f"\n请输入问题（示例：{test_general_question[:30]}...）：").strip()
            if not q:
                q = test_general_question
            chat_stream(q)
        else:
            print("❌ 输入错误，请输入1/2或q！")