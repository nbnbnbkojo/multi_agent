from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()


class QwenModel:
    def __init__(self, model_name="qwen-plus"):
        """
        初始化 Qwen-Plus 纯文本模型
        :param model_name: 模型名称，默认 "qwen-plus"
        """
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model = ChatTongyi(
            model=model_name,
            model_name=model_name,
            api_key=self.api_key,
            temperature=0.1,  # 温度参数，值越小输出越稳定
            top_p=0.9  # 可选：核采样参数，控制输出多样性
        )

    # 通用非流式调用方法（纯文本）
    def invoke(self, prompt: str, system_prompt: str = "") -> str:
        """
        非流式调用模型
        :param prompt: 用户文本提问
        :param system_prompt: 系统提示词（可选，用于设置模型角色）
        :return: 模型完整回复文本
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"❌ 调用失败：{str(e)}")
            return ""

    # 流式调用方法（纯文本，用于前端实时输出）
    def stream(self, prompt: str, system_prompt: str = ""):
        """
        流式调用模型（逐字返回回复）
        :param prompt: 用户文本提问
        :param system_prompt: 系统提示词（可选）
        :yield: 模型回复片段
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            for chunk in self.model.stream(messages):
                yield chunk.content
        except Exception as e:
            print(f"❌ 流式调用失败：{str(e)}")
