import requests
from dotenv import load_dotenv
import os
from Multi_Agent.models.qwen_model import QwenModel

load_dotenv()


class BoChaAgent:
    def __init__(self):
        self.api_key = os.getenv("BOCHA_API_KEY")
        self.api_url = os.getenv("BOCHA_API_URL")
        self.qwen = QwenModel()

    # 搜索API调用
    def bocha(self, question: str) -> str:
        print(question)
        params = {
              "query": question,
              "summary": True,
              "count": 1
            }
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        print(params)
        try:
            response = requests.post(
                url=self.api_url,
                headers=headers,
                json=params,
                timeout=10  # 超时保护
            )
            response.raise_for_status()
            print(self.api_url)
            print("搜索成功")
            results = response.json()
            print(f"API原始返回：{results}")  # 调试用，查看完整返回结构

            data = results.get("data", {})
            web_pages = data.get("webPages", {})
            organic_results = web_pages.get("value", [])  # 这是真实的搜索结果列表
            # =====================================================

            if not isinstance(organic_results, list):
                organic_results = [organic_results]  # 兼容非列表格式

            # 拼接搜索结果（提取name=标题，snippet=摘要）
            answer = ""
            for idx, result in enumerate(organic_results[:3], 1):
                title = result.get("name", "无标题")  # 对应返回里的"name"字段
                snippet = result.get("snippet", result.get("summary", "无内容"))  # 对应"snippet"字段
                answer += f"【结果{idx}】{title}\n{snippet}\n\n"

            print(answer)

            if not answer :
                answer = "未找到相关搜索结果"
            return answer
        except Exception as e:
            return f"搜索失败：{str(e)}"