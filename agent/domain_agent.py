from Multi_Agent.models.qwen_model import QwenModel
from Multi_Agent.prompts.domain_prompt import DOMAIN_PROMPT

class DomainAgent:
    def __init__(self):
        self.qwen = QwenModel()

    # 判断是否属于食品安全法领域
    def is_food_safety_domain(self, question: str) -> bool:
        prompt = DOMAIN_PROMPT.format(question=question)
        response = self.qwen.invoke(prompt)
        return response.strip() == "是"