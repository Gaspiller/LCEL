# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：ai-langchain-tutorial 
@File    ：langchain入门.py
@Author  ：dailaoban
@Date    ：2024/11/29 10:59 
"""
from dotenv import load_dotenv
import os


class langchainTest():

    def __init__(self):
        # 加载 .env 文件;安装 python-dotenv 读取 .env 环境变量文件
        env = load_dotenv()  # 它用于从项目根目录下的 .env 文件中加载环境变量。
        # 这个库允许你将敏感信息（如数据库密码、API密钥等）存储在 .env 文件中，而不是直接在代码中硬编码这些信息，这样做可以提高安全性和灵活性。
        print(env)
        # 读取环境变量
        self.API_BASE_URL = os.getenv("API_BASE_URL")
        self.API_KEY = os.getenv("API_KEY")

        print("API_BASE_URL:", self.API_BASE_URL)
        print("API_KEY:", self.API_KEY)

    def createModel(self):
        """创建 langchain-openai 模型调用对象"""
        from langchain_openai import ChatOpenAI

        CHAT_MODEL = "qwen2.5-instruct"
        self.model = ChatOpenAI(
            model=CHAT_MODEL,
            base_url=f'{self.API_BASE_URL}/v1',
            api_key=self.API_KEY,
            temperature=0,
            streaming=True
        )

    def chat(self):
        """直接调用大模型推理"""

        from langchain_core.messages import HumanMessage, SystemMessage

        self.messages = [
            SystemMessage(content="Translate and explain the following from English into Chinese"),
            HumanMessage(content="What can Langchain be used for by AI engineers?"),
        ]

        res = self.model.invoke(self.messages)
        print(res)

    def lctl_test(self):
        """使用管道方式调用格式化输出"""

        from langchain_core.output_parsers import StrOutputParser
        self.parser = StrOutputParser()
        result = self.model.invoke(self.messages)
        p1 = self.parser.invoke(result)
        print(p1)

        self.chain = self.model | self.parser
        p2 = self.chain.invoke(self.messages)
        print(p2)

    def prompts_example(self):
        """创建 langchain-core 提示词模版调用"""
        from langchain_core.prompts import ChatPromptTemplate
        system_template = "Translate and explain the following from English into {language}:"
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("human", "{text}")]
        )

        result = prompt_template.invoke({"language": "Chinese", "text": "What can Langchain be used for by AI engineers?"})
        re = result.to_messages()
        print(re)

        chain = prompt_template | self.model | self.parser
        chain = chain.invoke({"language": "Chinese", "text": "What can Langchain be used for by AI engineers?"})
        print(chain)


if __name__ == '__main__':
    lt = langchainTest()
    lt.createModel()
    lt.chat()
    lt.lctl_test()
    lt.prompts_example()
