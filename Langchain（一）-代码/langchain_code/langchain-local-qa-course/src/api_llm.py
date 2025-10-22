"""测试 langserve 怎么使用"""

from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI

from langserve import add_routes
from dotenv import dotenv_values

ENV_CONFIG = dotenv_values("../.env")
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    ChatOpenAI(api_key=ENV_CONFIG.get("API_KEY"), base_url=ENV_CONFIG.get("BASE_URL")),
    path="/openai",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
