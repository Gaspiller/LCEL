# uvicorn , nginx 相关的东西部署后端服务；
# Batch , 异步等

from loguru import logger
from dotenv import dotenv_values

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS

# 添加的
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

logger.info("init app succ")

ENV_CONFIG = dotenv_values("../.env")


def get_retriver() -> EnsembleRetriever:
    text_loader = TextLoader(
        "/Users/bytedance/work/person/langchain-local-qa/data/text.txt"
    )
    text_docs = text_loader.load()

    vectorstore = FAISS.from_documents(
        text_docs,
        OpenAIEmbeddings(
            api_key=ENV_CONFIG.get("API_KEY"), base_url=ENV_CONFIG.get("BASE_URL")
        ),
    )
    bm25_retriever = BM25Retriever.from_documents(documents=text_docs, k=1)
    emb_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, emb_retriever], weights=[0.5, 0.5]
    )
    return ensemble_retriever


def build_chain():
    # template = """Answer the question based only on the following context:
    # {context}

    # Question: {question}
    # """
    template = """Answer the question based on the following context.

    {context}

    Question: {question}

    If context is not useful. answer the question directly and Do not refer to context
    """

    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(
        api_key=ENV_CONFIG.get("API_KEY"), base_url=ENV_CONFIG.get("BASE_URL")
    )
    retriver = get_retriver()
    chain = (
        {"context": retriver, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain


def build_joke_chain():
    template = """帮我写一个关于{topic}的笑话"""
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(
        api_key=ENV_CONFIG.get("API_KEY"), base_url=ENV_CONFIG.get("BASE_URL")
    )
    joke_chain = prompt | model | StrOutputParser()
    return joke_chain


def api():
    import uvicorn

    qa_chain = build_chain()
    add_routes(app, qa_chain, path="/qa")

    joke_chain = build_joke_chain()
    add_routes(app, joke_chain, path="/joke")

    uvicorn.run(app, host="localhost", port=8005)


def test():
    qa_chain = build_chain()
    logger.info(qa_chain.invoke("RAG的本质是什么？"))

    joke_chain = build_joke_chain()
    logger.info(joke_chain.invoke({"topic": "cat"}))


def main():
    # test()
    api()


if __name__ == "__main__":
    main()

