"""
这种写法不好，最好用函数或者 类把每一步做的东西区分开
"""

from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores import FAISS, Chroma
from langchain.retrievers import BM25Retriever
from langchain.document_loaders import TextLoader

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

from dotenv import dotenv_values
import pathlib

ENV_CONFIG = dotenv_values("../.env")
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


# vectorstore = FAISS.load_local(
#     pathlib.Path("../data/db_index"),
#     OpenAIEmbeddings(
#         api_key=ENV_CONFIG.get("API_KEY"), base_url=ENV_CONFIG.get("BASE_URL")
#     ),
# )
vectorstore = Chroma(
    persist_directory="../chroma_db",
    embedding_function=OpenAIEmbeddings(
        api_key=ENV_CONFIG.get("API_KEY"), base_url=ENV_CONFIG.get("BASE_URL")
    ),
)
retriever = vectorstore.as_retriever()

# text_loader = TextLoader(
#     "/Users/bytedance/work/person/langchain-local-qa/data/text.txt"
# )
# text_docs = text_loader.load()

# retriever = BM25Retriever.from_documents(documents=text_docs, k=1)

_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(
        temperature=0,
        api_key=ENV_CONFIG.get("API_KEY"),
        base_url=ENV_CONFIG.get("BASE_URL"),
    )
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs
    | _context
    | ANSWER_PROMPT
    | ChatOpenAI(api_key=ENV_CONFIG.get("API_KEY"), base_url=ENV_CONFIG.get("BASE_URL"))
    | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
