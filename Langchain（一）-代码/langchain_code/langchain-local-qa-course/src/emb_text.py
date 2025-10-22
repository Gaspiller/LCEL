from dotenv import dotenv_values
from langchain.embeddings import OpenAIEmbeddings

env_config = dotenv_values("../.env")
emb = OpenAIEmbeddings(
    api_key=env_config.get("API_KEY"),
    base_url=env_config.get("BASE_URL"),
)
query_emb = emb.embed_query("hello world")
doc_embeds = emb.embed_documents(["hello ", "world"])
