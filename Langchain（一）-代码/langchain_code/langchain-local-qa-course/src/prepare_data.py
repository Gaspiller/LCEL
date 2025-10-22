# from langchain.embeddings import OpenAIEmbeddings

from pathlib import Path
from loguru import logger
from dotenv import dotenv_values
import tyro


from langchain.document_loaders import (
    TextLoader,
    CSVLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
)

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS


env_config = dotenv_values("../.env")

"""
doc1 : [1, 2, 3, , ...., 10, 12]
doc2: [8, 9, 10, ... 20]
"""
text_splitter = CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=200,
)

BASE_PATH = Path("../data")


def load_csv():  # -> List[Document]
    # load csv 的办法
    csv_loader = CSVLoader(
        file_path=BASE_PATH / "info.csv",
        metadata_columns=[
            "user_id",
        ],
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": [
                "user_id",
                "name",
                "birthday",
                "interest",
                "personal_website",
            ],
        },
    )

    csv_docs = csv_loader.load()

    logger.info(len(csv_docs))
    logger.info(type(csv_docs))
    logger.info(csv_docs[1].page_content)
    return csv_docs


def load_html():
    # load html
    html_loader = BSHTMLLoader(file_path=BASE_PATH / "query.html")
    html_docs = html_loader.load()

    logger.info(len(html_docs))
    logger.info(html_docs[0].page_content)
    return html_docs


def load_md():
    # load md
    md_loader = UnstructuredMarkdownLoader(BASE_PATH / "info.md")
    md_docs = md_loader.load()

    logger.info(md_docs)
    return md_docs


def load_text():
    ## load text
    text_loader = TextLoader(BASE_PATH / "text.txt")
    text_docs = text_loader.load()

    logger.info(text_docs)
    return text_docs


def load_local_data():
    text_docs = load_text()
    # md_docs = load_md()
    # html_docs = load_html()
    # csv_docs = load_csv()

    # 上面一共有有几种 docs
    # all_docs = [text_docs, md_docs, html_docs, csv_docs]
    all_docs = [
        text_docs,
    ]
    # 可以看到 document 是一样的；
    for item in all_docs:
        logger.info(f"type is is: {type(item[0])}")
    # doc item 很多
    # all_docs_item = [*text_docs, *md_docs, *html_docs, *csv_docs]
    all_docs_item = [
        *text_docs,
    ]
    # 进行 chunk 化;
    small_docs = text_splitter.transform_documents(all_docs_item)
    logger.info(len(small_docs))
    logger.info(small_docs[0])
    return small_docs


def prepare_db():
    small_docs = load_local_data()
    # embedding model
    emb_model = OpenAIEmbeddings(
        api_key=env_config.get("API_KEY"),
        base_url=env_config.get("BASE_URL"),
    )
    db = FAISS.from_documents(small_docs, emb_model)
    db.save_local("../data/db_index")


def load_db() -> FAISS:
    # embedding model
    emb_model = OpenAIEmbeddings(
        api_key=env_config.get("API_KEY"),
        base_url=env_config.get("BASE_URL"),
    )
    db = FAISS.load_local("../data/db_index", emb_model)
    return db


def main(pre_db: bool = False):
    # 是否是提前准备 data base
    if pre_db:
        prepare_db()
    else:
        db = load_db()
        # 进行检索
        query = "你好"
        res = db.similarity_search(query, k=3)
        logger.info(res)
        # 进行检索
        query = "你好"
        res = db.similarity_search_with_score(query, k=3)
        logger.info(res)
        # 进行检索
    # tyro, loguru


if __name__ == "__main__":
    # main()
    tyro.cli(main)
    # python prepare_data.py --pre-db
