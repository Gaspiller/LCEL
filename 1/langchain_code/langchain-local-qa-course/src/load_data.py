# from langchain.embeddings import OpenAIEmbeddings

from pathlib import Path
from loguru import logger

from langchain.document_loaders import (
    TextLoader,
    CSVLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
)


from langchain.text_splitter import CharacterTextSplitter


text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=200,
)

BASE_PATH = Path("../data")


# load csv 的办法
csv_loader = CSVLoader(
    file_path=BASE_PATH / "info.csv",
    metadata_columns=[
        "user_id",
    ],
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["user_id", "name", "birthday", "interest", "personal_website"],
    },
)

csv_docs = csv_loader.load()

logger.info(len(csv_docs))
logger.info(type(csv_docs))
logger.info(csv_docs[1].page_content)


# load html
html_loader = BSHTMLLoader(file_path=BASE_PATH / "query.html")
html_docs = html_loader.load()


logger.info(len(html_docs))
logger.info(html_docs[0].page_content)


# load md
md_loader = UnstructuredMarkdownLoader(BASE_PATH / "info.md")
md_docs = md_loader.load()

logger.info(md_docs)


## load text
text_loader = TextLoader(BASE_PATH / "text.txt")
text_docs = text_loader.load()

logger.info(text_docs)


# 上面一共有有几种 docs
all_docs = [text_docs, md_docs, html_docs, csv_docs]
# 可以看到 document 是一样的；
for item in all_docs:
    logger.info(f"type is is: {type(item[0])}")


all_docs_item = [*text_docs, *md_docs, *html_docs, *csv_docs]

small_docs = text_splitter.transform_documents(all_docs_item)


logger.info(len(small_docs))
logger.info(small_docs[0])
