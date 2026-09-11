import json
import dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)
from uuid import uuid4


# Завдання 1
# Добавте в створену базу даних файл
# data/lesson_rag/huge_file.txt про умови користування гуглом
# Оскільки файл надто великий, то його треба добавляти
# частинами. Для цього:
#  прочитайте вміст файлу
#  розділіть його на окремі блоки(між блоками два
# порожніх рядка, дивись файл)
#  отримайте перший рядок кожного блоку – це його
# назва
#  створіть документи для кожного блоку. В метаданих:
# o назва файлу
# o назва блоку
#  створіть ID та добавте все в існуючу базу даних
#  добавте ID у json файл
#  перевірте агента

dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

serper_search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key
)

pc = Pinecone(api_key=pinecone_api_key)

index_name = "itstep"

# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=3072,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# documents = []
#
# with open(r"data\lesson_rag\huge_file.txt", "r", encoding="utf-8") as f:
#     file_content = f.read()
#
# contents = file_content.split("\n\n")
#
# for n, content in enumerate(contents):
#     block_name = content.strip().split("\n")[0]
#     doc = Document(page_content=content, metadata={"file_name": "huge_file.txt", "block_name": block_name })
#     documents.append(doc)
#
# uuids = [str(uuid4()) for _ in range(len(documents))]
# with open("ids.json", "w") as f:
#     json.dump(uuids, f, ensure_ascii=False, indent=4)
#
# vector_store.add_documents(
#     documents=documents,
#     ids=uuids
# )
#

@tool
def document_search(query: str):
    """
    Шукає інформацію у векторній базі даних.

    База даних містить умови користування Google (huge_file.txt),
    а також інші документи про штучний інтелект.

    :param query: str -- запит користувача для пошуку
    :return: -- список знайдених документів з релевантною інформацією
    """
    results = vector_store.similarity_search(query, k=3)
    return results

agent = create_agent(
    model=llm,
    tools=[document_search]
)

messages = [
    SystemMessage("""
    Ти -- ввічливий та корисний чат-бот.
    Твоя задача -- відповідати на запити користувача про умови користування Google
    та інші теми.

    ###ІНСТРУКЦІЯ###
    1. Спочатку спробуй знайти відповідь у базі даних через document_search
    2. Якщо в базі даних немає потрібної інформації напиши що немає даних
    4. Не вигадуй інформацію
    """)
]

while True:
    message = input("Ви: ")

    if message == "":
        break

    user_message = HumanMessage(message)
    messages.append(user_message)
    data = {
        "messages": messages
    }

    data = agent.invoke(data)

    messages = data["messages"]

    response = messages[-1]
    print(response.text)
