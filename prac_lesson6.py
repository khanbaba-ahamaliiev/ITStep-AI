import dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

dotenv.load_dotenv()


# Завдання 1
# Створіть векторну базу даних, де кожен документ – це
# вміст файлу з папки data/lesson_rag/files
#  добавте в метадані шлях до файлу
#  створіть для кожного документу ID
#  збережіть створені ID та назви відповідних файлів в
# окремий json файл
# Перевірте чи працює правильно пошук
api_key = os.getenv("GEMINI_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

serper_search = GoogleSerperAPIWrapper(
    serper_api_key=serper_key
)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

pc = Pinecone(api_key=pinecone_api_key)

index_name = "itstep"
#
# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=3072,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         ),
#     )
#
index = pc.Index(index_name)
#
vector_store = PineconeVectorStore(
    index=index,
    embedding=embedding
)
#
# with open(r"data\lesson_rag\files\future_of_ai.txt", "r", encoding="utf-8") as f:
#     file_content = f.read()
# doc1 = Document(page_content=file_content, metadata={"source": "future_of_ai.txt"})
#
# with open(r"data\lesson_rag\files\intro.txt", "r", encoding="utf-8") as f:
#     file_content = f.read()
# doc2 = Document(page_content=file_content, metadata={"source": "intro.txt"})
#
# with open(r"data\lesson_rag\files\machine_learning.txt", "r", encoding="utf-8") as f:
#     file_content = f.read()
# doc3 = Document(page_content=file_content, metadata={"source": "machine_learning.txt"})
#
# with open(r"data\lesson_rag\files\neural_networks.txt", "r", encoding="utf-8") as f:
#     file_content = f.read()
# doc4 = Document(page_content=file_content, metadata={"source": "neural_networks.txt"})
#
# documents = [doc1, doc2, doc3, doc4]
# uuids = [str(uuid4()) for _ in range(len(documents))]
#
# vector_store.add_documents(
#     documents=documents,
#     ids=uuids
# )


# Завдання 2
# На основі створеної бази даних створіть агента та
# реалізуйте його у вигляді чат бота

@tool
def document_search(query: str):
    """
    Пошук документів у векторній базі даних

    База даних містить інформацію про штучний інтелект
    :param query: str -- запис користувача
    :return: -- знайдені документи
    """

    result = vector_store.similarity_search(query, k=1)
    return result

agent = create_agent(
    model=llm,
    tools=[document_search]
)

messages = [
    SystemMessage("""
    Ти -- вічливий бот.
    
    ###ІНСТРУКЦІЯ###
    1. Якщо користувач питає про штучний інтелект використовуй document_search
    2. Якщо не має інформації не вигадуй
    
    """)
]

while True:
    message = input("You: ")

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

