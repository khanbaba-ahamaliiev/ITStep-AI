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



# перевод текста в вектор

# text1 = "суп полезный при простуде"
# vector1 = embeddings.embed_query(text1)

# print(vector1)
# print(type(vector1))
# print(len(vector1))

# text2 = "суп придумали в китае"
# vector2 = embeddings.embed_query(text2)
#
# print(vector2)
# print(type(vector2))
# print(len(vector2))

# векторная база данных
pc = Pinecone(api_key=pinecone_api_key)

index_name = "itstep"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

#
# # добавить тексты
# # обработка сперва
#
# # document 1
# doc1 = Document(
#     page_content=,
#     metadata=
# )
#
# # создание идентификаторов
#
#
#
#
# # добавить документ в базу данных
#
#
#
#
# # найти похожих документов
# user_query = ""
# result_docs = vector_store.similarity_search(user_query, k=2)

@tool
def search_doc(query: str):
    """
    поиск документов в векторной базе данных
    :param query: -- запрос пользователя
    :return: -- похожие документы
    """

    results = vector_store.similarity_search(
        query,  # текст для пошуку схожих документів
        k=2,  # кількість документів яку шукаємо
    )

    return results


agent = create_agent(
        model=llm,
        tools=[]
)

messages = [
    SystemMessage("""
    Ти -- ввічлиіий чат бот
    
        ###ІНСТРУКЦІЯ###    
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
