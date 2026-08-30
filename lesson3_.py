import dotenv
import os

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI #GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

# ПАРСЕР


# Пользователь задает вопрос
# Нужно дать ответ и предложить интересные факты по этой
# же теме что и вопрос


# вариант 1 -- все в один промпт
prompt = PromptTemplate.from_template("""
    Ты -- учитель.
    Твоя задача давать короткие ответы на вопросы и предлагать интересные факты на похожую тему.

    ###ИНСТРУКЦИИ###
    1. Ответы до 3 предложений
    2. Предложи до 5 интересных фактов на туже тему
    3. Факты должны заинтересовать ученика узнавать больше

    ###ВХОДНЫЕ ДАННЫЕ###
    Вопрос: {question}

    Ответ:

""")

user_question = "Когда был полет на луну"

# data = {
#     "question": user_question
# }

# text = prompt.invoke(data)
#
# response = llm.invoke(text)
#
# # print(response)
# # из-за ChatGoogleGenerativeAI следует выводить ответ вот так
# text_response = response.content[0]["text"]
# print(text_response)


# объединяем в один шаг
# создаем цепочку
# chain = prompt | llm
#
# data = {
#     "question": user_question
# }
#
# response = chain.invoke(data)
# print(response)


# вариант 2 -- розбить на 2 шага
# дать ответ и определить тему вопроса
# сгенерировать интересные факты по теме

# структура ответа
class AnswerFact(BaseModel):
    answer: str = Field(description="ответ на вопрос")
    facts: list[str] = Field(description="список тем связанных с вопросом пользователя")

# создание парсера
parser = PydanticOutputParser(pydantic_object=AnswerFact)

# инструкция от парсера
instructions = parser.get_format_instructions()
# print(instructions)

# промпт
prompt1 = PromptTemplate.from_template("""
    Ты -- чатбот по обучению.
    Твоя задача дать ответ на вопрос пользователя и так же определить список тем связанных с вопросом
    
    ###ИНСТРУКЦИИ###
    1. Ответы до 3 предложений
    2. Список должен быть не больше 5 тем
    
    ###ФОРМАТ ОТВЕТА###
    {format_intrications}
    
    ###ВХОДНЫЕ ДАННЫЕ###
    Вопрос: {question}
""")


# # цепочка для 1 шага
chain1 = prompt1 | llm | parser
#
# # использование
# user_question = "Когда был полет на луну"
#
# data = {
#     "question": user_question,
#     "format_intrications": instructions
# }
#
# response = chain1.invoke(data)
# # print(response)
#
# answer = response.answer
# # print(answer)
#
# themes = response.facts
# print(type(themes))
# print(themes)

# шаг 2
# сгенерировать новые факты по теме

class Facts(BaseModel):
    facts: list[str] = Field(description="список интересных фактов на заданные темы")

parser2 = PydanticOutputParser(pydantic_object=Facts)

instructions2 = parser2.get_format_instructions()


prompt = PromptTemplate.from_template("""
    Ты — преподаватель.
    Твоя задача привести несколько интересных фактов по заданным темам, чтобы заинтересовать студента.
    
    ###ИНСТРУКЦИИ###
    1. Каждый факт должен быть одним предложением
    2. Не больше 3 фактов
    
    ###ФОРМАТ ОТВЕТА###
    {format_instructions}
    
    ###ВХОДНЫЕ ДАННЫЕ###
    Список тем: {themes}  
""", partial_variables={"format_instructions": instructions2})

chain2 = prompt | llm | parser2

user_question = "Когда был полет на луну"

data = {
    "question": user_question,
    "format_intrications": instructions
}

response1 = chain1.invoke(data)
print(f"answer: {response1.answer}")

data = {
    "themes": response1.facts
}

response2 = chain2.invoke(data)
print(f"facts: {response2.facts}")
