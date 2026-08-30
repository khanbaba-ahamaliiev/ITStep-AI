import dotenv
import os

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
)


# Завдання 1
# Напишіть модель для рекомендації книг з двох ланцюгів:
#  Перший ланцюг отримує назву книги та визначає її
# жанр
#  Другий отримує назву книги, жанр та повертає список
# схожих книг(того ж самого жанру та іншого)

# class BookGenre(BaseModel):
#     genre: str = Field(description="жанр книги")
#
# parser_genre = PydanticOutputParser(pydantic_object=BookGenre)
# instructions_genre = parser_genre.get_format_instructions()
#
# prompt_genre = PromptTemplate.from_template("""
#     Ти — літературний експерт.
#     Твоя задача визначити жанр книги за її назвою.
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Назва книги: {book_title}
# """, partial_variables={"format_instructions": instructions_genre})
#
# chain_genre = prompt_genre | llm | parser_genre
#
#
# class BookRecommendations(BaseModel):
#     same_genre: list[str] = Field(description="список книг того ж жанру")
#     other_genre: list[str] = Field(description="список книг іншого жанру")
#
# parser_books = PydanticOutputParser(pydantic_object=BookRecommendations)
# instructions_books = parser_books.get_format_instructions()
#
# prompt_books = PromptTemplate.from_template("""
#     Ти — книжковий консультант.
#     Твоя задача порекомендувати схожі книги за назвою та жанром.
#
#     ###ІНСТРУКЦІЇ###
#     1. Запропонуй 3 книги того ж жанру
#     2. Запропонуй 2 книги іншого жанру, які можуть сподобатись читачеві
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Назва книги: {book_title}
#     Жанр: {genre}
# """, partial_variables={"format_instructions": instructions_books})
#
# chain_books = prompt_books | llm | parser_books
#
# book_title = "Гаррі Поттер і філософський камінь"
#
# data = {"book_title": book_title}
#
# response_genre = chain_genre.invoke(data)
# print(f"\n=== Завдання 1: Рекомендація книг ===")
# print(f"Книга: {book_title}")
# print(f"Жанр: {response_genre.genre}")
#
# data = {
#     "book_title": book_title,
#     "genre": response_genre.genre
# }
#
# response_books = chain_books.invoke(data)
# print(f"Книги того ж жанру: {response_books.same_genre}")
# print(f"Книги іншого жанру: {response_books.other_genre}")


# Завдання 2
# Напишіть модель для генерації листа:
#  Перший ланцюг отримує короткий опис листа та
# генерує основний зміст
#  Другий ланцюг отримує основний зміст та стиль
# листа(формальний, неформальний, тощо) та генерує
# лист

# class LetterContent(BaseModel):
#     subject: str = Field(description="тема листа")
#     main_content: str = Field(description="основний зміст листа у вигляді тез")
#
# parser_content = PydanticOutputParser(pydantic_object=LetterContent)
# instructions_content = parser_content.get_format_instructions()
#
# prompt_content = PromptTemplate.from_template("""
#     Ти — асистент з написання листів.
#     Твоя задача визначити тему та сформувати основний зміст листа на основі короткого опису.
#
#     ###ІНСТРУКЦІЇ###
#     1. Визнач коротку тему листа
#     2. Сформулюй основні тези змісту (3-5 речень)
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Опис листа: {description}
# """, partial_variables={"format_instructions": instructions_content})
#
# chain_content = prompt_content | llm | parser_content
#
#
# class FinalLetter(BaseModel):
#     letter: str = Field(description="готовий лист у заданому стилі")
#
# parser_letter = PydanticOutputParser(pydantic_object=FinalLetter)
# instructions_letter = parser_letter.get_format_instructions()
#
# prompt_letter = PromptTemplate.from_template("""
#     Ти — професійний автор листів.
#     Твоя задача написати лист на основі змісту у заданому стилі.
#
#     ###ІНСТРУКЦІЇ###
#     1. Дотримуйся обраного стилю протягом усього листа
#     2. Лист повинен бути завершеним з привітанням та підписом
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Тема листа: {subject}
#     Основний зміст: {main_content}
#     Стиль листа: {style}
# """, partial_variables={"format_instructions": instructions_letter})
#
# chain_letter = prompt_letter | llm | parser_letter
#
# letter_description = "Прошу відпустку на 2 тижні у серпні для сімейної поїздки"
# letter_style = "формальний"
#
# data = {"description": letter_description}
#
# response_content = chain_content.invoke(data)
# print(f"\n=== Завдання 2: Генерація листа ===")
# print(f"Тема: {response_content.subject}")
# print(f"Основний зміст: {response_content.main_content}")
#
# data = {
#     "subject": response_content.subject,
#     "main_content": response_content.main_content,
#     "style": letter_style
# }
#
# response_letter = chain_letter.invoke(data)
# print(f"Готовий лист:\n{response_letter.letter}")


# Завдання 3
# Напишіть модель для генерації резюме:
#  Перший ланцюг отримує опис вакансії та повертає
# основні навички, які необхідні
#  Другий ланцюг отримує основні навички та опис
# кандидата і генерує резюме

class JobSkills(BaseModel):
    position: str = Field(description="назва посади")
    required_skills: list[str] = Field(description="список необхідних навичок для вакансії")

parser_skills = PydanticOutputParser(pydantic_object=JobSkills)
instructions_skills = parser_skills.get_format_instructions()

prompt_skills = PromptTemplate.from_template("""
    Ти — HR-спеціаліст.
    Твоя задача проаналізувати опис вакансії та визначити ключові навички кандидата.

    ###ІНСТРУКЦІЇ###
    1. Визнач точну назву посади
    2. Виділи 5-7 найважливіших навичок

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Опис вакансії: {job_description}
""", partial_variables={"format_instructions": instructions_skills})

chain_skills = prompt_skills | llm | parser_skills


class Resume(BaseModel):
    resume: str = Field(description="готове резюме кандидата")

parser_resume = PydanticOutputParser(pydantic_object=Resume)
instructions_resume = parser_resume.get_format_instructions()

prompt_resume = PromptTemplate.from_template("""
    Ти — кар'єрний консультант.
    Твоя задача скласти резюме кандидата на основі його опису та вимог вакансії.

    ###ІНСТРУКЦІЇ###
    1. Підкресли навички, які відповідають вакансії
    2. Структуруй резюме: особисті дані, досвід, навички, освіта
    3. Резюме повинно бути лаконічним та переконливим

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Посада: {position}
    Необхідні навички: {required_skills}
    Опис кандидата: {candidate_description}
""", partial_variables={"format_instructions": instructions_resume})

chain_resume = prompt_resume | llm | parser_resume

job_description = """
    Шукаємо Python-розробника для роботи над AI-проєктами.
    Вимоги: знання Python, досвід з LangChain та OpenAI API, розуміння ML,
    вміння працювати з REST API, знання SQL баз даних.
"""
candidate_description = """
    Маю 2 роки досвіду в Python. Працював з FastAPI та Django.
    Вивчаю LangChain і вже реалізував кілька чат-ботів.
    Знаю основи машинного навчання, PostgreSQL та Git.
"""

data = {"job_description": job_description}

response_skills = chain_skills.invoke(data)
print(f"\n=== Завдання 3: Генерація резюме ===")
print(f"Посада: {response_skills.position}")
print(f"Необхідні навички: {response_skills.required_skills}")

data = {
    "position": response_skills.position,
    "required_skills": response_skills.required_skills,
    "candidate_description": candidate_description
}

response_resume = chain_resume.invoke(data)
print(f"Резюме:\n{response_resume.resume}")
