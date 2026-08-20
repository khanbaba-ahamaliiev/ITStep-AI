from langchain_google_genai import GoogleGenerativeAI
import os
import dotenv


# Завдання 1
# Прочитайте файл data\lesson9\return_policy.txt Та
# напишіть простий чат бот для відповідей на питання
# користувачів стосовно повернення товару. Діалог завершується
# коли користувач вводить порожній рядок.
# Передавайте усю історію спілкування у форматі:
# Instruction: ….
# Human: massage1
# AI: message2
# Human: massage3
# AI: message4
# Human: massage5
# AI:
dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    api_key=api_key,
)

with open("data/lesson9/return_policy.txt", "r", encoding="utf-8") as f:
    return_policy = f.read()

instruction = """
Ти — чат-бот для консультацій щодо повернення товару.
Відповідай коротко, чітко й лише на основі наданої політики повернення.
Не вигадуй правила, яких немає в тексті.
"""
history = "Instruction: " + instruction
while True:
    question = input("enter your question: ")
    history += f"\nHuman: {question}"

    if question == '' or question == ' ':
        break

    response = llm.invoke(f"""
        Слідуй інструкції: {instruction}, орієнтуйся на історію розмови: {history}. 
        Тобі потрібно відповісти на питання: {question}.
        """)
    print(response)

    history += f"\nAI: {response}"



