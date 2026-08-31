import dotenv
import os

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


dotenv.load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)

# Завдання 1
# Напишіть модель для генерації персонального плану
# тренувань з двох ланцюгів:
#  Перший ланцюг отримує мету тренування(схуднення,
# набір м’язів, тощо) та повертає список вправ
#  Другий ланцюг отримує список вправ, рівень
# підготовки користувача(низький, середній,
# професіонал) та кількість часу на тиждень(в годинах)
# і повертає план тренувань

class ExerciseList(BaseModel):
    goal: str = Field(description="мета тренування")
    exercises: list[str] = Field(description="список рекомендованих вправ для досягнення мети")

parser_exercises = PydanticOutputParser(pydantic_object=ExerciseList)
instructions_exercises = parser_exercises.get_format_instructions()

prompt_exercises = PromptTemplate.from_template("""
    Ти — фітнес-тренер.
    Твоя задача підібрати список вправ відповідно до мети тренування.

    ###ІНСТРУКЦІЇ###
    1. Визнач мету тренування
    2. Запропонуй 6-8 вправ, які найкраще підходять для цієї мети

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Мета тренування: {training_goal}
""", partial_variables={"format_instructions": instructions_exercises})

chain_exercises = prompt_exercises | llm | parser_exercises


class TrainingPlan(BaseModel):
    plan: str = Field(description="детальний персональний план тренувань на тиждень")

parser_plan = PydanticOutputParser(pydantic_object=TrainingPlan)
instructions_plan = parser_plan.get_format_instructions()

prompt_plan = PromptTemplate.from_template("""
    Ти — персональний тренер.
    Твоя задача скласти індивідуальний план тренувань на тиждень на основі списку вправ.

    ###ІНСТРУКЦІЇ###
    1. Врахуй рівень підготовки користувача
    2. Розподіли вправи по днях тижня відповідно до доступного часу
    3. Для кожної вправи вкажи кількість підходів та повторень

    ###ФОРМАТ ВІДПОВІДІ###
    {format_instructions}

    ###ВХІДНІ ДАНІ###
    Мета: {goal}
    Список вправ: {exercises}
    Рівень підготовки: {fitness_level}
    Час на тиждень (годин): {hours_per_week}
""", partial_variables={"format_instructions": instructions_plan})

chain_plan = prompt_plan | llm | parser_plan

training_goal = "схуднення"
fitness_level = "середній"
hours_per_week = 4

data = {"training_goal": training_goal}

response_exercises = chain_exercises.invoke(data)
print(f"\n=== Завдання 1: Персональний план тренувань ===")
print(f"Мета: {response_exercises.goal}")
print(f"Вправи: {response_exercises.exercises}")

data = {
    "goal": response_exercises.goal,
    "exercises": response_exercises.exercises,
    "fitness_level": fitness_level,
    "hours_per_week": hours_per_week
}

response_plan = chain_plan.invoke(data)
print(f"План тренувань: {response_plan.plan}")

