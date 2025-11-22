from langchain_gigachat.chat_models import GigaChat
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import json
import os


giga_model_names = [
    "GigaChat",
    "GigaChat-Pro",
    "GigaChat-Max",
]

def main():
    # Credentials creation
    with open("giga_credentials.json") as file:
        credentials = json.load(file)
    auth_key = credentials["auth_key"]

    # Get Ideas from a file
    ideas = get_ideas(os.path.join(os.path.dirname(__file__),"ideas.txt"))

    # LLM initialization
    giga_models = model_init(giga_model_names,auth_key)

    # Main loop
    for i,idea in enumerate(ideas,1):
        prompt = prompt_creation(idea)
        for model,model_name in zip(giga_models,giga_model_names):
            result = model.invoke(prompt)
            save_answer(os.path.join(os.path.dirname(__file__),"answers",f"{i}_{model_name}"),result.content)

def save_answer(path:os.PathLike,answer:str):
    with open(f"{path}.md","w",encoding="utf-8") as file:
        file.write(answer)
    print(f"File{path} succesfully saved")

def make_dir(dir_name:str = "answers"):
    if dir_name not in os.listdir():
        os.mkdir(os.path.join(os.path.dirname(__file__),dir_name))

def prompt_creation(idea:str):
    return f"""
Проанализируй идею ниже. Тебе необходимо предоставить полный отчет об идее, который должен содержать следующие пункты:
1) Развернутое техническое описание
2) Список необходимых технологий/библиотек
3) Основные этапы реализации (3-5 пунктов)
4) Оценку сложности (легко/средне/сложно)

При подготовке отчета используй технический язык.

Формулировка идеи:
{idea}
"""

def get_ideas(path:os.PathLike):
    with open(path,"r",encoding="utf-8") as file:
        return file.readlines()
    
def model_init(models:list[str],auth_key:str) -> list[GigaChat]:
    result = []
    for model in models:
        result.append(
            GigaChat(credentials=auth_key, verify_ssl_certs=False, model=model)
        )
    
    return result


if __name__ == "__main__":
    main()
