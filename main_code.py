import os,sys
from os import PathLike
from huggingface_hub import AsyncInferenceClient,InferenceClient
import asyncio
import aiofiles as aiof


os.environ["HF_TOKEN"] = ""

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

models = [
    # "LLM-LAT/robust-llama3-8b-instruct:featherless-ai",
    # "Commencis/Commencis-LLM:featherless-ai",
    # "WiroAI/wiroai-turkish-llm-8b:featherless-ai",
    # "CohereLabs/c4ai-command-a-03-2025",
    # "HuggingFaceTB/SmolLM3-3B",
    "IlyaGusev/saiga_llama3_8b",
]

async def main():
    dir_name = "answers"
    make_dir(dir_name)

    dir_path = os.path.join(os.path.dirname(__file__),dir_name)

    items = get_ideas("ideas.txt")
    for item in items:
        print(item)

    # print(asyncio.run(ask_model(models[0],"What is the capital of Russia?")))
    question = prompt_creation(items[-1])
    print(question)
    # sys.exit()
    tasks = []
    async with asyncio.TaskGroup() as tg:
        for model in models:
            task = tg.create_task(ask_model(model,question))
            tasks.append(task)
    
    
    save_tasks = []
    async with asyncio.TaskGroup() as tg:
        for item in tasks:
            file_name = item.model.replace("/","_")
            file_path = os.path.join(dir_path,file_name)
            task = tg.create_task(save_answer(file_path,item.choices[0].message))
    
    

async def async_ask_model(model:str,text:str):
    model_name = model.split('/')[0]
    print(f"{f'Question sent to {model_name}':=^150}")
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
    )

    print(f"{f'Got answer from {model_name}':=^150}")
    return completion

def ask_model(model:str,text:str):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
    )

    return completion

async def save_answer(path:os.PathLike,answer:str):
    async with aiof.open(path,"w",encoding="utf-8") as file:
        await file.write(answer)
        await file.flush()
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

def get_ideas(path:PathLike):
    with open(path,"r",encoding="utf-8") as file:
        return file.readlines()

if __name__=="__main__":
    # asyncio.run(async_ask_model(models[0],"How are you?"))
    asyncio.run(main())