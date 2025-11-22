import torch
from transformers import pipeline
import os
from os import PathLike
import asyncio
import aiofiles as aiof


model_ids = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "google/gemma-3-1b-it",
    # "IlyaGusev/saiga_llama3_8b",
]

print("Cuda availability:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")


async def main():
    make_dir()
    models = create_models(model_ids)
    for model in models:
        print(models[model](prompt_creation("AI-система для торговли финансовыми активами в реальном времени. Система, позволяющая успешно спекулировать на фондовом рынке с целью финансовой выгоды.")))


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
    
async def save_answer(path:os.PathLike,answer:str):
    async with aiof.open(path,"w",encoding="utf-8") as file:
        await file.write(answer)
        await file.flush()
    print(f"File{path} succesfully saved")

def make_dir(dir_name:str = "answers"):
    if dir_name not in os.listdir():
        os.mkdir(os.path.join(os.path.dirname(__file__),dir_name))

def create_models(model_ids:list[str]):
    result = {}
    for model_id in model_ids:
        result[model_id] = pipeline(task = "question-answering",model = model_id,device = device)
    return result
    
if __name__=="__main__":
    asyncio.run(main())