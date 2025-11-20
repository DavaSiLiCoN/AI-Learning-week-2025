import os
from os import PathLike
from huggingface_hub import InferenceClient
import asyncio

os.environ["HF_TOKEN"] = ""

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

models = [
    "LLM-LAT/robust-llama3-8b-instruct:featherless-ai",
    "Commencis/Commencis-LLM:featherless-ai",
    "WiroAI/wiroai-turkish-llm-8b:featherless-ai",
]

def main():
    for item in get_ideas("ideas.txt"):
        print(item)

    # print(asyncio.run(ask_model(models[0],"What is the capital of Russia?")))
    print(ask_model(models[1],"What is the capital of Russia?"))

async def async_ask_model(model:str,text:str):
    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": text
            }
        ],
    )

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

def prompt_cration(idea:str):
    return """
Ты Участник совета директоров в компании, кторая занимается венчурными инвестициями. К тебе пришел
"""

def get_ideas(path:PathLike):
    with open(path,"r",encoding="utf-8") as file:
        return file.readlines()

if __name__=="__main__":
    main()