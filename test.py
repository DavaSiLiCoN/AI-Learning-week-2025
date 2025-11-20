import os
from huggingface_hub import InferenceClient

os.environ["HF_TOKEN"] = ""

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

completion = client.chat.completions.create(
    model="google/gemma-2-2b-it:nebius",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)