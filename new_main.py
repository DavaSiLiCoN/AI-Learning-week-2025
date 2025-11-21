from transformers import pipeline


model = pipeline(task = "text-generation",model = "HuggingFaceTB/SmolLM3-3B")
print(model("Расширь следующаю идею: AI-система для торговли финансовыми активами в реальном времени. Система, позволяющая успешно спекулировать на фондовом рынке с целью финансовой выгоды."))