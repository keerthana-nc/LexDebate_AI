from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# getting Flan T5 XL model which has 2048 tokens as input limit ~ 1 token = 4 charac's  => 8192 charc's
model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda") # seq to seq ( input : sentence / sequence to output: sentence / sequence )

generated_dataset = []

for name, date, text in tqdm(zip(names, dates, all_cases_texts), total = len(all_cases_texts)):
    instruction = "Generate defense and prosecutor arguments for this case"
    # To find the target
    input_text = f"{instruction} {text[0:4096]}" # extracting the first 1024 tokens or 4096 charc's . I'm thinking ill first load 1024 charc of content of case. "
    # convert the input_text that contains cases's contents to numbers  -> tokenize
    inputs = tokenizer(input_text,return_tensors='pt', truncation = True, max_length = 512).to(device='cuda')

    outputs = model.generate(**inputs, max_length = 512, num_beams = 4, early_stopping = True) # the aruguments that are in numbers format
    decoded = tokenizer(outputs[0], skip_special_tokens=True)

    print(f"\nCase: {name} | Date: {date}\nArguments:\n{decoded}\n")

    generated_dataset.append({
        "inputs": instruction,
        "targets": decoded
    })





