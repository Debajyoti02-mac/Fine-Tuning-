from transformers import AutoTokenizer , AutoModelForCausalLM 

# Auto Tokenizer is for the convert text to a token 
# Automodelforcasualllm is for loads a text generating model 

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)

data = [
    {
        "question":"what is python?",
        "answer":"python is high level programing language."
    },
    {
        "question": "What is RAG?",
        "answer": "RAG retrieves relevant information and gives it to a language model."
    },
    {
        "question": "What is FastAPI?",
        "answer": "FastAPI is a Python framework for building APIs."
    }
    
]

from datasets import load_dataset 
dataset  = load_dataset(
    "json",
    data_files="dataset.json"
)
print(dataset)

print(f'the first line for train : {dataset['train'][0]}')
    
# Format change for data

def format_data(example):
    return {
        "text":
            f"Question :{example['question']}\n"
            f'answer :{example['answer']}'
    }
dataset = dataset.map(format_data)
print(dataset['train'][0])

print(tokenizer)
print(type(tokenizer))
tokenizer.pad_token = tokenizer.eos_token

# Tokenizer 
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenize_dataset = dataset.map(tokenize)

print(tokenize_dataset["train"][0])

def add_label(example):
    example['labels'] = [token if mask == 1 else -100 for token , mask in zip (example['input_ids'] , example['attention_mask']) ]
    return example 


from transformers import TrainingArguments, Trainer
tokenize_dataset = tokenize_dataset.map(add_label)

print(tokenize_dataset["train"][0]["labels"])

training_args = TrainingArguments(
    output_dir="./fine_tune_model",
    num_train_epochs=10 , 
    per_device_train_batch_size=1 , 
    learning_rate=5e-5,
    logging_steps=1 , 
    save_strategy='no',
    report_to='none'
)

trainer = Trainer(model=model, args=training_args,train_dataset=tokenize_dataset['train'])

trainer.train()

model.eval()

prompt = "Question :what is python?\nanswer :"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

output = model.generate(
    **inputs,
    max_new_tokens=30
)

print(tokenizer.decode(output[0]))