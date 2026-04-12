import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
# Disable MPS as it's causing device mismatch errors in SetFit linear layers
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

import pandas as pd
import json
from setfit import SetFitModel, Trainer, TrainingArguments, SetFitHead
from sentence_transformers import SentenceTransformer
from datasets import Dataset

# Define Training Data
train_data = [
    # Inventory queries (label: 0)
    {"text": "What is the total stock?", "label": 0},
    {"text": "Show me the inventory summary", "label": 0},
    {"text": "How many products do we have in Store A?", "label": 0},
    {"text": "Which items are low on stock?", "label": 0},
    {"text": "Summarize the current inventory", "label": 0},
    {"text": "List all products in stock", "label": 0},
    {"text": "Check restocking needs at Store B", "label": 0},
    {"text": "What's the status of inventory for product X?", "label": 0},
    {"text": "Show me the total units sold per store", "label": 0},
    {"text": "Give me a breakdown of product-wise stock", "label": 0},

    # Predictive queries (label: 1)
    {"text": "Predict future sales for the next month", "label": 1},
    {"text": "Forecast inventory needs for next week", "label": 1},
    {"text": "What will be the stock levels after Christmas?", "label": 1},
    {"text": "Give me a sales prediction for Store C", "label": 1},
    {"text": "Analyze trends and predict stock exhaustion", "label": 1},
    {"text": "Run a predictive model on inventory data", "label": 1},
    {"text": "Forecast total units received next quarter", "label": 1},
    {"text": "When should I reorder items based on trends?", "label": 1},
    {"text": "Apply the predictive model to the current dataset", "label": 1},
    {"text": "What are the predicted trends for product Z?", "label": 1},

    # Irrelevant queries (label: 2)
    {"text": "What's the weather today?", "label": 2},
    {"text": "Tell me a joke.", "label": 2},
    {"text": "How do I bake a cake?", "label": 2},
    {"text": "What's the meaning of life?", "label": 2},
    {"text": "Recommend a movie.", "label": 2},
    {"text": "Who won the World Cup?", "label": 2},
    {"text": "What is the capital of France?", "label": 2},
    {"text": "Can you sing a song?", "label": 2},
    {"text": "Tell me about dinosaurs.", "label": 2},
    {"text": "How do I fix my computer?", "label": 2},
]

def train():
    print("Preparing dataset...")
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

    print("Loading base model: all-MiniLM-L6-v2")
    # Force use of CPU if needed, but defaults are usually fine
    sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    model = SetFitModel(
        model_body=sentence_transformer,
        model_head=SetFitHead(
            in_features=384,  # all-MiniLM-L6-v2 has 384 dimensions
            out_features=3    # inventory, predictive, irrelevant
        )
    )

    print("Starting training...")
    args = TrainingArguments(
        batch_size=8,
        num_epochs=1,
        use_amp=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()

    output_dir = os.path.join(os.path.dirname(__file__), "..", "config", "inventory_query_classifier")
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)

    label_map = {
        0: "inventory",
        1: "predictive",
        2: "irrelevant"
    }

    with open(os.path.join(output_dir, "label_mapping.json"), 'w') as f:
        json.dump(label_map, f, indent=2)
    print("Training complete and mapping saved.")

if __name__ == "__main__":
    train()
