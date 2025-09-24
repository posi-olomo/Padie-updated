import numpy as np
import pandas as pd
from datasets import load_dataset 
from collections import Counter
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score
import warnings

from padie.core.constants import LANGUAGES
# from padie.core.utils import load_and_inspect_dataset

# -----------------------------------------------------------------------------
# Constants & Configurations
# -----------------------------------------------------------------------------
MODEL_OUTPUT_DIR = "./models/full/language_detection"
MODEL_NAME = "Davlan/afro-xlmr-base"
SEED = 42


# Label Mappings
id2label = {i: lang for i, lang in enumerate(LANGUAGES)}
label2id = {lang: i for i, lang in enumerate(LANGUAGES)}


# -----------------------------------------------------------------------------
# Preprocessing Class
# -----------------------------------------------------------------------------
class LanguageDetectionProcessor:
    """
    Converts text examples into token IDs and maps string labels to numeric IDs.
    """

    def __init__(self, tokenizer, label2id, max_length=64):
        """
        Args:
            tokenizer: A Hugging Face tokenizer.
            label2id: Dict mapping string labels to label IDs (e.g., {"english": 0, ...}).
            max_length: Max sequence length for truncation.
        """
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, examples):
        texts = examples["text"]
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Let DataCollator handle padding dynamically
        )
        # Map string labels to numeric
        tokenized["labels"] = [
            self.label2id[label_str] for label_str in examples["label"]
        ]
        return tokenized


# -----------------------------------------------------------------------------
# Compute Metrics
# -----------------------------------------------------------------------------
def compute_metrics_fn(eval_pred):
    """
    Calculate Weighted F1 and Accuracy.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# -----------------------------------------------------------------------------
# Weighted Trainer
# -----------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies class weights in the cross-entropy loss.
    """

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def calculate_class_weights(train_dataset):
    """
    Calculates inverse frequency weights for each class.
    """
    labels = train_dataset["labels"]
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    class_weights = [
        total_samples / (num_classes * label_counts.get(label_id, 1))
        for label_id in range(num_classes)
    ]
    return torch.tensor(class_weights, dtype=torch.float)


def create_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    class_weights,
):
    return WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,
    )


# -----------------------------------------------------------------------------
# Load model function
# -----------------------------------------------------------------------------
def load_trained_model(model_path=MODEL_OUTPUT_DIR):
    """
    Loads the trained model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the saved model and tokenizer.

    Returns:
        Pipeline: Hugging Face pipeline for text classification.
    """
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.backends.mps.is_available() else -1,  # Use MPS if available
    )
    return classifier


# -----------------------------------------------------------------------------
# Main Training Script
# -----------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Load datasets
    datasets = load_dataset(
    "json",
    data_files={
        "train": "datasets/language_detection/train_dataset.jsonl",
        "eval": "datasets/language_detection/eval_dataset.jsonl",
    }
)
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label ),
        id2label=id2label,
        label2id=label2id,
    )

<<<<<<< HEAD:train_language_detection_updated.py
    # Freeze all transformer (base model) layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Only classifier head isgit trainable now
    for param in model.classifier.parameters():
        param.requires_grad = True

    print("Frozen base model parameters, training only classifier head.")

=======
>>>>>>> 1a471b8cfd5f67d0f91e11b6bbaf4a9e24ff175a:train_language_detection_old.py
    # 3. Processing / Tokenization
    processor = LanguageDetectionProcessor(tokenizer, label2id, max_length=64)
    train_dataset = train_dataset.map(
        processor, batched=True, remove_columns=["text", "label"]
    )
    eval_dataset = eval_dataset.map(
        processor, batched=True, remove_columns=["text", "label"]
    )

    # 4. Compute Class Weights
    class_weights = calculate_class_weights(train_dataset).to(device)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        seed=SEED,
        logging_strategy="steps",  # log at regular intervals
        logging_steps=10,
    )

    # 7. Data Collator
    data_collator = DataCollatorWithPadding(tokenizer)

    print("==== Starting training ====")
    # 8. Trainer Initialization & Training
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        class_weights=class_weights,
    )
    trainer.train()

    print("==== Train complete ====")

    print("==== Saving model ====")
    # 9. Save Model & Tokenizer
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print("==== Complete ====")


if __name__ == "__main__":
    main()
