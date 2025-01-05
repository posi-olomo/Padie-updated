import numpy as np
from collections import Counter
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
)
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings

from gistai.core.utils import load_and_inspect_dataset

# -------------------------
# 1. Define Constants and Paths
# -------------------------

MODEL_OUTPUT_DIR = "./models/intent_recognition"
MODEL_NAME = "bert-base-multilingual-cased"
SEED = 42

# Globals for intent recognition
tokenizer = None
id2label = None
label_mapping = None
accuracy_metric = evaluate.load("accuracy")  # Preload metrics for efficiency


def load_trained_model(model_path):
    """
    Loads the trained model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the saved model and tokenizer.

    Returns:
        Pipeline: Hugging Face pipeline for text classification.
    """
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    )


# -------------------------
# 3. Define Label Mappings
# -------------------------


def create_label_mapping(dataset):
    global id2label, label_mapping
    unique_intents = sorted(set(dataset["intent"]))
    label_mapping = {intent: idx for idx, intent in enumerate(unique_intents)}
    id2label = {v: k for k, v in label_mapping.items()}


# -------------------------
# 4. Preprocessing Class and Functions
# -------------------------


class IntentProcessor:
    def __init__(self, tokenizer, label_mapping, max_length=32):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.max_length = max_length

    def __call__(self, examples):
        try:
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
            )
            tokenized["label"] = [
                self.label_mapping[label] for label in examples["intent"]
            ]
            return tokenized
        except KeyError as e:
            print(f"Error during preprocessing: {e}")
            print(f"Examples: {examples}")
            raise e


# -------------------------
# 5. Compute Metrics
# -------------------------


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "f1": f1}


# -------------------------
# 6. Custom Trainer Class with Class Weights
# -------------------------


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss, using class weights if provided.
        """
        labels = inputs.get("labels").to(
            model.device
        )  # Ensure labels are on the same device as the model
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Move class weights to the same device as the model
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(model.device)

        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -------------------------
# 7. Calculate Class Weights
# -------------------------


def calculate_class_weights(dataset):
    label_counts = Counter(dataset["label"])
    total_samples = len(dataset)
    num_classes = len(label_mapping)
    class_weights = [
        total_samples / (num_classes * label_counts[label_id])
        for label_id in sorted(label_mapping.values())
    ]
    return torch.tensor(class_weights, dtype=torch.float)


# -------------------------
# 8. Main Training Function
# -------------------------


def main():
    global tokenizer, label_mapping, id2label

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Load and inspect dataset
    dataset = load_and_inspect_dataset("intent_recognition", "intent")
    create_label_mapping(dataset)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processor = IntentProcessor(tokenizer, label_mapping)

    # Tokenize dataset and convert to DataFrame
    tokenized_dataset = dataset.map(
        processor, batched=True, remove_columns=["text", "intent"]
    )
    df = tokenized_dataset.to_pandas()

    # Perform stratified train-test split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    # Convert back to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_mapping),
        id2label=id2label,
        label2id=label_mapping,
    )

    # Calculate class weights
    class_weights = calculate_class_weights(train_df)
    class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=2e-5,  # Adjusted learning rate
        per_device_train_batch_size=4,  # Adjusted batch size
        per_device_eval_batch_size=4,  # Adjusted batch size
        gradient_accumulation_steps=8,  # Adjusted accumulation steps
        num_train_epochs=30,  # Increased epochs
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        logging_steps=10,
        logging_dir="./logs",
        report_to="none",  # Disable integrations like WandB
        seed=SEED,
        fp16=False,  # Ensure mixed precision is disabled
    )

    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_fn,
        class_weights=class_weights,
    )

    # Train the model
    print("\n=== Starting Training ===")
    trainer.train()

    # Evaluate the model
    print("\n=== Evaluating on Validation Set ===")
    eval_results = trainer.evaluate()
    print(f"Validation Results: {eval_results}")

    # Save the best model
    print("\n=== Saving the Model ===")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model saved to {MODEL_OUTPUT_DIR}")


# -------------------------
# 9. Run Main and Test
# -------------------------

if __name__ == "__main__":
    main()

    # Test predictions
    classifier = load_trained_model(MODEL_OUTPUT_DIR)
    test_samples = [
        {"text": "Hello, how are you?", "expected_intent": "greeting"},
        {"text": "Can you help me?", "expected_intent": "help_request"},
        {"text": "Thank you!", "expected_intent": "farewell"},
    ]

    print("\n=== Testing Predictions ===")
    for sample in test_samples:
        prediction = classifier(sample["text"])[0]
        predicted_intent = prediction["label"]
        confidence = prediction["score"]
        print(
            f"Text: '{sample['text']}'\n"
            f"Expected Intent: {sample['expected_intent']}\n"
            f"Predicted Intent: {predicted_intent}, Confidence: {confidence:.2f}\n"
        )
