import numpy as np
from collections import Counter

import torch
from torch.nn import CrossEntropyLoss

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
import evaluate

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings

from gistai.core.constants import LANGUAGES
from gistai.core.utils import load_and_inspect_dataset
from misc.test_language import predict_language, predict_languages

# -------------------------
# 1. Define Constants and Paths
# -------------------------

# Path to your dataset (ensure it's a JSON file with 'text' and 'label' fields)
DATA_PATH = "datasets/language_detection.json"

# Directory to save the trained model
MODEL_OUTPUT_DIR = "./models/language_detection"

# Number of folds for cross-validation (optional)
K_FOLDS = 5

# Random seed for reproducibility
SEED = 42

MOODEL_NAME = (
    "bert-base-multilingual-cased"  # You can switch to a different model if needed
)


# -------------------------
# 3. Define Label Mappings
# -------------------------

label_mapping = {count: intent for count, intent in enumerate(LANGUAGES)}
id2label = {v: k for k, v in label_mapping.items()}

# -------------------------
# 4. Data Preprocessing Classes and Functions
# -------------------------


class LanguageDetectionProcessor:
    def __init__(self, tokenizer, label_mapping, max_length=16):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.max_length = max_length

    def __call__(self, examples):
        normalized_text = [normalize_text(text) for text in examples["text"]]
        tokenized = self.tokenizer(
            normalized_text,
            truncation=True,
            padding=False,  # Padding handled by DataCollator
            max_length=self.max_length,
        )
        tokenized["label"] = [self.label_mapping[label] for label in examples["label"]]
        return tokenized


def normalize_text(text):
    """
    Normalizes text by stripping leading/trailing whitespaces and ensuring consistent encoding.

    Args:
        text (str): The input text string.

    Returns:
        str: The normalized text string.
    """
    import unicodedata

    return unicodedata.normalize("NFC", text.strip())


# -------------------------
# 5. Compute Metrics Function
# -------------------------


def compute_metrics_fn(eval_pred):
    """
    Computes accuracy and F1 score for the evaluation.

    Args:
        eval_pred (tuple): A tuple containing logits and labels.

    Returns:
        dict: A dictionary with accuracy and F1 scores.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy_metric = evaluate.load("accuracy")
    acc = accuracy_metric.compute(predictions=predictions, references=labels)[
        "accuracy"
    ]

    # Compute F1 Score using sklearn
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


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

        Args:
            model (torch.nn.Module): The model being trained.
            inputs (Dict[str, torch.Tensor]): Batch of inputs with 'labels'.
            return_outputs (bool, optional): Whether to return the outputs. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: Loss or (loss, outputs).
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# -------------------------
# 7. Calculate and Set Class Weights
# -------------------------


def calculate_class_weights(train_dataset, label_mapping, id2label, epsilon=1e-6):
    """
    Calculates class weights inversely proportional to class frequencies.

    Args:
        train_dataset (Dataset): The training dataset.
        label_mapping (dict): Mapping from label names to IDs.
        id2label (dict): Mapping from IDs to label names.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Tensor containing class weights.
    """
    # Count the occurrences of each class
    label_counts = Counter(train_dataset["label"])

    # Calculate total samples
    total_samples = len(train_dataset)

    # Calculate weights: total_samples / (num_classes * count + epsilon)
    num_classes = len(label_mapping)
    class_weights = [
        total_samples / (num_classes * label_counts[label_id] + epsilon)
        for label_id in sorted(id2label.keys())
    ]

    # Convert to torch tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    return class_weights_tensor


# -------------------------
# 8. Freeze Base Model Layers
# -------------------------


def freeze_base_model(model):
    """
    Freezes all layers of the base model to prevent them from being updated during training.

    Args:
        model (transformers.PreTrainedModel): The model whose base layers are to be frozen.
    """
    for param in model.base_model.parameters():
        param.requires_grad = False


# -------------------------
# 8.a Unfreeze More Layers for Fine-Tuning
# -------------------------


def unfreeze_layers(model, num_layers=4):
    """
    Unfreezes the last `num_layers` transformer layers of the base model.

    Args:
        model (transformers.PreTrainedModel): The model to modify.
        num_layers (int): Number of layers to unfreeze from the end.
    """
    # Assuming the base model has a `transformer` or similar attribute
    # Adjust based on your specific model architecture
    base_model = model.base_model

    # Get all layers
    layers = list(base_model.children())

    # Identify transformer layers (e.g., transformer.encoder.layer in BERT)
    # Adjust based on your specific model
    if hasattr(base_model, "encoder"):
        transformer_layers = base_model.encoder.layer
    elif hasattr(base_model, "transformer"):
        transformer_layers = base_model.transformer.layer
    else:
        raise AttributeError(
            "Base model does not have `encoder` or `transformer` attribute."
        )

    # Unfreeze the last `num_layers` layers
    for layer in transformer_layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    print(f"Unfroze the last {num_layers} layers of the base model.")


# -------------------------
# 9. Print Trainable Parameters
# -------------------------


def print_trainable_parameters(model):
    """
    Prints the names of model parameters that require gradients.

    Args:
        model (transformers.PreTrainedModel): The model to inspect.
    """
    trainable_params = []
    non_trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            non_trainable_params.append(name)

    print("\n=== Trainable Parameters ===")
    if trainable_params:
        for name in trainable_params:
            print(f"  {name}")
    else:
        print("  None")

    print("\n=== Non-Trainable Parameters ===")
    if non_trainable_params:
        for name in non_trainable_params:
            print(f"  {name}")
    else:
        print("  None")


# -------------------------
# 10. Initialize Trainer Function
# -------------------------


def initialize_trainer(
    model,
    training_args,
    train_dataset,
    eval_dataset,
    data_collator,
    class_weights=None,
    processing_class=None,  # Add processing_class as a parameter if needed
):
    """
    Initializes the Hugging Face WeightedTrainer.

    Args:
        model (transformers.PreTrainedModel): The model to train.
        training_args (transformers.TrainingArguments): Training arguments.
        train_dataset (Dataset): The training dataset.
        eval_dataset (Dataset): The evaluation dataset.
        data_collator (transformers.DataCollator): Data collator for dynamic padding.
        class_weights (torch.Tensor, optional): Class weights for the loss function.
        processing_class (callable, optional): Preprocessing function.

    Returns:
        WeightedTrainer: The initialized WeightedTrainer object.
    """
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights,  # Pass the class weights here
        # processing_class=processing_class,  # Uncomment if you integrate processing_class within Trainer
    )
    return trainer


# -------------------------
# 11. K-Fold Cross-Validation (Optional)
# -------------------------


def perform_k_fold_cross_validation(
    tokenized_dataset, tokenizer, model_name, label_mapping, id2label, k_folds=5
):
    """
    Performs k-fold cross-validation on the dataset.

    Args:
        tokenized_dataset (Dataset): The tokenized dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        model_name (str): The pretrained model name.
        label_mapping (dict): Label to ID mapping.
        id2label (dict): ID to label mapping.
        k_folds (int): Number of folds.

    Returns:
        dict: Aggregated results across all folds.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_results = {}

    # Convert Dataset to list of indices
    full_indices = list(range(len(tokenized_dataset)))

    for fold, (train_indices, val_indices) in enumerate(kf.split(full_indices)):
        print(f"\n=== Fold {fold + 1} ===")

        # Split the dataset
        train_subset = tokenized_dataset.select(train_indices)
        val_subset = tokenized_dataset.select(val_indices)

        # Initialize the model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label_mapping),
            id2label=id2label,
            label2id=label_mapping,
        )

        # Freeze base model layers
        freeze_base_model(model)

        # Enable gradient checkpointing (optional)
        model.gradient_checkpointing_enable()

        # Calculate class weights for this fold
        class_weights = calculate_class_weights(train_subset, label_mapping, id2label)
        print(f"Class Weights for Fold {fold + 1}: {class_weights}")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=f"./models/language_detection_fold_{fold + 1}",
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

        # Define data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Initialize Trainer with class weights
        trainer = initialize_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=data_collator,
            class_weights=class_weights,  # Pass the class weights here
        )

        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate(eval_subset=val_subset)
        fold_results[f"Fold {fold + 1}"] = eval_results
        print(f"Fold {fold + 1} Evaluation Results: {eval_results}")

    # Aggregate results
    avg_accuracy = np.mean(
        [result["eval_accuracy"] for result in fold_results.values()]
    )
    avg_f1 = np.mean([result["eval_f1"] for result in fold_results.values()])

    print("\n=== Cross-Validation Results ===")
    for fold, results in fold_results.items():
        print(
            f"{fold}: Accuracy = {results['eval_accuracy']:.4f}, F1 Score = {results['eval_f1']:.4f}"
        )
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

    return fold_results


# -------------------------
# 12. Prediction Functions
# -------------------------


def load_trained_model(model_path):
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
        top_k=None,  # Use top_k=None to get all scores
    )
    return classifier


# -------------------------
# 13. Execute Main Function
# -------------------------


def main():
    # Suppress the FutureWarning temporarily
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

    # Check device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Clear MPS cache before training
    if device.type == "mps":
        torch.mps.empty_cache()

    # 1. Load and inspect dataset
    dataset = load_and_inspect_dataset("language_detection", "label")

    # 2. Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MOODEL_NAME)

    # 3. Initialize processing class
    processor = LanguageDetectionProcessor(
        tokenizer=tokenizer, label_mapping=label_mapping, max_length=16
    )

    # 4. Tokenize the dataset using the processing class
    print("\nTokenizing the dataset...")
    tokenized_dataset = dataset.map(
        processor,
        batched=True,
        remove_columns=["text"],  # Remove raw text
    )

    # 5. Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        MOODEL_NAME,
        num_labels=len(label_mapping),
        id2label=id2label,
        label2id=label_mapping,
    )

    # 6. Freeze base model layers
    freeze_base_model(model)

    # 6.a Unfreeze last 4 layers
    unfreeze_layers(model, num_layers=4)

    # 7. Disable gradient checkpointing (optional)
    # model.gradient_checkpointing_enable()  # Comment out or remove this line

    # 8. Move model to device (optional, Trainer handles it)
    model.to(device)

    # 9. Define training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        learning_rate=3e-5,  # Adjusted learning rate
        per_device_train_batch_size=16,  # Adjusted batch size
        per_device_eval_batch_size=16,  # Adjusted batch size
        gradient_accumulation_steps=8,  # Adjusted accumulation steps
        num_train_epochs=15,  # Increased epochs
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
        # no_cuda=True,  # Uncomment if you decide to switch to CPU
    )

    # 10. Define data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 11. Split the dataset into training and validation with stratification
    # Convert to pandas DataFrame for stratified splitting
    df = tokenized_dataset.to_pandas()

    # Perform stratified split using sklearn
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"], shuffle=True
    )

    # Convert 'label' columns to int
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    # Convert back to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    # Verify that no class has zero samples in training set
    train_label_counts = Counter(train_dataset["label"])
    print("\nTraining Label Distribution After Stratification:")
    for label, count in train_label_counts.items():
        print(f"  {id2label[label]}: {count}")

    zero_count_labels = [
        label for label, count in train_label_counts.items() if count == 0
    ]
    if zero_count_labels:
        print("\nError: The following labels have zero samples in the training set:")
        for label in zero_count_labels:
            print(f"  {id2label[label]}")
        print("Please ensure that the dataset has sufficient samples for each class.")
        exit(1)  # Exit the script

    # 12. Calculate class weights based on the training dataset
    class_weights = calculate_class_weights(train_dataset, label_mapping, id2label)
    class_weights = class_weights.to(
        device
    )  # Move class_weights to the same device as the model
    print(f"\nClass Weights: {class_weights}")

    # 13. Print Trainable Parameters to Verify
    print_trainable_parameters(model)

    # 14. Initialize Trainer with class weights
    trainer = initialize_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        class_weights=class_weights,  # Pass the class weights here
    )

    # 15. Train the model
    print("\n=== Starting Training ===")
    trainer.train()

    # 16. Evaluate the model
    print("\n=== Evaluating on Validation Set ===")
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Validation Results: {eval_results}")

    # 17. Save the best model
    print("\n=== Saving the Model ===")
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model saved to {MODEL_OUTPUT_DIR}")

    # 18. Optionally, perform K-Fold Cross-Validation
    perform_cv = False  # Set to True to perform cross-validation
    if perform_cv:
        print("\n=== Starting K-Fold Cross-Validation ===")
        perform_k_fold_cross_validation(
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            model_name=MOODEL_NAME,
            label_mapping=label_mapping,
            id2label=id2label,
            k_folds=K_FOLDS,
        )


# -------------------------
# 15. Execute Main Function
# -------------------------

if __name__ == "__main__":
    main()

    # Optional: After training, you can load the model and make predictions
    classifier = load_trained_model(MODEL_OUTPUT_DIR)

    # Single Prediction
    sample_text = "Biko, i nwere ike iso m gaa obodo?"
    lang, conf = predict_language(sample_text, classifier)
    if lang is not None and conf is not None:
        print(
            f"\nText: '{sample_text}' => Predicted Language: {lang} (Confidence: {conf:.2f})"
        )
    else:
        print(f"\nText: '{sample_text}' => Prediction failed.")

    # Batch Predictions
    batch_texts = [
        "I'm feeling great today!",
        "Abeg, make you help me.",
        "Bawo ni ọjọ rẹ ṣe n lọ?",
        "Ina ta ruwa sosai a yau.",
        "Kedu ka ị si mee?",
    ]
    batch_predictions = predict_languages(batch_texts, classifier)
    for text, (lang, conf) in zip(batch_texts, batch_predictions):
        if lang is not None and conf is not None:
            print(
                f"Text: '{text}' => Predicted Language: {lang} (Confidence: {conf:.2f})"
            )
        else:
            print(f"Text: '{text}' => Prediction failed.")

    # Edge Cases
    edge_case_sentences = [
        "I dey go market now, how about you?",  # Code-mixed: English + Pidgin
        "Yes.",  # Ambiguous: Could be English or Pidgin
        "Biko, i nwere ike iso m gaa obodo?",  # Igbo + English
        "Mo fe see movie.",  # Yoruba + English
        "Ina kwana, how you dey?",  # Hausa + Pidgin
    ]
    edge_predictions = predict_languages(edge_case_sentences, classifier)
    for text, (lang, conf) in zip(edge_case_sentences, edge_predictions):
        if lang is not None and conf is not None:
            print(
                f"Text: '{text}' => Predicted Language: {lang} (Confidence: {conf:.2f})"
            )
        else:
            print(f"Text: '{text}' => Prediction failed.")
