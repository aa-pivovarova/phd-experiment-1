import os
import evaluate
import matplotlib.pyplot as plt
import torch, torchaudio

def compute_metrics(pred):
    print("Computing metrics...")
    predictions = pred.predictions.argmax(-1)
    references = pred.label_ids

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    return {
        "accuracy": accuracy.compute(
            predictions=predictions,
            references=references)["accuracy"],
        "f1": f1.compute(
            predictions=predictions,
            references=references)["f1"],
        "precision": precision.compute(
            predictions=predictions,
            references=references)["precision"],
        "recall": recall.compute(
            predictions=predictions,
            references=references)["recall"],
    }

def build_graphs(fold, trainer):
    # Извлечение логов
    logs = trainer.state.log_history
    train_loss = [log["loss"] for log in logs if "loss" in log]
    eval_f1 = [log["eval_f1"] for log in logs if "eval_f1" in log]
    eval_accuracy = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]

    # Построение графиков
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(eval_f1, label="F1", color="orange")
    plt.title("Validation F1")
    plt.xlabel("Step")
    plt.ylabel("F1")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(eval_accuracy, label="Accuracy", color="green")
    plt.title("Validation Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(
        "./graphs/",
        f"training_curves-fold{fold}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training curves saved to {plot_path}")

def predict_sample(model, processor, path):
    speech, _ = torchaudio.load(path)
    inputs = processor(
        audio=speech,
        sampling_rate=16000,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return "Dysarthria" if predicted_label == 1 else "Healthy"