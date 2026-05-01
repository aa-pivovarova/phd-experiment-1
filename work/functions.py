import os
from functools import partial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from huggingface_hub import notebook_login
from datasets import Audio, Dataset, Value, load_from_disk
import IPython.display as ipd
from IPython.display import display, HTML
from transformers import XLMRobertaTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor, Wav2Vec2BertForSequenceClassification, TrainingArguments, Trainer
import torch
import torchaudio
import evaluate
import optuna
from dataclasses import dataclass
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, PeftModel
from DataCollatorForSpeechClassification import DataCollatorForSpeechClassification
import librosa
import soundfile
#############################################################################

def show_random_elements(dataset, num_examples=10):
    picks = []
    for i, example in enumerate(dataset):
        if i >= num_examples:
            break
        picks.append(example)

    df = pd.DataFrame(picks)

def prepare_dataset(batch, processor=None):
    if processor is None:
        raise ValueError("Processor must be provided")

    audio = batch["audio"]
    if (audio is None) or ("array" not in audio) or (audio["array"] is None):
        return None

    try:
        waveform = audio["array"]
        sampling_rate = audio["sampling_rate"]

        min_length_samples = int(0.5 * sampling_rate)  # 0.5 seconds minimum
        if len(waveform) < min_length_samples:
            #print(f"⚠️ Audio too short ({len(waveform) / sampling_rate:.2f}s): {batch.get('filename', 'unknown')}")
            return None

        batch["input_values"] = waveform
        batch["labels"] = batch["label"]
        return batch

    except Exception as e:
        # print(f"❌ Failed to process audio: {e}")
        return None  # Пропускаем битый файл

def load_torgo(root_dir, output_csv):
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print(f"Found CSV: {output_csv}")
    else:
        from pathlib import Path
        data = []
        root_path = Path(root_dir)
        for wav_path in root_path.rglob("*.wav"):
            if "wav_arrayMic" not in str(wav_path) and "wav_headMic" not in str(wav_path):
                continue
            # Путь: .../F/F01/Session1/wav_arrayMic/file.wav → parent.parent.parent.parent.name = F01
            subject_dir = wav_path.parent.parent.parent  # .../F01, FC01 и т.д.
            group_dir = subject_dir.parent  # .../F, FC, M, MC
            subject_id = subject_dir.name.lower() # F01 / FC01 и т.д.
            group_folder = group_dir.name.lower()  # F / FC / M / MC

            if group_folder in ["fc", "mc"]:
                label = 0
                group = "control"
            else:
                label = 1
                group = "dysarthria"
            gender = "f" if "f" in group_folder else "m"
            data.append({
                "audio": str(wav_path.resolve().as_posix()),
                "label": label,
                "group": group,
                "gender": gender,
                "subject_id": subject_id,
                "filename": wav_path.name
            })
        df = pd.DataFrame(data, columns=[
            "audio", "label", "group", "gender", "subject_id", "filename"
        ])
        df = df.sort_values(by=["group", "subject_id", "filename"]).reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        print(f"Created CSV: {output_csv}")

    print(f"Total audiofiles: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"Dysarthric/Healthy distribution:\n{df['group'].value_counts()}")
    return df

def load_easycall(root_dir, output_csv):
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print(f"Found CSV: {output_csv}")
    else:
        from pathlib import Path
        data = []
        root_path = Path(root_dir)
        for wav_path in root_path.rglob("*.wav"):
            # Путь: .../f01/Sessione_01/f01_01_text.wav
            subject_dir = wav_path.parent.parent  # .../f01
            subject_folder = subject_dir.name.lower()  # f01
            subject_id = subject_dir.name.lower()
            if ("fc" or "mc") in subject_folder:
                label = 0
                group = "control"
            else:
                label = 1
                group = "dysarthria"
            gender = "f" if "f" in subject_folder else "m"
            data.append({
                "audio": str(wav_path.resolve().as_posix()),
                "label": label,
                "group": group,
                "gender": gender,
                "subject_id": subject_id,
                "filename": wav_path.name
            })
        df = pd.DataFrame(data, columns=[
            "audio", "label", "group", "gender", "subject_id", "filename"
        ])
        df = df.sort_values(by=["group", "subject_id", "filename"]).reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        print(f"Created CSV: {output_csv}")
    print(f"Total audiofiles: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"Dysarthric/Healthy distribution:\n{df['group'].value_counts()}")
    return df

def transform_to_hfdataset(train_df, val_df, test_df):
    def create_filtered_dataset(df, name="dataset"):
         if len(df) == 0:
             print(f"{name} is empty.")
             return None
         # Define features explicitly
         from datasets import Features, Audio, Value
         features = Features({
             'audio': Value('string'),
             'label': Value('int64'),
             'group': Value('string'),
             'gender': Value('string'),
             'subject_id': Value('string'),
             'filename': Value('string')
         })
         df["audio"] = df["audio"].astype('string')

         ds = Dataset.from_pandas(df.reset_index(drop=True), features=features)
         print(f"📁 {name} created with {len(ds)} samples.")

         def path_exists(example):
             #print(example["audio"])
             return os.path.exists(example["audio"])

         ds_valid = ds.filter(path_exists)
         removed = len(ds) - len(ds_valid)
         if removed > 0:
             print(f"❌ Removed {removed} samples with missing audio files in {name}.")
         else:
             print(f"✅ All files in {name} exist.")
         print(f"✅ {name} final size: {len(ds_valid)} samples")
         return ds_valid

    train_dataset = create_filtered_dataset(train_df, "train_df")
    val_dataset = create_filtered_dataset(val_df, "val_df")
    test_dataset = create_filtered_dataset(test_df, "test_df")

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

    try:
        _ = train_dataset[0]["audio"]["array"]
        _ = val_dataset[0]["audio"]["array"]
        _ = test_dataset[0]["audio"]["array"]
    except Exception as e:
        print(f"Failed to load audio: {e}")

    return train_dataset, val_dataset, test_dataset

@dataclass
class TrainingConfig:
    train_size: float
    val_size: float
    test_size: float
    learning_rate: float
    batch_size: int
    epochs: int

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

def is_model_saved(model_dir):
    """Проверяет, сохранена ли полноценная LoRA-модель."""
    required_files = [
        "adapter_config.json",
        "adapter_model.bin",
        "preprocessor_config.json",  # от processor
    ]
    return all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)

def create_processor(model_dir="w2v2bert-dysarthria-model"):
    if os.path.exists(model_dir):
        try:
            print(f"Loading processor from {model_dir}...")
            return Wav2Vec2BertProcessor.from_pretrained(model_dir)
        except Exception as e:
            print(f"Could not load processor: {e}")

    print("Creating a new processor...")
    feature_extractor = SeamlessM4TFeatureExtractor(
        feature_size=80,
        num_mel_bins=80,
        sampling_rate=16000,
        padding_value=0.0
    )
    tokenizer = XLMRobertaTokenizer.from_pretrained("facebook/w2v-bert-2.0")
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    print("Processor created")
    return processor

def create_wav2vec2bert_for_classification(num_labels, model_dir="w2v2bert-dysarthria-model"):
    if is_model_saved(model_dir):
        print(f"Model found at {model_dir}. Loading...")
        try:
            model = Wav2Vec2BertForSequenceClassification.from_pretrained(
                "facebook/w2v-bert-2.0",
                num_labels=num_labels,
                ignore_mismatched_sizes=True  # на случай, если classifier не совпадает
            )
            model = PeftModel.from_pretrained(model, model_dir)
            print("LoRA model loaded successfully.")
            return model
        except Exception as e:
            print(f"Failed to load model from {model_dir}: {e}")
            print("Rebuilding model from scratch...")

    print("Creating Wav2Vec2BertForClassification model...")
    model = Wav2Vec2BertForSequenceClassification.from_pretrained(
        "facebook/w2v-bert-2.0",
        num_labels=num_labels,  #2: здоров / дизартрия
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1, #classifier_dropout=0.1,
        layerdrop=0.0
    )

    # Замораживаем все параметры модели
    model.freeze_base_model()
    # Определяем, куда применять LoRA
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,
        target_modules=[
            "linear_q",      # Q-проекции в self-attention
            "linear_v",      # V-проекции — ключевые для LoRA в attention
            #"linear_k",      # K-проекции (опционально)
            #"linear_out",    # Output projection
            "intermediate_dense"#,  # FFN intermediate layer
            #"output_dense"   # FFN output layer
        ],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],  # Не забываем про классификатор
    )
    # Оборачиваем модель
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("Model created")
    return model

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
    print(f"📉 Training curves saved to {plot_path}")

def create_trainer(
        fold, config, repo_name, model,
        data_collator, train_dataset,
        val_dataset, processor):
    print("Starting training...")
    training_args = TrainingArguments(
        output_dir=f"{repo_name}-fold-{fold}",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=config.epochs,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=100,
        eval_steps=50,
        logging_steps=50,
        learning_rate=config.learning_rate,
        warmup_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        eval_strategy="steps",
        save_strategy="steps",
        per_device_eval_batch_size=config.batch_size,
        disable_tqdm=False,
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(f"{repo_name}-fold-{fold}/best_model")
    build_graphs(fold, trainer)
    return metrics

##########################################################

def subject_kfold_cross_validation(
        config: TrainingConfig,
        df: pd.DataFrame = None,
        df_name : str = "",
        k: int = 5,
        repo_name: str = "",
        num_labels: int = 2,
):
    print(f"Starting {k}-fold CV on {df_name} dataset with {df['subject_id'].nunique()} subjects")
    # Уникальные субъекты + метки
    subjects_df = df[["subject_id", "label"]].drop_duplicates().reset_index(drop=True)
    subject_ids = subjects_df["subject_id"].to_numpy()  # или .values.astype(str) если нужно
    subject_labels = subjects_df["label"].to_numpy(dtype="int64", copy=True)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    print(f"Label dtype: {df["label"].dtype}")

    for fold, (train_val_idx, test_idx) in enumerate(skf.split(subject_ids, subject_labels)):
        print(f"\nLaunching Fold {fold+1} out of {k}")
        processor = create_processor()
        data_collator = DataCollatorForSpeechClassification(processor=processor)

        if all(os.path.exists(f"saved_datasets/{split}_dataset-fold-{fold}") for split in ["train", "val", "test"]):
            print("Loading datasets from disk...")
            train_dataset = load_from_disk(f"saved_datasets/train_dataset-fold-{fold}")
            val_dataset = load_from_disk(f"saved_datasets/val_dataset-fold-{fold}")
            test_dataset = load_from_disk(f"saved_datasets/test_dataset-fold-{fold}")
            print("Datasets loaded successfully.")

        else:
            print("Saved datasets not found. Creating from scratch...")
            train_val_subjects = subject_ids[train_val_idx]
            test_subjects = subject_ids[test_idx]
            train_val_labels = subject_labels[train_val_idx]

            total = config.train_size + config.val_size + config.test_size
            train_ratio = config.train_size / total
            val_ratio = config.val_size / total
            test_ratio = config.test_size / total

            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                train_size=train_ratio / (train_ratio + val_ratio),
                stratify=train_val_labels,
                random_state=42
            )

            train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
            val_df = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
            test_df = df[df["subject_id"].isin(test_subjects)].reset_index(drop=True)

            if len(val_df) == 0 or len(test_df) == 0:
                print("⚠️  Empty validation or test set. Skipping this fold.")
                continue

            print(f"Train: {len(train_df)} files | {len(train_subjects)} subjects")
            print(f"Val: {len(val_df)} files | {len(val_subjects)} subjects")
            print(f"Test: {len(test_df)} files | {len(test_subjects)} subjects")

            train_dataset, val_dataset, test_dataset = transform_to_hfdataset(
                train_df, val_df, test_df)

            prepare_fn = partial(prepare_dataset, processor=processor)
            def map_fn(batch):
                result = prepare_fn(batch)
                if result is None:
                    return {
                        "input_values": None,
                        "labels": None
                    }
                else: return result
            train_dataset = train_dataset.map(map_fn, remove_columns=train_dataset.column_names)
            val_dataset = val_dataset.map(map_fn, remove_columns=val_dataset.column_names)
            test_dataset = test_dataset.map(map_fn, remove_columns=test_dataset.column_names)

            train_dataset = train_dataset.filter(lambda x: x["input_values"] is not None and x["labels"] is not None)
            val_dataset = val_dataset.filter(lambda x: x["input_values"] is not None and x["labels"] is not None)
            test_dataset = test_dataset.filter(lambda x: x["input_values"] is not None and x["labels"] is not None)

            train_dataset.save_to_disk(f"saved_df/train_dataset-fold-{fold}")
            val_dataset.save_to_disk(f"saved_df/val_dataset-fold-{fold}")
            test_dataset.save_to_disk(f"saved_df/test_dataset-fold-{fold}")

        model = create_wav2vec2bert_for_classification(num_labels)
        metrics = create_trainer(
            fold=fold + 1,
            config=config,
            repo_name=repo_name,
            model=model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            processor=processor
        )
        fold_results.append(metrics)
        print(f"Fold {fold + 1} Validation Metrics: {metrics}")

    if not fold_results:
        return 0.0  # худший случай

    max_f1 = np.max([r["eval_f1"] for r in fold_results])
    max_acc = np.max([r["eval_accuracy"] for r in fold_results])
    print(f"\nAverage F1 over {k} folds: {max_f1:.4f}")
    print(f"Average Accuracy: {max_acc:.4f}")
    return fold_results

def launch_optuna_search(df, df_name, repo_name, k=5, n_trials=20, num_labels=2):
    print("🚀 Starting Optuna hyperparameter search...")

    def objective(trial):
        config = TrainingConfig(
            train_size=trial.suggest_float("train_size", 0.5, 0.8),
            val_size=trial.suggest_float("val_size", 0.1, 0.3),
            test_size=trial.suggest_float("test_size", 0.1, 0.3),
            learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
            epochs=trial.suggest_int("epochs", 5, 15),
        )
        return subject_kfold_cross_validation(
            config=config,
            df=df,
            df_name=df_name,
            k=k,
            num_labels=num_labels,
            repo_name=repo_name
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"F1: {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    return study

###########################################################

def cross_dataset_evaluation(train_ds, test_ds, best_config, repo_name, model_dir="w2v2bert-dysarthria-model"):
    if is_model_saved(model_dir):
        print(f"Full model found at {model_dir}. Skipping training.")
        processor = create_processor()
        model = None
        results = None
    else:
        print("Starting training")

        train_subjects, val_subjects = train_test_split(
            train_ds["subject_id"].unique(),
            test_size=0.2,
            stratify=train_ds.drop_duplicates("subject_id")["label"],
            random_state=42,
        )
        train_df = train_ds[train_ds["subject_id"].isin(train_subjects)]
        val_df = train_ds[train_ds["subject_id"].isin(val_subjects)]

        test_df = test_ds
        train_dataset, val_dataset, test_dataset = transform_to_hfdataset(train_df, val_df, test_df)
        processor = create_processor()
        data_collator = DataCollatorForSpeechClassification(processor=processor)

        train_dataset = train_dataset.map(
            partial(prepare_dataset, processor=processor),
            remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(
            partial(prepare_dataset, processor=processor),
            remove_columns=val_dataset.column_names)
        test_dataset = test_dataset.map(
            partial(prepare_dataset, processor=processor),
            remove_columns=test_dataset.column_names)

        model = create_wav2vec2bert_for_classification(num_labels=2)

        if not is_model_saved(model_dir):  # Только если ещё не обучали
            training_args = TrainingArguments(
                output_dir=f"{repo_name}-cross-eval",
                per_device_train_batch_size=best_config["batch_size"],
                gradient_accumulation_steps=2,
                evaluation_strategy="steps",
                eval_steps=50,
                num_train_epochs=best_config["epochs"],
                learning_rate=best_config["learning_rate"],
                fp16=True,
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                logging_steps=50,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=processor,
            )
            trainer.train()
            results = trainer.evaluate(test_dataset)
            print("Evaluation results:", results)

            trainer.save_model()
            print(f"LoRA model saved to {model_dir}")
            # Сохраняем processor, чтобы потом восстановить
            processor.save_pretrained(model_dir)
        else:
            results = None

    # Если модель уже была — загружаем её здесь
    if model is None:
        print(f"Loading saved model from {model_dir}...")
        base_model = Wav2Vec2BertForSequenceClassification.from_pretrained(
            "facebook/w2v-bert-2.0", num_labels=2, ignore_mismatched_sizes=True
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
    return results, model, processor

##===================================================================================

def predict_sample(model, processor, path):
    speech, _ = torchaudio.load(path)
    inputs = processor(
        raw_speech=speech.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    return "Dysarthria" if predicted_label == 1 else "Healthy"