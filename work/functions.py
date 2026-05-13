import os
from functools import partial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from datasets import load_from_disk
import IPython.display as ipd
from IPython.display import display, HTML
from transformers import Wav2Vec2BertForSequenceClassification, TrainingArguments, Trainer
import optuna
from peft import PeftModel
from DataCollatorForSpeechClassification import DataCollatorForSpeechClassification
import librosa
import soundfile
import pipeline_objects, load_datasets, auxiliary
#############################################################################

def subject_fold_training(
        config: pipeline_objects.TrainingConfig,
        df: pd.DataFrame = None,
        df_name : str = "",
        repo_name: str = "",
        num_labels: int = 2,
):
    print(f"Starting on {df_name} dataset with {df['subject_id'].nunique()} subjects")

    processor = pipeline_objects.create_processor()
    data_collator = DataCollatorForSpeechClassification(processor=processor)

    train_val_subjects = df[["subject_id", "label"]].drop_duplicates().reset_index(drop=True)
    test_subjects = train_val_subjects["subject_id"].to_numpy()  # или .values.astype(str) если нужно
    train_val_labels = train_val_subjects["label"].to_numpy(dtype="int64", copy=True)

    print(config.train_size)
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        train_size=config.train_size,
        stratify=train_val_labels,
        random_state=42
    )
    train_df = df[df["subject_id"].isin(train_subjects["subject_id"])].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects["subject_id"])].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjects)].reset_index(drop=True)

    if len(val_df) == 0 or len(test_df) == 0:
        print("⚠️ Empty validation or test set")

    print(f"Train: {len(train_df)} files | {len(train_subjects)} subjects")
    print(f"Val: {len(val_df)} files | {len(val_subjects)} subjects")
    print(f"Test: {len(test_df)} files | {len(test_subjects)} subjects")

    train_dataset, val_dataset, test_dataset = load_datasets.transform_to_hfdataset(
        train_df, val_df, test_df)

    prepare_fn = partial(load_datasets.prepare_dataset, processor=processor)
    def map_fn(batch):
        result = prepare_fn(batch)
        if result is None:
            return {
                "input_values": None,
                "labels": None
            }
        else: return result
    train_dataset = train_dataset.map(map_fn, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(map_fn, remove_columns=test_dataset.column_names)
    train_dataset = train_dataset.filter(lambda x: x["input_values"] is not None and x["labels"] is not None)
    test_dataset = test_dataset.filter(lambda x: x["input_values"] is not None and x["labels"] is not None)

    model = pipeline_objects.create_wav2vec2bert_for_classification(num_labels)
    metrics = pipeline_objects.create_trainer(
        fold=0,
        config=config,
        repo_name=repo_name,
        model=model,
        data_collator=data_collator,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        processor=processor
    )
    print(f"Metrics: {metrics}")
    return metrics

def subject_kfold_cross_validation(
        config: pipeline_objects.TrainingConfig,
        df: pd.DataFrame = None,
        df_name : str = "",
        k: int = 1,
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
        processor = pipeline_objects.create_processor()
        data_collator = DataCollatorForSpeechClassification(processor=processor)

        if all(os.path.exists(f"saved_df/{split}_dataset-fold-{fold}") for split in ["train", "val", "test"]):
            print("Loading datasets from disk...")
            train_dataset = load_from_disk(f"saved_df/train_dataset-fold-{fold}")
            val_dataset = load_from_disk(f"saved_df/val_dataset-fold-{fold}")
            test_dataset = load_from_disk(f"saved_df/test_dataset-fold-{fold}")
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
                print("⚠️ Empty validation or test set. Skipping this fold.")
                continue

            print(f"Train: {len(train_df)} files | {len(train_subjects)} subjects")
            print(f"Val: {len(val_df)} files | {len(val_subjects)} subjects")
            print(f"Test: {len(test_df)} files | {len(test_subjects)} subjects")

            train_dataset, val_dataset, test_dataset = load_datasets.transform_to_hfdataset(
                train_df, val_df, test_df)

            prepare_fn = partial(load_datasets.prepare_dataset, processor=processor)
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

        model = pipeline_objects.create_wav2vec2bert_for_classification(num_labels)
        # for entry in os.scandir(f"work/DATASETS/TORGO/M/M04/Session2/wav_headMic"):
        #     if entry.is_file():
        #         print(predict_sample(model, processor, entry.path))
        # print("!!!! NOW DOING CONTROL !!!!")
        # for entry in os.scandir(f"work/DATASETS/TORGO/MC/MC02/Session1/wav_headMic"):
        #     if entry.is_file():
        #         print(predict_sample(model, processor, entry.path))

        metrics = pipeline_objects.create_trainer(
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
    print(f"\nMax F1 over {k} folds: {max_f1:.4f}")
    print(f"Max Accuracy: {max_acc:.4f}")
    return fold_results

def launch_optuna_search(df, df_name, repo_name, k, n_trials, num_labels):
    print("🚀 Starting Optuna hyperparameter search...")

    def objective(trial):
        # config = pipeline_objects.TrainingConfig(
        #     train_size=trial.suggest_float("train_size", 0.5, 0.8),
        #     val_size=trial.suggest_float("val_size", 0.1, 0.3),
        #     test_size=trial.suggest_float("test_size", 0.1, 0.3),
        #     learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        #     batch_size=trial.suggest_categorical("batch_size", [2, 4, 6]),
        #     epochs=trial.suggest_int("epochs", 5, 15),
        # )
        config = pipeline_objects.TrainingConfig(
            train_size=0.6,
            val_size=0.1,
            test_size=0.3,
            learning_rate=1e-4,
            batch_size=2,
            epochs=1,
        )
        return subject_fold_training(
            config=config,
            df=df,
            df_name=df_name,
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
    if pipeline_objects.is_model_saved(model_dir):
        print(f"Full model found at {model_dir}. Skipping training.")
        processor = pipeline_objects.create_processor()
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
        train_dataset, val_dataset, test_dataset = load_datasets.transform_to_hfdataset(train_df, val_df, test_df)
        processor = pipeline_objects.create_processor()
        data_collator = DataCollatorForSpeechClassification(processor=processor)

        train_dataset = train_dataset.map(
            partial(load_datasets.prepare_dataset, processor=processor),
            remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(
            partial(load_datasets.prepare_dataset, processor=processor),
            remove_columns=val_dataset.column_names)
        test_dataset = test_dataset.map(
            partial(load_datasets.prepare_dataset, processor=processor),
            remove_columns=test_dataset.column_names)

        model = pipeline_objects.create_wav2vec2bert_for_classification(num_labels=2)

        if not pipeline_objects.is_model_saved(model_dir):  # Только если ещё не обучали
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
                compute_metrics=auxiliary.compute_metrics,
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
