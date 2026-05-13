from transformers import XLMRobertaTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor, Wav2Vec2BertForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from dataclasses import dataclass
import os
import auxiliary

@dataclass
class TrainingConfig:
    train_size: float
    val_size: float
    test_size: float
    learning_rate: float
    batch_size: int
    epochs: int

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

def create_trainer(
        fold, config, repo_name, model,
        data_collator, train_dataset,
        val_dataset, processor):
    print("Starting training...")
    training_args = TrainingArguments(
        output_dir=f"{repo_name}-fold-{fold}",
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        fp16=True,
        optim="adafactor",
        num_train_epochs=config.epochs,
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
        per_device_eval_batch_size=1,
        disable_tqdm=False,
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=auxiliary.compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(f"{repo_name}-fold-{fold}/best_model")
    auxiliary.build_graphs(fold, trainer)
    return metrics