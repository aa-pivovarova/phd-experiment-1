from huggingface_hub import notebook_login
from datasets import load_dataset, Audio, ClassLabel
import random
import pandas as pd
import IPython.display as ipd
from IPython.display import display, HTML
import re
import json
from transformers import Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, TrainingArguments, Trainer, AutoModelForCTC
import torch
import torchaudio
import evaluate
import numpy as np
import random
import torchcodec
from functools import partial

#############################################################################
### HELPER FUNCTIONS

def huggingface_login():    
    notebook_login()

def show_random_elements(dataset, num_examples=10):
    picks = []
    for i, example in enumerate(dataset):
        if i >= num_examples:
            break
        picks.append(example)

    df = pd.DataFrame(picks)
    display(HTML(df.to_html()))

def remove_special_characters(batch, chars_to_remove_regex):
    # remove special characters
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def remove_latin_characters(batch):
    batch["sentence"] = re.sub(r'[a-z]+', '', batch["sentence"])
    return batch

def prepare_dataset(processor, batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])

    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

#################################################################################
###### BODY FUNCTIONS

def load_and_prepare_dataset():
    print("Loading and preparing dataset...")
    common_voice_train = load_dataset("fixie-ai/common_voice_17_0", "ru", split="train")
    common_voice_validation = load_dataset("fixie-ai/common_voice_17_0", "ru", split="validation")
    common_voice_test = load_dataset("fixie-ai/common_voice_17_0", "ru", split="test")

    # Remove unneeded columns
    print("Removing unneeded columns")
    common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_validation = common_voice_validation.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])

    print("Showing random elements in train")
    show_random_elements(common_voice_train.cast_column("audio", Audio(sampling_rate=16000)), num_examples=10)

    # Remove special characters
    print("Removing Special Characters")
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\ \'\»\«]'
    common_voice_train = common_voice_train.map(remove_special_characters, chars_to_remove_regex=chars_to_remove_regex)
    common_voice_validation = common_voice_validation.map(remove_special_characters,  chars_to_remove_regex=chars_to_remove_regex)
    common_voice_test = common_voice_test.map(remove_special_characters, chars_to_remove_regex=chars_to_remove_regex)

    print("Showing random elements in train")
    show_random_elements(common_voice_train.remove_columns(["path","audio"]))

    # Show unique characters
    print("Showing unique characters")
    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
    vocab_validation = common_voice_validation.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_validation.column_names)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
    
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    vocab_dict

    # Remove latin characters
    print("Removing latin characters")
    common_voice_train = common_voice_train.map(remove_latin_characters)
    common_voice_validation = common_voice_validation.map(remove_latin_characters)
    common_voice_test = common_voice_test.map(remove_latin_characters)
    
    # Extract unique characters again
    print("Extracting unique characters again")
    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
    vocab_validation = common_voice_validation.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_validation.column_names)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)
    
    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_validation["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
    print(vocab_dict)
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print(len(vocab_dict))

    print("Dataset loaded and prepared")
    # Save vocabulary as json
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    return common_voice_train, common_voice_validation, common_voice_test

def preprocess_data(processor, common_voice_train, common_voice_validation, common_voice_test):
    print("Preprocessing data...")
    print(torch.__version__)
    print(torchaudio.__version__)
    print(torchcodec.__version__)
    
    print(common_voice_train[0]["path"])
    print(common_voice_train[0]["audio"])

    print("Casting columns")
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_validation = common_voice_validation.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
    print(common_voice_train[0]["audio"])
    
    rand_int = random.randint(0, len(common_voice_train)-1)
    
    print(common_voice_train[rand_int]["sentence"])
    ipd.Audio(data=common_voice_train[rand_int]["audio"]["array"], autoplay=True, rate=16000)
    
    rand_int = random.randint(0, len(common_voice_train)-1)
    
    print("Target text:", common_voice_train[rand_int]["sentence"])
    print("Input array shape:", common_voice_train[rand_int]["audio"]["array"].shape)
    print("Sampling rate:", common_voice_train[rand_int]["audio"]["sampling_rate"])

    common_voice_train = common_voice_train.map(prepare_dataset, processor=processor, remove_columns=common_voice_train.column_names)
    common_voice_validation = common_voice_validation.map(prepare_dataset, processor=processor, remove_columns=common_voice_validation.column_names)
    common_voice_test = common_voice_test.map(prepare_dataset, processor=processor, remove_columns=common_voice_test.column_names)
    print("Data preprocessed")

def compute_metrics(pred, processor):
    print("Computing metrics...")
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    print("WER:", wer)
    return {"wer": wer}

def create_tokenizer_and_processor(repo_name):
    print("Creating tokenizer...")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        "./",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
        )
    print("Tokenizer created")
    tokenizer.push_to_hub(repo_name)
    print("Tokenizer pushed to hub")

    print("Creating processor...")
    processor = Wav2Vec2BertProcessor(
        feature_extractor = SeamlessM4TFeatureExtractor(
            feature_size=80,
            num_mel_bins=80,
            sampling_rate=16000,
            padding_value=0.0),
        tokenizer = tokenizer
        )
    print("Processor created")
    processor.push_to_hub(repo_name)
    print("Processor pushed to hub")
    return processor

def wav2vec2bertforctc_create(processor):
    print("Creating WAV2VEC2BERTFORCTC model...")
    model = Wav2Vec2BertForCTC.from_pretrained(
        "facebook/w2v-bert-2.0",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    print("Model created")
    return model

def trainer(repo_name, model, data_collator, common_voice_train, common_voice_test, processor):
    print("Starting training...")
    training_args = TrainingArguments(
      output_dir=repo_name,
      group_by_length=True,
      per_device_train_batch_size=16,
      gradient_accumulation_steps=2,
      evaluation_strategy="steps",
      num_train_epochs=10,
      gradient_checkpointing=True,
      fp16=True,
      save_steps=600,
      eval_steps=300,
      logging_steps=300,
      learning_rate=5e-5,
      warmup_steps=500,
      save_total_limit=2,
      push_to_hub=True,
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=partial(compute_metrics, processor=processor),
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    print("Finished training.")
    trainer.push_to_hub()
    return training_args

def evaluate(common_voice_test):
    print("Beginning evaluation...")
    model = AutoModelForCTC.from_pretrained("ylacombe/w2v-bert-2.0-mongolian-colab-CV16.0").to("cuda")
    processor = Wav2Vec2BertProcessor.from_pretrained("ylacombe/w2v-bert-2.0-mongolian-colab-CV16.0")
    input_dict = common_voice_test[0]
    logits = model(torch.tensor(input_dict["input_features"]).to("cuda").unsqueeze(0)).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]
    processor.decode(pred_ids)
    processor.decode(input_dict["labels"]).lower()

##===================================================================================