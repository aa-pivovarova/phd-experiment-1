from huggingface_hub import notebook_login
from datasets import load_dataset, Audio, ClassLabel
from dataclasses import dataclass, field
import random
import pandas as pd
import IPython.display as ipd
from IPython.display import display, HTML
import re
import json
from transformers import Wav2Vec2CTCTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor, Wav2Vec2BertForCTC, TrainingArguments, Trainer, AutoModelForCTC, Wav2Vec2BertProcessor
import torch
import torchaudio
from typing import Any, Dict, List, Optional, Union
import evaluate
import numpy as np
import random

from DataCollatorCTCWithPadding import DataCollatorCTCWithPadding
import functions

def main() -> int:
    common_voice_train, common_voice_validation, common_voice_test = functions.load_and_prepare_dataset()
    functions.preprocess_data(common_voice_train, common_voice_validation, common_voice_test)
    
    repo_name = "aapivovarova-w2v-bert-2.0-russian-pycharm"
    processor = functions.create_tokenizer_and_processor(repo_name)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    model = functions.wav2vec2bertforctc_create(processor)
    
    functions.trainer(repo_name, model, data_collator, common_voice_train, common_voice_test, processor)
    
    functions.evaluate(common_voice_test)

if __name__=="__main__":
    main()