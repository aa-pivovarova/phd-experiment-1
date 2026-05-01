import torch
from dataclasses import dataclass

@dataclass
class DataCollatorForSpeechClassification:
    processor: object
    padding: bool = True

    def __call__(self, features):
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        inputs = self.processor.feature_extractor(
            raw_speech=input_values,
            sampling_rate=16000,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True  # важно для некоторых моделей
        )

        batch = {
            "input_features": inputs.input_features,
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        if "attention_mask" in inputs:
            batch["attention_mask"] = inputs.attention_mask
        return batch