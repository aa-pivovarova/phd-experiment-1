import os
import pandas as pd
from datasets import Audio, Dataset

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
