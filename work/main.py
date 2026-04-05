import functions
from DataCollatorCTCWithPadding import DataCollatorCTCWithPadding

def main() -> int:
    print("Starting main...")
    repo_name = "aapivovarova-w2v-bert-2.0-russian-pycharm"
    common_voice_train, common_voice_validation, common_voice_test = functions.load_dataset_and_create_vocab()
    processor = functions.create_tokenizer_and_processor(repo_name)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    model = functions.wav2vec2bertforctc_create(processor)
    common_voice_train, common_voice_validation, common_voice_test = functions.preprocess_data(processor, common_voice_train, common_voice_validation, common_voice_test)
    functions.trainer(repo_name, model, data_collator, common_voice_train, common_voice_test, processor)
    functions.evaluate(common_voice_test)

if __name__=="__main__":
    main()