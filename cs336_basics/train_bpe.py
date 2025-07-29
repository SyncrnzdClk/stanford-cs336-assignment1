from bpe_tokenizer_train import bpe_tokenizer_training
import pathlib
import time
import pickle
DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
DATA_FILE_NAME = "TinyStoriesV2-GPT4-Train.txt"

def main():
    input_path = DATA_PATH / DATA_FILE_NAME
    start_time = time.time()
    vocab, merges = bpe_tokenizer_training(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()

    print(f"BPE Training on {DATA_FILE_NAME} Finish! Time Cost: {end_time-start_time}")
    print("Saving vocabulary and merges list to disk...")

    with open(DATA_PATH/"vocabulary.pkl", "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DATA_PATH/"merges.pkl", "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == "__main__":
    main()