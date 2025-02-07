from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
import torch


class ShakespeareDataset(Dataset):
    def __init__(self, file_path: str, vocab_size: int, min_freq: int, evaluate: bool = False):
        self.file_path =file_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.evaluate = evaluate
        self.train_tokenizer()
        self.split_train_test()
        self.tokenize_data()

    def tokenize_data(self):
        tokenizer = ByteLevelBPETokenizer(
            "shakespeare-vocab.json",
            "shakespeare-merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = Path("../Data/").glob("*-eval.txt") if self.evaluate else Path("../Data/").glob("*-train.txt")
        for src_file in src_files:
            print("🔥Burning through🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]
        tokens = sum(len(lst) for lst in self.examples)
        print(f'number of samples:{len(self.examples)}')
        print(f'total number of tokens: {tokens}')


    def train_tokenizer(self):
        """ 
        Byte-pair encoding tokenizer trained using standard huggingface module. Parameters are training data (.txt),
        vocab size and minimum frequency for appearances 
        saves json with vocab and .txt of merges made
        """

        # paths = [str(x) for x in Path(file_path).glob("**/*.txt")]
        paths =[self.file_path]
        print(f'Retrieved training data from: {paths}')
        # Initialize a tokenizer
        tokenizer = ByteLevelBPETokenizer()

        # Customize training
        tokenizer.train(files=paths, vocab_size=self.vocab_size, min_frequency=self.min_freq, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        # Save files to disk
        tokenizer.save_model(".", "shakespeare")

    def split_train_test(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        n = int(0.9*len(data)) #Use first 90% of data for training
        train_data = data[:n]
        val_data = data[n:]
        with open('../Data/shakespeare-train.txt', 'w') as output:
            output.write(train_data)
        with open('../Data/shakespeare-eval.txt', 'w') as output:
            output.write(val_data)


if __name__ == "__main__":
    print('Hello pretrainers!')
    file_path = "../Data/tinyshakespeare.txt" # Location of data in .txt file format
    vocab_size = 18000  #Number of unique tokens
    min_freq = 2 # minimum frequency of tokens for it to be included in the vocabulary

    shaky_data = ShakespeareDataset(file_path=file_path, vocab_size=vocab_size, min_freq=min_freq)
