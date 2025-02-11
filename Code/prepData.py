import torch

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset


class TokenizedTextDataset(Dataset):
    def __init__(self, tokenized_text: torch.Tensor, seq_length: int):
        """
        Dataset for training a language model.
        - tokenized_text (torch.Tensor): A single long tensor of tokenized text.
        - seq_length (int): The sequence length for input samples.
        """
        self.tokenized_text = tokenized_text
        self.seq_length = seq_length

    def __len__(self):
        """Returns the number of possible sequences in the dataset."""
        return len(self.tokenized_text) - self.seq_length

    def __getitem__(self, idx):
        """Returns (x, y) pair where y is the next token prediction."""
        x = self.tokenized_text[idx : idx + self.seq_length]       # Input sequence
        y = self.tokenized_text[idx + 1 : idx + self.seq_length + 1]  # Target sequence (shifted by 1)
        return x, y
    
class TokenizeData(Dataset):
    def __init__(self, DataConfig):
        self.file_path = DataConfig.file_path
        self.min_freq = DataConfig.min_freq
        self.block_size = DataConfig.block_size
        self.evaluate = False
        self.vocab_size = None
        self.val_data = None
        self.train_data = None
        self.tokenizer = None
        self.train_tokenized = []
        self.val_tokenized = []
        self.train_data_location = Path("../Data/shakespeare-train.txt")
        self.val_data_location = Path("../Data/shakespeare-eval.txt")
        self.set_vocab_size()
        self.train_tokenizer()
        self.split_train_val()
        self.tokenize_data()

    def tokenize_data(self):
        self.tokenizer = ByteLevelBPETokenizer(
            "shakespeare-vocab.json",
            "shakespeare-merges.txt",
        )
        self.tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", self.tokenizer.token_to_id("</s>")),
            ("<s>", self.tokenizer.token_to_id("<s>")),
        )
        # tokenizer.enable_truncation()
        # or use the RobertaTokenizer from `transformers` directly.

        self.train_tokenized = self.tokenizer.encode(self.train_data)
        self.val_tokenized = self.tokenizer.encode(self.val_data)
        
        self.train_tokenized = torch.tensor(self.train_tokenized.ids, dtype=torch.long)
        self.val_tokenized = torch.tensor(self.val_tokenized.ids, dtype=torch.long)
        tokens = len(self.train_tokenized)
        print(f'total number of train tokens: {tokens}')

    def train_tokenizer(self):
        """ 
        Byte-pair encoding tokenizer trained using standard huggingface module. Parameters are training data (.txt),
        vocab size and minimum frequency for appearances 
        saves json with vocab and .txt of merges made
        """

        paths =[self.file_path]
        print(f'Retrieved training data from: {paths}')
        # Initialize a tokenizer
        self.tokenizer = ByteLevelBPETokenizer()

        # Customize training
        #TODO Add start/end tokens?
        self.tokenizer.train(files=paths, vocab_size=self.vocab_size, min_frequency=self.min_freq, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        # Save files to disk
        self.tokenizer.save_model(".", "shakespeare")

    def split_train_val(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        n = int(0.9*len(data)) #Use first 90% of data for training
        self.train_data = data[:n]
        self.val_data = data[n:]
        with open('../Data/shakespeare-train.txt', 'w') as output:
            output.write(self.train_data)
        with open('../Data/shakespeare-eval.txt', 'w') as output:
            output.write(self.val_data)

    def set_vocab_size(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        unique_words = len(set(data.split(' ')))
        print(f'Number of unique words: {unique_words}')
        self.vocab_size = round(unique_words*0.75)
        print(f'Using BPE suggested vocab size: {self.vocab_size}')    
    
    def get_tokenized_data(self):
        return self.train_tokenized, self.val_tokenized
    