from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import math
from utils import CfgNode as CN


class ShakespeareDataset(Dataset):
    def __init__(self, file_path: str, vocab_size: int, min_freq: int, block_size: int, batch_size:int, evaluate: bool = False):
        self.file_path =file_path
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.block_size = block_size
        self.evaluate = evaluate
        self.val_data = None
        self.train_data = None
        self.train_tokenized = []
        self.val_tokenized = []
        self.batch_size = batch_size
        self.train_data_location = Path("../Data/shakespeare-train.txt")
        self.val_data_location = Path("../Data/shakespeare-eval.txt")
        self.train_tokenizer()
        self.split_train_val()
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
        # tokenizer.enable_truncation()
        # or use the RobertaTokenizer from `transformers` directly.

        self.train_tokenized = tokenizer.encode(self.train_data)
        self.val_tokenized = tokenizer.encode(self.val_data)
        
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

    def get_batch(self):
        data = self.train_tokenized if self.evaluate == False else self.val_tokenized
        ix = torch.randint(len(data) - block_size, (self.batch_size,))

        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])

        return x, y
    
class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class Block(nn.Module):
    """ an unassuming Transformer block
    Unassuming in the sense of vanilla transformer block as minimalist block includes multiheaded causal self attention
    Feed forward network and layernorm
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class ToMTrainer(nn.Module):
    """
    Thanks Andrej
    """
    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = 8
        C.n_head = 4
        C.n_embd =  512
        # these options must be filled in externally
        C.vocab_size = 10000
        C.block_size = 8
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        if True:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'ToMTrainer':   dict(n_layer=8, n_head=4, n_embd=512),  # 117M params
            }[config.model_type])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): ## Initialization for linear layer
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): ## Embedding initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm): ## Layernorm initialization
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = ToMTrainer(config)
        sd = model.state_dict()

        return model
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss



if __name__ == "__main__":
    # TODO Add validation loss
    print('Hello pretrainers!')
    file_path = "../Data/tinyshakespeare.txt" # Location of data in .txt file format
    vocab_size = 10000  #Number of unique tokens
    min_freq = 2 # minimum frequency of tokens for it to be included in the vocabulary
    block_size = 8
    num_epochs = 1000
    shaky_data = ShakespeareDataset(file_path=file_path, vocab_size=vocab_size, min_freq=min_freq, block_size=block_size, batch_size=32)
    ToMTrainerModel = ToMTrainer.from_pretrained(model_type="ToMTrainer")
    optimizer = optim.AdamW(ToMTrainerModel.parameters(), lr=0.001)
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        xb, yb = shaky_data.get_batch()
        logits, loss  = ToMTrainerModel.forward(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss, epoch) #tensorboard --logdir=runs to run tensorboard found at : http://localhost:6006/ 
        
        # if epoch % 5 == 0:
        #     print(f'Epoch: {epoch}, current loss: {loss.item()}')

    writer.flush()

    