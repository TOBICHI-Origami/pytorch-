
#%% [code]
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import torchtext
import torch
from torch import nn
import torch.nn.functional as F
from torchtext.data import get_tokenizer, to_map_style_dataset
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator

VOCAB_SIZE = 15000
class mo(nn.Module):
    def __init__( self, vocab_size = VOCAB_SIZE, embedding_dim = 64, num_class =2):
        super( mo, self).__init__()
        self.embedding_table = nn.Embedding( vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_table.weight)

        self.conv_A_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7)

        self.conv_A_2 = nn.Conv1d(64, 64, 15,stride=7)
        self.conv_B_2 = nn.Conv1d(64, 64, 15, stride=7)

        self.output_linear1 = nn.Linear(64, 128)
        self.output_linear2 = nn.Linear(128, num_class)

    def forward(self,word_index):
        word_embedding = self.embedding_table(word_index)

        word_embedding = word_embedding.transpose(1, 2)
        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)
        H = A * torch.sigmoid(B)

        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B)

        pool_output = torch.mean(H, dim=-1)
        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output)
        return logits
BATCH_SIZE = 64

def yield_token(train_data_iter, tokenizer):
    for i , sample in enumerate(train_data_iter):
        label , comment = sample
        yield tokenizer(comment)
train_data_iter = IMDB(root='./deeeeeep', split='train')
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(yield_token(train_data_iter,tokenizer), min_freq=20, specials=['<unk>'])
vocab.set_default_index(0)
print("单词表大小"+str(len(vocab)))

def collate_fn(batch):
    target = []
    token_index = []
    max_len=0
    for i ,(label,comment) in enumerate(batch):
        tokens = tokenizer(comment)
        token_index.append(vocab(tokens))
        if len(tokens) > max_len:
            max_len = len(tokens)
        if label == 'pos':
            target.append(0)
        else:
            target.append(1)
    token_index = [index+[0]*(max_len-len(index))for index in token_index]
    return (torch.tensor(target).to(torch.int64),torch.tensor(token_index).to(torch.int32))



