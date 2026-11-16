import torch
import torch.nn as nn
import nltk 
from nltk.tokenize import word_tokenize
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class Spoiler_Dataset (Dataset) :
    def __init__ (self, df) :
        self.ids = df ["ids"].tolist ()
        self.labels = df ["has_spoiler"].tolist ()

    def __len__ (self) :
        return len (self.labels)
    
    def __getitem__ (self, i) :
        return torch.tensor (self.ids [i], dtype = torch.long), torch.tensor (self.labels [i], dtype = torch.float)
    
def collate (batch) : 
    seqs, labels = zip (*batch)
    lengths = torch.tensor ([len (s) for s in seqs])
    padded = pad_sequence (seqs, batch_first = True, padding_value = 0)
    return padded, lengths, torch.stack (labels)

class Single_Direction_LSTM (nn.Module) :
    def __init__ (self, n_inputs, n_hiddens, Σ = 0.01) :
        super ().__init__ ()
        self.n_hiddens = n_hiddens
        init = lambda * shape : nn.Parameter (torch.randn (*shape) * Σ)
        triple = lambda : (init (n_inputs, n_hiddens),
                           init (n_hiddens, n_hiddens),
                           nn.Parameter (torch.zeros (n_hiddens)))
        
        self.W_xi, self.W_hi, self.b_i = triple ()
        self.W_xf, self.W_hf, self.b_f = triple ()
        self.W_xo, self.W_ho, self.b_o = triple ()
        self.W_xc, self.W_hc, self.b_c = triple ()

    def forward (self, X, mask, reverse = False) :        
        B, T, _ = X.shape
        H = torch.zeros (B, self.n_hiddens, device = X.device)
        C = torch.zeros_like (H)
        steps = range (T -1 , -1, -1) if reverse else range (T)
        
        for t in steps :
            x = X [:, t]
            I = torch.sigmoid (x @ self.W_xi + H @ self.W_hi + self.b_i)
            F = torch.sigmoid (x @ self.W_xf + H @ self.W_hf + self.b_f)
            O = torch.sigmoid (x @ self.W_xo + H @ self.W_ho + self.b_o)
            C_ = torch.tanh (x @ self.W_xc + H @ self.W_hc + self.b_c)

            C_prime = F * C + I * C_
            H_prime = O * torch.tanh (C_prime)
            m = mask [:, t].unsqueeze (1)
            C = m * C_prime + (1 - m) * C
            H = m * H_prime + (1 - m) * H
        return H

class BiLSTM (nn.Module) :
    def __init__ (self, vocab_size, embed_dim = 100, hidden_dim = 128) :
        super ().__init__ ()
        self.embedding = nn.Embedding (vocab_size, embed_dim, padding_idx = 0)
        self.fwd = Single_Direction_LSTM (embed_dim, hidden_dim)
        self.rvrs = Single_Direction_LSTM (embed_dim, hidden_dim)
        self.fc = nn.Linear (hidden_dim * 2, 1)

    def forward (self, x) :
        m = (x != 0).float ()
        em = self.embedding (x)
        h = torch.cat ([self.fwd (em, m), self.rvrs (em, m, reverse = True)], dim = 1)
        return self.fc (h).squeeze (1)

SAMPLE_SIZE = 5_000
MAX_LENGTH = 400

nltk.download ("punkt_tab")
DATA_PATH = Path (__file__).resolve ().parent.parent / "dataset" / "df_book_reviews_spoiler_features.csv"
print ("Data path: ", DATA_PATH)
df = pd.read_csv (DATA_PATH)

df = df [["review_text", "has_spoiler"]]
df ["has_spoiler"] = df ["has_spoiler"].astype (int)
df = df.dropna (subset = ["review_text"])

print ("df.shape: ", df.shape)
series = df ["has_spoiler"].value_counts ()
print (df ["has_spoiler"].value_counts ())
ratio = series [0] / series [1]
print ("Ratio of non-spoilers to spoilers = " f"{ratio:.2f}")
print ("Sample size = ", SAMPLE_SIZE)

df_small = df if SAMPLE_SIZE is None else df.sample (SAMPLE_SIZE, random_state = 42)
df_small ["tokens"] = df_small ["review_text"].str.lower ().apply (word_tokenize)

print (df_small ["tokens"].iloc [0] [:20])

train_df, temp_df = train_test_split (df_small, test_size = 0.3, 
                                    stratify = df_small ["has_spoiler"], random_state = 42)

val_df, test_df = train_test_split (temp_df, test_size = 0.5, stratify = temp_df ["has_spoiler"], random_state = 42)

counts = Counter (tok for tokens in train_df ["tokens"] for tok in tokens)

vocabulary = {"<PAD>" : 0, "<UNK>" : 1}

for word, count in counts.items () :
    if count >= 2 :
        vocabulary [word] = len (vocabulary)

print ("vocabulary length: ", len (vocabulary))
print ("describe: ", train_df ["tokens"].apply (len).describe ())

def encode (tokens) :
    ids = [vocabulary.get (tok, 1) for tok in tokens]
    return ids [: MAX_LENGTH]

for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    counts_splits = part ["has_spoiler"].value_counts ()
    print (f"{name} : {len (part)} reviews, {counts_splits [1]} spoilers ({counts_splits [1] / len (part):.1%})")
    
lengths_all = train_df ["tokens"].apply (len)
print (lengths_all.groupby (train_df ["has_spoiler"]).describe ())

print (f"truncated : {(lengths_all > MAX_LENGTH).mean ():.1%}")

print ("vocabulary size : ", len (vocabulary))

train_df ["ids"] = train_df ["tokens"].apply (encode)
val_df ["ids"] = val_df ["tokens"].apply (encode)
test_df ["ids"] = test_df ["tokens"].apply (encode)

val_ids = [i for ids in val_df ["ids"] for i in ids]
print (f"unknown words in validation : {val_ids.count (1) / len (val_ids):.1%}")

print (train_df ["ids"].iloc [0] [: 20])

train_loader = DataLoader (Spoiler_Dataset (train_df), batch_size = 32, shuffle = True, collate_fn = collate)
val_loader = DataLoader (Spoiler_Dataset (val_df), batch_size = 32, collate_fn = collate)
test_loader = DataLoader (Spoiler_Dataset (test_df), batch_size = 32, collate_fn = collate)

x, lengths, y = next (iter (train_loader))
print (x.shape, lengths [: 5], y [: 5])