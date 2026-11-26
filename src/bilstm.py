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
import time
import argparse
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def _parse_args () :
    parser = argparse.ArgumentParser (description = "bilstm.py")
    parser.add_argument('--domain', choices = ["books", "movies"], default = "books", help = "Train on either 'books' or 'movies' review dataset.")
    parser.add_argument('--num_samples', type = int, default = 100_000)
    parser.add_argument('--num_epochs', type = int, default = 5, help = "Number of Epochs default is 5 and should require less for troubleshooting.")
    parser.add_argument('--max_length', type = int, default = 400, help = "Max length of reviews. 400 shouldn't need to be changed.")
    parser.add_argument('--batch_size', type = int, default = 32, help = "default set to 32")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate default = 0.001')
    parser.add_argument('--neg_per_pos', type = int, default = 3,
                    help = "Non-spoilers kept per spoiler in the training set. 0 keeps the natural ratio.")
    parser.add_argument ('--threshold', type = float, default = 0.5)
    parser.add_argument ('--dropout', type = float, default = 0.3, help = "Dropout rate after embeddings and before the output layer.")
    args = parser.parse_args ()
    return args

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
    def __init__ (self, vocab_size, embed_dim = 100, hidden_dim = 128, dropout = 0.3) :
        super ().__init__ ()
        self.embedding = nn.Embedding (vocab_size, embed_dim, padding_idx = 0)
        self.fwd = Single_Direction_LSTM (embed_dim, hidden_dim)
        self.rvrs = Single_Direction_LSTM (embed_dim, hidden_dim)
        self.rc_dropout = nn.Dropout (dropout)
        self.fc = nn.Linear (hidden_dim * 2, 1)

    def forward (self, x) :
        m = (x != 0).float ()
        em = self.embedding (x)
        h = torch.cat ([self.fwd (em, m), self.rvrs (em, m, reverse = True)], dim = 1)
        h = self.rc_dropout (h)

        return self.fc (h).squeeze (1)

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

def predict (self, text) :
    tokens = word_tokenize (text.lower ())
    ids = [self.vocabulary.get (tok, 1) for tok in tokens] [: max_length]
    x = torch.tensor ([ids], dtype = torch.long, device = self.device)

    self.model.eval ()
    with torch.no_grad () :
        prob = torch.sigmoid (self.model (x).item ())
    
    return int (prob > self.threshold), prob

def get_probs (model, loader, device) :
    model.eval ()
    all_probs = []
    all_labels = []

    with torch.no_grad () :
        for inputs, lengths, labels in loader :
            probs = torch.sigmoid (model (inputs.to (device)))
            all_probs.extend (probs.cpu ().tolist ())
            all_labels.extend (labels.long ().tolist ())

    return np.array (all_probs), np.array (all_labels)

def metrics_at (probs, labels, threshold) :
   
    preds = (probs > threshold).astype (int)
    
    return {
        "accuracy" : accuracy_score (labels, preds),
        "precision" : precision_score (labels, preds, zero_division = 0),
        "recall" : recall_score (labels, preds, zero_division = 0),
        "f1" : f1_score (labels, preds, zero_division = 0),
    }

def evaluate (model, loader, device, threshold = 0.5) :
    
    probs, labels = get_probs (model, loader, device)
    
    return metrics_at (probs, labels, threshold)

BOOK_PATH = "df_book_reviews_spoiler_features.csv"
MOVIE_PATH = "df_movie_reviews_spoiler_features.csv"
torch.manual_seed (42)
random.seed (42)
np.random.seed (42)
args = _parse_args () 
print (args)

sample_size = args.num_samples
epochs = args.num_epochs
max_length = args.max_length
size_of_batch = args.batch_size 
learning_rate = args.lr
cli_threshold = args.threshold
drop = args.dropout

if args.domain == "books" : 
    review_path = BOOK_PATH
else :
    review_path = MOVIE_PATH

nltk.download ("punkt_tab")
DATA_PATH = Path (__file__).resolve ().parent.parent / "dataset" / review_path
print ("Data path: ", DATA_PATH)
df = pd.read_csv (DATA_PATH)

df = df [["review_text", "has_spoiler"]]
df ["has_spoiler"] = df ["has_spoiler"].astype (int)
df = df.dropna (subset = ["review_text"])

series = df ["has_spoiler"].value_counts ()
ratio = series [1] / series [0]

print ("df.shape: ", df.shape)
print (df ["has_spoiler"].value_counts ())

df_small = df if sample_size is None else df.sample (sample_size, random_state = 42)
df_small ["tokens"] = df_small ["review_text"].str.lower ().apply (word_tokenize)

train_df, temp_df = train_test_split (df_small, test_size = 0.3, 
                                    stratify = df_small ["has_spoiler"], random_state = 42)

val_df, test_df = train_test_split (temp_df, test_size = 0.5, stratify = temp_df ["has_spoiler"], random_state = 42)

counts = Counter (tok for tokens in train_df ["tokens"] for tok in tokens)
vocabulary = {"<PAD>" : 0, "<UNK>" : 1}

for word, count in counts.items () :
    if count >= 2 :
        vocabulary [word] = len (vocabulary)

def encode (tokens) :
    ids = [vocabulary.get (tok, 1) for tok in tokens]
    return ids [: max_length]

for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
    counts_splits = part ["has_spoiler"].value_counts ()
    print (f"{name} : {len (part)} reviews, {counts_splits [1]} spoilers ({counts_splits [1] / len (part):.1%})")
    
lengths_all = train_df ["tokens"].apply (len)

train_df ["ids"] = train_df ["tokens"].apply (encode)
val_df ["ids"] = val_df ["tokens"].apply (encode)
test_df ["ids"] = test_df ["tokens"].apply (encode)

n_neg = (train_df ["has_spoiler"] == 0).sum ()
n_pos = (train_df ["has_spoiler"] == 1).sum ()

val_ids = [i for ids in val_df ["ids"] for i in ids]

print (lengths_all.groupby (train_df ["has_spoiler"]).describe ())
print (f"truncated : {(lengths_all > max_length).mean ():.1%}")
print ("vocabulary size : ", len (vocabulary))
print ("describe train_fd: ", train_df ["tokens"].apply (len).describe ())
print ("")
print ("describe val_df: ", val_df ["tokens"].apply (len).describe ())
print ("")
print ("describe test_df: ", test_df ["tokens"].apply (len).describe ())
print (f"Percentage of unknown words in validation : {val_ids.count (1) / len (val_ids):.1%}")

train_loader = DataLoader (Spoiler_Dataset (train_df), batch_size = size_of_batch, shuffle = True, collate_fn = collate)
val_loader = DataLoader (Spoiler_Dataset (val_df), batch_size = size_of_batch, collate_fn = collate)
test_loader = DataLoader (Spoiler_Dataset (test_df), batch_size = size_of_batch, collate_fn = collate)

x, lengths, y = next (iter (train_loader))

device = torch.device ("mps" if torch.backends.mps.is_available () else "cpu")
model = BiLSTM (vocab_size = len (vocabulary), dropout = drop).to (device)
optimizer = torch.optim.Adam (model.parameters (), lr = learning_rate)
criterion = nn.BCEWithLogitsLoss (pos_weight = torch.tensor (n_neg / n_pos, dtype = torch.float, device = device ))

best_f1 = -1.0
total_start = time.time ()
print ("Training start")

for epoch in range (epochs) :

    model.train ()
    running_loss = 0.0
    start = time.time ()

    for inputs, lengths, labels in (train_loader) :
        inputs = inputs.to (device)
        labels = labels.to (device)

        optimizer.zero_grad ()
        outputs = model (inputs)
        loss = criterion (outputs, labels)
        loss.backward ()
        torch.nn.utils.clip_grad_norm_ (model.parameters (), max_norm = 1.0)
        optimizer.step ()

        running_loss += loss.item () * len (labels)
    
    val = evaluate (model, val_loader, device)
    print (f'epoch {epoch + 1}  loss {running_loss:.1f}  '
           f'val acc {val ['accuracy']:.3f}  val P {val ['precision']:.3f}  '
           f'val R {val ['recall']:.3f}  val F1 {val ['f1']:.3f}  ({time.time () - start:.0f} s)')

    if val ['f1'] > best_f1 : 
        best_f1 = val ['f1']
        torch.save (model.state_dict (), 'best_bilstm.pt')

model.load_state_dict (torch.load ('best_bilstm.pt'))

val_probs, val_labels = get_probs (model, val_loader, device)
test_probs, test_labels = get_probs (model, test_loader, device)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_t = max (thresholds, key = lambda t : metrics_at (val_probs, val_labels, t) ["f1"])

for t in thresholds :
    v = metrics_at (val_probs, val_labels, t)
    s = metrics_at (test_probs, test_labels, t)
    print (f't={t:.1f}  val F1 {v ["f1"]:.3f}  |  TEST acc {s ["accuracy"]:.3f}  P {s ["precision"]:.3f}  '
           f'R {s ["recall"]:.3f}  F1 {s ["f1"]:.3f}')

test = metrics_at (test_probs, test_labels, best_t)
print (f'BEST threshold (chosen on val) = {best_t}')
print (f'TEST  acc {test ["accuracy"]:.3f}  P {test ["precision"]:.3f}  '
       f'R {test ["recall"]:.3f}  F1 {test ["f1"]:.3f}')

total = time.time () - total_start
print (f'Total time: {total / 60:.1f} min ({total:.0f} s)')