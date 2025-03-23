#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence
import os
seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# In[2]:


class MultiHeadAttention(nn.Module):
    def __init__(self, emd_dim, heads=4, dropout = 0.2):
        super().__init__()
        assert emd_dim % heads == 0
        self.heads = heads
        self.head_dim = emd_dim//heads
        self.scale = self.head_dim ** -0.5
        self.multiHead = nn.Linear(emd_dim, emd_dim*3)
        self.output = nn.Linear(emd_dim,emd_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def add_masking(attn_scores, padding_mask):
        col_mask = padding_mask[:, None, None, :]
        attn_scores.masked_fill_((col_mask == 0), float('-inf'))
        return attn_scores

    def forward(self, x, padding_mask=None, attn_mask=False):
        B, T, C = x.shape
        qkv = self.multiHead(x)
        q, k, v = torch.chunk(qkv,3,dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, T, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask:
            tril = torch.tril(torch.ones(T,T))
            attn_scores = attn_scores.masked_fill(tril==0, float('-inf'))
        if padding_mask is not None:
            attn_scores = self.add_masking(attn_scores, padding_mask)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs_drop = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs_drop,v)
        fn_attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.output(fn_attn_output)


# In[3]:


class CrossAttention(nn.Module):
    def __init__(self, emd_dim, heads=4, dropout = 0.2):
        super().__init__()
        assert emd_dim % heads == 0
        self.heads = heads
        self.head_dim = emd_dim//heads
        self.scale = self.head_dim ** -0.5
        self.Wk = nn.Linear(emd_dim, emd_dim)
        self.Wq = nn.Linear(emd_dim, emd_dim)
        self.Wv = nn.Linear(emd_dim, emd_dim)
        self.output = nn.Linear(emd_dim,emd_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def add_masking(attn_scores, padding_mask):
        if padding_mask is not None:
            col_mask = padding_mask[:, None, None, :]
            attn_scores.masked_fill_((col_mask == 0), float('-inf'))
        return attn_scores

    def forward(self, x, encoder_outputs, padding_mask=None):
        B, T_trg, C = x.shape
        _, T_src, _ = encoder_outputs.shape
        key = self.Wk(encoder_outputs)
        query = self.Wq(x)
        values = self.Wv(encoder_outputs)
        query = query.view(B, T_trg, self.heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(B, T_src, self.heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(B, T_src, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_scores = self.add_masking(attn_scores, padding_mask)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs_drop = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs_drop,values)
        fn_attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T_trg, C)
        return self.output(fn_attn_output)


# In[4]:


class LayerNorm1D(nn.Module):
  def __init__(self, dim, eps=1e-5):
    super(LayerNorm1D, self).__init__()
    self.gamma = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1,keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    xhat = (x-mean)/torch.sqrt(var+self.eps)
    return (self.gamma * xhat) +self.beta


# In[5]:


class FeedForward(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.2):
    super().__init__()
    self.feed_forward_layer = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, output_dim),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.feed_forward_layer(x)


# In[6]:


class EncoderBlock(nn.Module):
    def __init__(self,embed_dim, heads=4):
        super().__init__()
        self.layer_norm1 = LayerNorm1D(embed_dim)
        self.layer_norm2 = LayerNorm1D(embed_dim)
        self.multi_head_attn =  MultiHeadAttention(embed_dim, heads)
        self.feed_forward_layer = FeedForward(embed_dim, embed_dim*4, embed_dim)
    
    def forward(self, x, padding_mask):
        x = x + self.multi_head_attn(self.layer_norm1(x), padding_mask)
        x = x + self.feed_forward_layer(self.layer_norm2(x))
        return x


# In[7]:


class Encoder(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, src_max_length, heads = 4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(src_max_length, embed_dim)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim,heads) for _ in range(num_layers)])

    def forward(self, x, padding_mask = None):
        _, T = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding(torch.arange(T))
        for block in self.encoder_blocks:
            x = block(x, padding_mask = padding_mask) 
        return x
        


# In[8]:


class DecoderBlock(nn.Module):
    def __init__(self,embed_dim, heads=4):
        super().__init__()
        self.layer_norm1 = LayerNorm1D(embed_dim)
        self.layer_norm2 = LayerNorm1D(embed_dim)
        self.layer_norm3 = LayerNorm1D(embed_dim)
        self.multi_head_attn =  MultiHeadAttention(embed_dim, heads)
        self.cross_attn = CrossAttention(embed_dim, heads)
        self.feed_forward_layer = FeedForward(embed_dim, embed_dim*4, embed_dim)
    
    def forward(self, x, encoder_outputs, src_mask, trg_mask, attn_mask = True):
        x = x + self.multi_head_attn(self.layer_norm1(x), padding_mask = trg_mask,attn_mask = attn_mask)
        x = x + self.cross_attn(self.layer_norm2(x), encoder_outputs, src_mask)
        x = x + self.feed_forward_layer(self.layer_norm3(x))
        return x
        


# In[9]:


class Decoder(nn.Module):
    def __init__(self, embed_dim, trg_vocab_size, trg_max_length, heads = 4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(trg_vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(trg_max_length, embed_dim)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim,heads) for _ in range(num_layers)])

    def forward(self, x, encoder_outputs, src_mask = None, trg_mask = None):
        _, T = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding(torch.arange(T))
        for block in self.decoder_blocks:
            x = block(x, encoder_outputs, src_mask, trg_mask, attn_mask = True) 
        return x
        


# In[10]:


class Transformers(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, src_max_length, trg_vocab_size, trg_max_length, heads = 4, encoder_layers = 4, decoder_layers = 4):
        super().__init__()
        self.encoder = Encoder(embed_dim, src_vocab_size, src_max_length, heads = 4, num_layers = 4)
        self.decoder = Decoder(embed_dim, trg_vocab_size, trg_max_length, heads = 4, num_layers = 4)
        self.linear = nn.Linear(embed_dim, trg_vocab_size)

    def forward(self, src, src_mask, trg, trg_mask):
        encoder_outputs = self.encoder(src, src_mask)
        decoder_outputs = self.decoder(trg, encoder_outputs, src_mask, trg_mask)
        output = self.linear(decoder_outputs)
        return output


# In[11]:


path = r'C:\\Users\\harish-4072\\Downloads\\eng_french.csv'
df = pd.read_csv(path, names=['English','French'], header=0)


# In[12]:


def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()  
    return tokens


# In[13]:


english_sentences = df['English'].dropna().apply(preprocess_text)
english_vocab = Counter([token for sentence in english_sentences for token in sentence])

french_sentences = df['French'].dropna().apply(preprocess_text)
french_vocab = Counter([token for sentence in french_sentences for token in sentence])


# In[14]:


english_token_to_id = {token: idx + 1 for idx, token in enumerate(english_vocab)}  # Start from 1 to reserve 0 for padding
french_token_to_id = {token: idx + 3 for idx, token in enumerate(french_vocab)}

english_token_to_id['<PAD>'] = 0

french_token_to_id['<PAD>'] = 0
french_token_to_id['<SOS>'] = 1
french_token_to_id['<EOS>'] = 2


# In[15]:


french_id_to_token= {value:key for key,value in french_token_to_id.items()}
english_id_to_token= {value:key for key,value in english_token_to_id.items()}


# In[16]:


english_vocab_size = len(english_token_to_id)
french_vocab_size = len(french_token_to_id)


# In[17]:


def tokenize_text(tokens,token_to_id):
    return [token_to_id.get(token,0) for token in tokens]

english_sequences = english_sentences.apply(lambda x: tokenize_text(x, english_token_to_id))
french_sequences = french_sentences.apply(lambda x: tokenize_text(x, french_token_to_id))


# In[18]:


def add_sos_eos(tokens):
    return [1]+tokens+[2]


# In[19]:


french_sequences = french_sequences.apply(lambda x: add_sos_eos(x))


# In[20]:


class SentencesDataset(Dataset):
    def __init__(self,english_sequences,french_sequences):
        self.english_sequences = english_sequences
        self.french_sequences = french_sequences
        assert len(self.english_sequences) == len(self.french_sequences)

    def __len__(self):
        return len(self.english_sequences)

    def __getitem__(self,idx):
        X= self.english_sequences[idx]
        y= self.french_sequences[idx]
        return torch.tensor(X,dtype=torch.long),torch.tensor(y,dtype=torch.long)


# In[21]:


def collate_fn(batch):
    X,y = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    X_mask = (X_padded != 0) 
    y_mask = (y_padded != 0)
    return X_padded, y_padded, X_mask, y_mask


# In[22]:


english_temp, french_temp = english_sequences[:50000].reset_index(drop=True), french_sequences[:50000].reset_index(drop=True)


# In[23]:


dataset = SentencesDataset(english_temp,french_temp)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,collate_fn = collate_fn)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,collate_fn = collate_fn)


# In[24]:


EMBEDDING_DIM = 512
HIDDEN_DIM = 128
ENCODER_LAYERS = 4
DECODER_LAYERS = 4
DROPOUT = 0.5
SRC_VOCAB_SIZE = english_vocab_size  
PAD_IDX = 0 
TRG_VOCAB_SIZE = french_vocab_size  
MAX_ENGLISH_LEN = max(english_sequences.apply(len))
MAX_FRENCH_LEN = max(french_sequences.apply(len))


# In[25]:


model = Transformers( embed_dim = EMBEDDING_DIM, src_vocab_size = SRC_VOCAB_SIZE, src_max_length = MAX_ENGLISH_LEN, trg_vocab_size = TRG_VOCAB_SIZE, trg_max_length = MAX_FRENCH_LEN, heads = 4, encoder_layers = ENCODER_LAYERS, decoder_layers = DECODER_LAYERS)
if os.path.exists("transformers_model.pth"):
    model.load_state_dict(torch.load("transformers_model.pth")) 
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


# In[26]:


def train(model :nn.Module, criterion: nn.Module, optimizer: torch.optim, train_data: DataLoader, epochs: int = 4):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X,y, X_mask, y_mask in train_data:
            optimizer.zero_grad()
            outputs = model(X, X_mask, y, y_mask)
            outputs = outputs[:,:-1,:]
            y = y[:,1:]
            loss = criterion(outputs.reshape(-1,TRG_VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            torch.save(model.state_dict(), "transformers_model.pth")
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_data):.4f}")


# In[27]:


train(model, criterion, optimizer, train_loader, 10)


# In[28]:


def val(model :nn.Module, criterion: nn.Module, optimizer: torch.optim, val_data: DataLoader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X,y, X_mask, y_mask in val_data:
            outputs = model(X, X_mask, y, y_mask)
            outputs = outputs[:,1:,:]
            y = y[:,1:]
            loss = criterion(outputs.reshape(-1,TRG_VOCAB_SIZE), y.reshape(-1))
            val_loss += loss.item()
        print(f"Loss: {val_loss / len(val_data):.4f}")


# In[29]:


val(model, criterion, optimizer, val_loader)


# In[106]:


def generate(model :nn.Module, criterion: nn.Module, optimizer: torch.optim, input: torch.tensor):
    output_tokens = []
    encoder_outputs = model.encoder(input,None)
    current_token = torch.tensor([[0]]) #SOS token
    i = 0
    while True:
        prediction = model.decoder(current_token, encoder_outputs, None, None)
        pred_probs = model.linear(prediction)
        logits = pred_probs[:, -1, :]
        next_token = logits.argmax(-1)
        if next_token.item() == 2: #EOS token
                break
        output_tokens.append(next_token.item())
        current_token = torch.cat([current_token, torch.tensor([[next_token]])], dim=1)
        
    return output_tokens


# In[107]:


import torch.nn.functional as F
def generate_topk(model :nn.Module, criterion: nn.Module, optimizer: torch.optim, input: torch.tensor, k: int =5):
    output_tokens = []
    encoder_outputs = model.encoder(input,None)
    current_token = torch.tensor([[0]]) #SOS token
    i = 0
    while True:
        prediction = model.decoder(current_token, encoder_outputs, None, None)
        pred_probs = model.linear(prediction)
        logits = pred_probs[:, -1, :].squeeze(0)
        top_k_logits, top_k_indices = torch.topk(logits, k)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        next_token = top_k_indices[torch.multinomial(top_k_probs, 1).item()]
        if next_token.item() == 2: #EOS token
                break
        output_tokens.append(next_token.item())
        current_token = torch.cat([current_token, torch.tensor([[next_token]])], dim=1)
        
    return output_tokens


# In[ ]:





# In[108]:


sentence = "are you stupid?"
text = preprocess_text(sentence)


# In[109]:


input = torch.tensor(tokenize_text(text, english_token_to_id)).unsqueeze(0)


# In[110]:


# input = torch.tensor(english_sequences[1000]).unsqueeze(0)


# In[111]:


translation = generate(model, criterion, optimizer, input)


# In[112]:


[french_id_to_token[idx] for idx in translation]


# In[120]:


translation = generate_topk(model, criterion, optimizer, input, k =2)


# In[121]:


[french_id_to_token[idx] for idx in translation]


# In[113]:


[english_id_to_token[idx] for idx in english_sequences[1000]]


# In[ ]:




