import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import spacy

import random
import math
import time

from torchtext.datasets import TranslationDataset, Multi30k #WMT14, IWSLT
from torchtext.data import Field, BucketIterator

import torch.nn.functional as F

seed = 43

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# python -m spacy download en
# python -m spacy download de

# nlp = spacy.load('de_core_news_sm')
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

# немецкий язык является полем SRC, а английский в поле TRG
SRC = Field(tokenize = tokenize_de, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

# В датасете содержится ~ 30к предложений средняя длина которых 11
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),  fields = (SRC, TRG))

labels = ['train', 'validation', 'test']
dataloaders = [train_data, valid_data, test_data]
for d, l in zip(dataloaders, labels):
    print("Number of sentences in {} : {}".format(l, len(d.examples)))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
print("Number of words in source vocabulary", len(SRC.vocab))
print("Number of words in target vocabulary", len(TRG.vocab))

#============================================================= Encoder

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        :param: input_dim is the size/dimensionality of the one-hot vectors that will be input to the encoder. This is equal to the input (source) vocabulary size.
        :param: emb_dim is the dimensionality of the embedding layer. This layer converts the one-hot vectors into dense vectors with emb_dim dimensions.
        :param: hid_dim is the dimensionality of the hidden and cell states.
        :param: n_layers is the number of layers in the RNN.
        :param: percentage of the dropout to use
        
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.input_dim, self.emb_dim)
 
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        :param: src sentences (src_len x batch_size)
        """
        # embedded = <TODO> (src_len x batch_size x embd_dim)
        embedded = self.embedding(src)
        # dropout over embedding
        embedded = self.dropout(embedded)
        outputs, (hidden, cell) = self.rnn(embedded)
        # [Attention return is for lstm, but you can also use gru]
        return outputs, hidden, cell
#===========================================================Decoder

class Attention(nn.Module):
    def __init__(self, batch_size, hidden_size=None, method="dot"): # add parameters needed for your type of attention
        super(Attention, self).__init__()
        self.method = method # attention method you'll use. e.g. "cat", "one-layer-net", "dot", ...
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        
        if self.method == 'general':
          self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
          self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
          self.v = nn.Parameter(torch.FloatTensor(hidden_size)) # Назначаем обучаемый параметр
        
    def dot_score(self, rnn_output, encoder_outputs):
      if self.method == 'dot':
        return torch.sum(rnn_output * encoder_outputs, dim=2)

      elif self.method == 'general':
        energy = self.attn(encoder_outputs)
        return torch.sum(rnn_output * energy, dim=2)

      elif self.method == 'concat':
        # конкатенируем rnn_output и encoder_outputs, применяем линейную трансформацию и регуляризацию
        energy = (self.attn(torch.cat((rnn_output.expand(encoder_output.size(0), -1, -1), encoder_outputs), 2))).tanh()
        
        return torch.sum(self.v * energy, dim=2)

    def forward(self, rnn_output, encoder_outputs, seq_len=None):

      score = self.dot_score(rnn_output, encoder_outputs)

      # Применяем softmax для нормализации и дополняем размерностью для возможности перемножения
      attn_weights = F.softmax(score, dim=0).unsqueeze(1)

      context_vector = torch.bmm(attn_weights.transpose(0,2), encoder_outputs.transpose(0,1))

      return context_vector


class DecoderAttn(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, attention, dropout=0.1):
        super(DecoderAttn, self).__init__()
        
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.attn = attention # instance of Attention class

        # define layers
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        
        self.rnn = nn.LSTM(self.emb_dim, self.hid_dim, self.n_layers) #(lstm embd, hid, layers, dropout?)
        self.out = nn.Linear(self.hid_dim, self.output_dim) # Projection :hid_dim x output_dim
        self.dropout = nn.Dropout(dropout)

        # more layers you'll need for attention
        self.concat = nn.Linear(self.hid_dim * 2, self.hid_dim)
        
    def forward(self, input_, last_hidden, cell, encoder_output):
        # make decoder with attention
        # use code from seminar notebook as base and add attention to it
        input_ = input_.unsqueeze(0)

        # (1 x batch_size x emb_dim)
        embedded = self.embedding(input_) #1 embd over input and dropout
        embedded = self.dropout(embedded)

        # Вычмляем new_hidden
        rnn_output, (new_hidden, cell) = self.rnn(embedded, (last_hidden, cell))

        # Вычисляем вектор контекста (функция forward в Attention)
        context_vector = self.attn(rnn_output, encoder_output)

        # Размеры rnn_output и context_vector -> 1*128*512 и 128*1*512. Удаляем лишние размерности

        rnn_output = rnn_output.squeeze(0)
        context_vector = context_vector.squeeze(1)

        # Конкатенируем и увеличиваем размерность
        concat = torch.cat((context_vector, rnn_output), dim=1)
        concat_out = self.concat(concat)

        # Делаем предсказания, возвращаем результат и состояния
        prediction = self.out(concat_out)

        return prediction, new_hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#=======================================================Seq2Seq module

# we need that to put it first to the decoder in 'translate' method
BOS_IDX = SRC.vocab.stoi['<sos>']

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        # Hidden dimensions of encoder and decoder must be equal
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self._init_weights() 
        self.max_len=30
    
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        """
        :param: src (src_len x batch_size)
        :param: tgt
        :param: teacher_forcing_ration : if 0.5 then every second token is the ground truth input
        """
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        # пропускаем источник через энкодер
        enc_out, hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input_ = trg[0, :]
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(input_, hidden, cell, encoder_output=enc_out) #TODO pass state and input throw decoder 
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input_ = (trg[t] if teacher_force else top1)
        
        return outputs
    
    def translate(self, src):
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = []
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        src = torch.tensor(src).to(self.device).unsqueeze(1)
        enc_out, hidden, cell = self.encoder(src) # TODO pass src throw encoder
        
        #first input to the decoder is the <sos> tokens
        input_ = torch.tensor([SRC.vocab.stoi['<sos>']] * src.shape[1]).to(device) # TODO trg[idxs]
        
        for t in range(1, self.max_len):
            
            output, hidden, cell = self.decoder(input_, hidden, cell, enc_out) #TODO pass state and input throw decoder 
            top1 = output.max(1)[1]
            outputs.append(top1)
            input_ = (top1)
        
        return outputs
    
    def _init_weights(self):
        p = 0.08
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -p, p)
        
input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)
src_embd_dim =  tgt_embd_dim = 256
hidden_dim = 512
num_layers =  2
dropout_prob = 0.2

batch_size = 64
PAD_IDX = TRG.vocab.stoi['<pad>']

iterators = BucketIterator.splits((train_data, valid_data, test_data),
                                  batch_size = batch_size, device = device)
train_iterator, valid_iterator, test_iterator = iterators

enc = Encoder(input_dim = input_dim, emb_dim = src_embd_dim, hid_dim=hidden_dim,
              n_layers=num_layers, dropout=dropout_prob)
attn = Attention(batch_size, hidden_dim, method='concat')
dec = DecoderAttn(output_dim=output_dim, emb_dim=tgt_embd_dim, hid_dim=hidden_dim,
                  n_layers=num_layers, attention=attn, dropout=dropout_prob)
model = Seq2Seq(enc, dec, device).to(device)

print(model)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        print(batch.src.shape)
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing !!
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)


            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

max_epochs = 10
CLIP = 1

# TODO
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

best_valid_loss = float('inf')

for epoch in range(max_epochs):
    
    
    train_loss = round(train(model, train_iterator, optimizer, criterion, CLIP), 5)
    valid_loss = round(evaluate(model, valid_iterator, criterion),5)
    
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    
    print('Epoch: {} \n Train Loss {}  Val loss {}:'.format(epoch, train_loss, valid_loss))
    print('Train Perplexity {}  Val Perplexity {}:'.format(np.exp(train_loss), np.exp(valid_loss)))


model.load_state_dict(torch.load('model.pt'))
model.eval()

test_loss = evaluate(model, test_iterator, criterion)

print('| Test Loss: {} Test PPL:{}|'.format(test_loss, np.exp(test_loss)))

EOS_IDX = SRC.vocab.stoi['<eos>']

def translate(sentence):
    """
    function that uses .translate() method of the model to translate german sentence into english
    params:
        sentence: tokenized gernam sentence
    """
    sentence = sentence.lower()
    sent_vec = [SRC.vocab.stoi[token] for token in sentence.split()]
    
    translation_idx = model.translate(torch.tensor(sent_vec))

    res = []
    for t in translation_idx:
        if t[0] != EOS_IDX:
            # print(TRG.vocab.itos[t[0]], end=' ')
            res.append(TRG.vocab.itos[t[0]])
        else:
            break

    return ' '.join(res)

print(translate_b("ein klein apfel"))

