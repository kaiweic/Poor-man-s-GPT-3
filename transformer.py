import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, heads, d, k, m, dropout=0.):
        super(TransformerBlock, self).__init__()
        self.k = k
        self.heads = heads

        self.wq = nn.Linear(d, heads*k, bias=False)
        self.wk = nn.Linear(d, heads*k, bias=False)
        self.wv = nn.Linear(d, heads*k, bias=False)
        self.wc = nn.Linear(heads*k, d, bias=False)
        self.dropoutatt = nn.Dropout(dropout)

        self.w1 = nn.Linear(d, m)
        self.dropoutfc = nn.Dropout(dropout)
        self.w2 = nn.Linear(m, d)

        self.layernorm1 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d)
        self.dropout2 = nn.Dropout(dropout)

        nn.init.normal_(self.wq.weight, 0, .02)
        nn.init.normal_(self.wk.weight, 0, .02)
        nn.init.normal_(self.wv.weight, 0, .02)
        nn.init.normal_(self.wc.weight, 0, .02)

        nn.init.normal_(self.w1.weight, 0, .02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, .02)
        nn.init.constant_(self.w2.bias, 0.0)

    def forward(self, x, mask):
        seq_len, batch_size, embed_dim = x.shape

        # Hint: Writing efficient code is almost as important as writing correct code in ML.
        #       Avoid writing for-loops! Consider using the batch matrix multiplication operator torch.bmm

        q = self.wq(x)  # seq_len, batch_size, heads * k
        k = self.wk(x)  # seq_len, batch_size, heads * k
        v = self.wv(x)  # seq_len, batch_size, heads * k
        q = torch.reshape(q, (seq_len, batch_size, self.heads, self.k))
        k = torch.reshape(k, (seq_len, batch_size, self.heads, self.k))
        v = torch.reshape(v, (seq_len, batch_size, self.heads, self.k))
        q = torch.transpose(q, 0, 2)  # self.heads, batch_size, seq_len, self.k
        k = torch.transpose(k, 0, 2)  # self.heads, batch_size, seq_len, self.k
        v = torch.transpose(v, 0, 2)  # self.heads, batch_size, seq_len, self.k
        q = torch.reshape(q, (self.heads * batch_size, seq_len, self.k))
        k = torch.reshape(k, (self.heads * batch_size, seq_len, self.k))
        v = torch.reshape(v, (self.heads * batch_size, seq_len, self.k))
        alpha = torch.bmm(q, torch.transpose(k, 1, 2))  # self.heads * batch_size, seq_len, seq_len
        alpha = alpha / math.sqrt(self.k) + mask
        s = nn.Softmax(dim=-1)
        alpha = s(alpha)  # self.heads * batch_size, seq_len, seq_len(softmax)
        alpha = self.dropoutatt(alpha)
        alpha_v = torch.bmm(alpha, v)  # self.heads * batch_size, seq_len, self.k
        alpha_v = torch.reshape(alpha_v, (self.heads, batch_size, seq_len, self.k))
        alpha_v = torch.transpose(alpha_v, 0, 2)  # seq_len, batch_size, self.heads, self.k
        alpha_v = torch.reshape(alpha_v, (seq_len, batch_size, self.heads * self.k))
        u = self.wc(alpha_v)  # seq_len, batch_size, d
        u = self.dropoutfc(u)
        u = self.layernorm1(u + x)
        z = self.w2(self.dropout1(torch.relu(self.w1(u))))
        z = self.layernorm2(self.dropout2(z) + u)
        return z

class Transformer(nn.Module):
    def __init__(self, seq_len, tokens, d, k, m, heads, layers, tied_weights=False, dropout=0., dropoutio=0.):
        super(Transformer, self).__init__()
        self.mask = None
        self.pos = None
        self.dims = d
        self.tied_weights = tied_weights
        self.dropout=dropout

        self.positional_embedding = nn.Embedding(seq_len, d)
        self.dropi = nn.Dropout(dropoutio)
        self.word_embedding = nn.Embedding(tokens, d)
        self.transformer = nn.ModuleList()
        for i in range(layers):
            self.transformer.append(TransformerBlock(heads, d, k, m, dropout))

        if not tied_weights: self.decoder = nn.Linear(d, tokens)
        self.dropo = nn.Dropout(dropoutio)
        self.bias = nn.Parameter(torch.ones(tokens))

        nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.decoder.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        if self.mask is None or self.mask.shape[0] != x.shape[0]:
            self.mask = torch.triu(torch.ones(len(x), len(x)))
            self.mask.masked_fill_(self.mask == 0, float('-inf')).masked_fill_(self.mask == 1, float(0.0))
            self.mask = self.mask.transpose(0,1).to(x.device)
            self.pos = torch.arange(0, x.shape[0], dtype=torch.long).to(x.device)

        x = self.word_embedding(x) * math.sqrt(self.dims)
        p = self.positional_embedding(self.pos)[:,None,:]
        z = F.relu(self.dropi(x) + self.dropi(p))
        for layer in self.transformer:
            z = layer(z, self.mask)

        z = self.dropo(z)
        outputs = torch.matmul(z, self.word_embedding.weight.t()) if self.tied_weights else self.decoder(z)
        return F.log_softmax(outputs + self.bias, dim=-1)