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
        self.wke = nn.Linear(d, heads*k, bias=False)
        self.wkr = nn.Linear(d, heads*k, bias=False)
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
        nn.init.normal_(self.wke.weight, 0, .02)
        nn.init.normal_(self.wkr.weight, 0, .02)
        nn.init.normal_(self.wv.weight, 0, .02)
        nn.init.normal_(self.wc.weight, 0, .02)

        nn.init.normal_(self.w1.weight, 0, .02)
        nn.init.constant_(self.w1.bias, 0.0)
        nn.init.normal_(self.w2.weight, 0, .02)
        nn.init.constant_(self.w2.bias, 0.0)

    # p = positional embedding, R_{i-j} in paper
    def forward(self, x, p, mask, memory, u_bias, v_bias):
        seq_len, batch_size, embed_dim = x.shape
        mem_len = (memory.shape[0] // seq_len) if memory is not None else 0

        # d = embed_dim
        # k = head_dim
        # memory.shape[0] = mem_length * seq_len
        
        # x.shape       : seq_len, batch_size, d
        # p.shape       : (mem_length + 1) * seq_len, batch_size, d
        # mask.shape    : seq_len, (mem_length + 1) * seq_len ???
        # memory.shape  : mem_length * seq_len, batch_size, d
        # u_bias        : heads, k
        # v_bias        : heads, k

        q = self.wq(x)  # seq_len, batch_size, heads * k
        # x_mem = torch.cat([memory, x.unsqueeze(0)], dim=0) if memory is not None else x.unsqueeze(0)  # mem_len + 1, seq, batch, emb
        x_mem = torch.cat([memory, x], dim=0) if memory is not None else x  # (mem_length + 1) * seq, batch_size, d
        k_e = self.wke(x_mem)  # (mem_length + 1) * seq, batch, heads * k
        k_r = self.wkr(p)  # (mem_length + 1) * seq, batch, heads * k

        v = self.wv(x_mem)  # (mem_length + 1) * seq, batch_size, heads * k
        q = torch.reshape(q, (seq_len, batch_size, self.heads, self.k)) # seq_len, batch_size, heads, k
        q_bias_u = q + u_bias # Assume correct broadcasting, (seq_len, batch_size, heads, k)
        q_bias_v = q + v_bias # Assume correct broadcasting, (seq_len, batch_size, heads, k)
        k_e = torch.reshape(k_e, ((mem_len + 1) * seq_len, batch_size, self.heads, self.k)) # (mem_length + 1) * seq, batch, heads, k
        k_r = torch.reshape(k_r, ((mem_len + 1) * seq_len, batch_size, self.heads, self.k)) # (mem_length + 1) * seq, batch, heads, k
        v = torch.reshape(v, ((mem_len + 1) * seq_len, batch_size, self.heads, self.k)) # (mem_length + 1) * seq, batch_size, heads, k
        q_bias_u = torch.transpose(q_bias_u, 0, 2)  # self.heads, batch_size, seq_len, self.k
        q_bias_v = torch.transpose(q_bias_v, 0, 2)  # self.heads, batch_size, seq_len, self.k
        k_e = torch.transpose(k_e, 0, 2)  # self.heads, batch_size, (mem_len + 1) * seq_len, self.k
        k_r = torch.transpose(k_r, 0, 2)  # self.heads, batch_size, (mem_len + 1) * seq_len, self.k
        v = torch.transpose(v, 0, 2)  # self.heads, batch_size, (mem_len + 1) * seq_len, self.k
        q_bias_u = torch.reshape(q_bias_u, (self.heads * batch_size, seq_len, self.k)) # self.heads * batch_size, seq_len, self.k
        q_bias_v = torch.reshape(q_bias_v, (self.heads * batch_size, seq_len, self.k)) # self.heads * batch_size, seq_len, self.k
        k_e = torch.reshape(k_e, (self.heads * batch_size, (mem_len + 1) * seq_len, self.k)) # self.heads * batch_size, (mem_len + 1) * seq_len, self.k
        k_r = torch.reshape(k_r, (self.heads * batch_size, (mem_len + 1) * seq_len, self.k)) # self.heads * batch_size, (mem_len + 1) * seq_len, self.k
        v = torch.reshape(v, (self.heads * batch_size, (mem_len + 1) * seq_len, self.k)) # self.heads * batch_size, (mem_len + 1) * seq_len, self.k

        # self.heads * batch_size, seq_len, (mem_len + 1) * seq_len
        AC = torch.bmm(q_bias_u, torch.transpose(k_e, 1, 2)) # k_e.T(1,2) -> # self.heads * batch_size, self.k, (mem_len + 1) * seq_len
        BD = torch.bmm(q_bias_v, torch.transpose(k_r, 1, 2)) 
        # if AC.shape[2] > BD.shape[2]:
        #     BD = torch.cat([torch.zeros(AC.shape[0], AC.shape[1], AC.shape[2] - BD.shape[2]).cuda(), BD], dim=2)

        # print(AC.shape, BD.shape)
        alpha = AC + BD
        alpha = alpha / math.sqrt(self.k) + mask # Assume correct masking
        s = nn.Softmax(dim=-1)
        alpha = s(alpha)  # self.heads * batch_size, seq_len, seq_len * (mem_len + 1) applied softmax
        alpha = self.dropoutatt(alpha)
        alpha_v = torch.bmm(alpha, v)  # self.heads * batch_size, seq_len, self.k
        alpha_v = torch.reshape(alpha_v, (self.heads, batch_size, seq_len, self.k))
        alpha_v = torch.transpose(alpha_v, 0, 2)  # seq_len, batch_size, self.heads, self.k
        alpha_v = torch.reshape(alpha_v, (seq_len, batch_size, self.heads * self.k))
        u = self.wc(alpha_v)  # seq_len, batch_size, d
        u = self.dropoutfc(u)
        u = self.layernorm1(u + x)
        # return u
        z = self.w2(self.dropout1(torch.relu(self.w1(u))))
        z = self.layernorm2(self.dropout2(z) + u)
        return z


# https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L15-L31
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class Transformer(nn.Module):
    def __init__(self, seq_len, tokens, d, k, m, heads, layers, tied_weights=False, dropout=0., dropoutio=0., max_mem_length=1):
        super(Transformer, self).__init__()
        self.mask = None
        self.pos = None
        self.dims = d
        self.tied_weights = tied_weights
        self.dropout=dropout
        self.layers = layers
        self.max_mem_length = max_mem_length

        self.positional_embedding = PositionalEmbedding(d)
        self.dropi = nn.Dropout(dropoutio)
        self.word_embedding = nn.Embedding(tokens, d)
        self.transformer = nn.ModuleList()
        for i in range(self.layers):
            self.transformer.append(TransformerBlock(heads, d, k, m, dropout))

        if not tied_weights: self.decoder = nn.Linear(d, tokens)
        self.dropo = nn.Dropout(dropoutio)
        self.bias = nn.Parameter(torch.ones(tokens))

        # nn.init.normal_(self.positional_embedding.weight, 0, .02)
        nn.init.normal_(self.word_embedding.weight, 0, .02)
        if not self.tied_weights: nn.init.normal_(self.decoder.weight, 0, .02)
        nn.init.constant_(self.bias, 0.0)

        self.memories = [None] * self.layers  #layers, mem_len, seq, batch, emb

        self.u = nn.Parameter(torch.Tensor(heads, k))
        self.v = nn.Parameter(torch.Tensor(heads, k))

    def forward(self, x):
        padded = False
        if x.shape[1] != 32:
            padded = True
            old_x_shape = x.shape[1]
            x = torch.cat([x, torch.zeros((x.shape[0], 32 - x.shape[1]), dtype=torch.long).cuda()], dim=1)

        # len(self.memories[0]) = mem_length * seq_len
        # x.shape[0] = seq_len
        # print(f"x.shape: {x.shape}")
        mem_len = (len(self.memories[0]) // x.shape[0]) if self.memories[0] is not None else 0
        if self.mask is None or self.mask.shape[0] != x.shape[0] or mem_len <= self.max_mem_length:
            self.mask = torch.triu(torch.ones(len(x) * (mem_len + 1), len(x)))
            self.mask.masked_fill_(self.mask == 0, float('-inf')).masked_fill_(self.mask == 1, float(0.0))
            self.mask = self.mask.transpose(0,1).to(x.device)
            self.pos = torch.arange((mem_len + 1) *  x.shape[0] - 1, -1, -1, dtype=torch.long).to(x.device)

        x = self.dropi(self.word_embedding(x) * math.sqrt(self.dims))
        p = self.dropi(self.positional_embedding(self.pos, x.shape[1]))
        # z = F.relu(x + p)

        hids = [x]  #layers, seq, batch, emb
        # add memory for transformer xl
        # if x.shape[1] == 32:
        for i in range(self.layers):
            x = self.transformer[i](x, p, self.mask, self.memories[i], self.u, self.v)  #seq, batch, emb
            hids.append(x)
        # else:
        #     for i in range(self.layers):
        #         z = self.transformer[i](z, self.mask, None)  #seq, batch, emb
        #         hids.append(z)

        x = self.dropo(x)
        # print(x.shape)

        # if this is not the last batch (size = 10)
        # if x.shape[1] == 32:
        for i in range(self.layers):
            # self.memories[i] = torch.cat([self.memories[i], hids[i].detach().unsqueeze(0)], dim=0) if self.memories[i] is not None \
            #                                                                         else hids[i].detach().unsqueeze(0)
            self.memories[i] = torch.cat([self.memories[i], hids[i].detach()], dim=0) if self.memories[i] is not None \
                                                                                    else hids[i].detach()

            self.memories[i] = self.memories[i][-(self.max_mem_length * x.shape[0]):] # x.shape[0] = seq_len

        outputs = torch.matmul(x, self.word_embedding.weight.t()) if self.tied_weights else self.decoder(x)
        outputs = F.log_softmax(outputs + self.bias, dim=-1)
        if padded:
            outputs = outputs[:, :old_x_shape, :]
        return outputs

    def reset_memory(self):
        self.memories = [None] * self.layers
