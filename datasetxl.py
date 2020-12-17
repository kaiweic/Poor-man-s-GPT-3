import os,enum

import torch

DatasetSplit = enum.Enum('DatasetSplit', 'train valid test')

class WikiText2(torch.utils.data.Dataset):
    """
        PyTorch Dataset for the WikiText2 corpus:
            https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
    """

    def __init__(self, root, context, batch_size, split, block=True):
        self.context = context # sequence length
        self.batch_size = batch_size
        self.block = block
        self.word2idx = {}
        self.idx2word = []

        # build the vocabulary from the training data
        self._tokenize(os.path.join(root, 'train.txt')) # TODO: do we need this call?
        self.raw = self._tokenize(os.path.join(root, split.name + '.txt'))

        # trim
        seq_len = context
        num_tokens_in_batch = seq_len * batch_size # number of tokens in batch

        if len(self.raw) % num_tokens_in_batch != 0:
            self.raw = self.raw[:-(len(self.raw) % num_tokens_in_batch)]

        # le hack: y should be 1 token shifted over from x, so voila!
        # print(self.raw[-1])
        # print(self.raw.shape)
        self.raw = torch.cat([self.raw, self.raw[-1].unsqueeze(0)])
        # print(self.raw.shape)

        dataset = []
        num_batches = len(self.raw) // num_tokens_in_batch
        for batch in range(num_batches):
            dataset.append([[], []])
            x = dataset[-1][0]
            y = dataset[-1][1]
            for batch_index in range(batch_size):
                start = batch * seq_len + batch_index * num_tokens_in_batch
                end = start + seq_len
                x.append(self.raw[start:end])
                y.append(self.raw[start+1:end+1])
                assert(len(x[-1]) == seq_len)
                assert(len(y[-1]) == seq_len)

        self.dataset = dataset
        self.seq_count = num_batches * batch_size


    def __len__(self):
        if self.block:
            # return len(self.raw) // self.context
            return self.seq_count
        else:
            return len(self.raw)

    def __getitem__(self, idx):
        if self.block:
            # x = self.raw[idx*self.context:(idx+1)*self.context]
            # y = self.raw[idx*self.context+1:(idx+1)*self.context+1].view(-1)
            x = torch.LongTensor(self.dataset[idx // self.batch_size][0][idx % self.batch_size])
            y = torch.LongTensor(self.dataset[idx // self.batch_size][1][idx % self.batch_size])
        else:   # assumed not called cause block=true by default 
            x = torch.tensor([self.word2idx['<pad>']] * self.context)
            y = torch.tensor([self.word2idx['<pad>']] * self.context)
            context = min(self.context,idx)
            if idx > 0: x[-context:] = self.raw[idx-context:idx]
            context = min(self.context,idx+1)
            y[-context:] = self.raw[idx+1-context:idx+1]

        return x, y
        
    def word_count(self):
        # don't count <pad> as a word
        return len(self.idx2word)

    def _add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def _tokenize(self, path):
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    self._add_word(word)
                    ids.append(self.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))

        self._add_word('<pad>')
        return torch.cat(idss)

