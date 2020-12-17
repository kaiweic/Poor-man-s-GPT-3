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

        # so now, self.raw is a tokenized version of the .txt file.
        # now, we need to make sure to return batches in the style of dataloader.py
        # which basically means we want consecutive batches to contain consecutive sequences
        # as opposed to a batch containing consecutive sequences
        if len(self.raw) % batch_size != 0:
            self.raw = self.raw[0:-(len(self.raw) % batch_size)]

        dataset = []
        index = 0
        batch_length = len(self.raw) // batch_size
        seq_count = 0
        for batch in range(0, batch_length // context + 1):
            dataset.append([[], []])
            data = dataset[-1][0]
            label = dataset[-1][1]
            for sequence in range(context):
                for batch_index in range(batch_size):
                    if sequence % context == 0:
                        seq_count += 1
                        data.append([])
                        label.append([])
                    if batch * context + sequence < batch_length - 1:
                        data[batch_index].append(self.raw[batch * context + batch_index * batch_length + sequence])
                        label[batch_index].append(self.raw[batch * context + batch_index * batch_length + sequence + 1])
        self.dataset = dataset
        self.seq_count = seq_count


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

