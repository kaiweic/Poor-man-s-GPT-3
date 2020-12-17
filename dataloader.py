class HarryPotterDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sequence_length, batch_size):
        super(HarryPotterDataset, self).__init__()

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.vocab = Vocabulary(data_file)

        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)

        # TODO: Any preprocessing on the data to get it to the right shape.
        raw = dataset['tokens']

        print('size of entire txt', len(raw))

        # trim away uneven data
        if len(raw) % batch_size != 0:
            raw = raw[0:-(len(raw) % batch_size)]
        # assign data to the correct shape [batch1[data, labels], batch2[data, labels], etc.]
        dataset = []
        index = 0
        batch_length = len(raw) // batch_size
        seq_count = 0
        for batch in range(0, batch_length // sequence_length + 1):
            dataset.append([[], []])
            data = dataset[-1][0]
            label = dataset[-1][1]
            for sequence in range(sequence_length):
                for batch_index in range(batch_size):
                    if sequence % sequence_length == 0:
                        seq_count += 1
                        data.append([])
                        label.append([])
                    if batch * sequence_length + sequence < batch_length - 1:
                        data[batch_index].append(raw[batch * sequence_length + batch_index * batch_length + sequence])
                        label[batch_index].append(
                            raw[batch * sequence_length + batch_index * batch_length + sequence + 1])
        self.dataset = dataset
        self.seq_count = seq_count

    def __len__(self):
        # TODO return the number of unique sequences you have, not the number of characters.
        return self.seq_count

    def __getitem__(self, idx):
        # Return the data and label for a character sequence as described above.
        # The data and labels should be torch long tensors.
        # You should return a single entry for the batch using the idx to decide which chunk you are
        # in and how far down in the chunk you are.
        return torch.LongTensor(self.dataset[idx // self.batch_size][0][idx % self.batch_size]), \
               torch.LongTensor(self.dataset[idx // self.batch_size][1][idx % self.batch_size])

    def vocab_size(self):
        return len(self.vocab)