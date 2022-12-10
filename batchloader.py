import numpy as np
import torch as t
from torch.autograd import Variable


class BatchLoader:
    def __init__(self, data_path='./data/ru.txt', max_seq_len=209, vocab=None):
        self.split = 1500

        self.data_path = data_path
        self.go_token = '>'
        self.pad_token = ''
        self.stop_token = '<'

        """
        performs data preprocessing
        """

        data = open(self.data_path, 'r', encoding='utf-8').read()

        if vocab:
            self.vocab_size, self.idx_to_char, self.char_to_idx = vocab
        else:
            self.vocab_size, self.idx_to_char, self.char_to_idx = self.build_vocab(data)

        self.max_seq_len = max_seq_len
        if max_seq_len == 209:
            data = np.array([[self.char_to_idx[char] for char in line] for line in data.split('\n')[:-1]
                             if 70 <= len(line) <= self.max_seq_len], dtype=object)
        elif max_seq_len == 50:
            data = np.array([[self.char_to_idx[char] for char in line] for line in data.split('\n')[:-1]
                             if 2 <= len(line) <= self.max_seq_len])
        else:
            raise ValueError("max_seq_len must be either 50 or 209 for now")

        self.valid_data, self.train_data = data[:self.split], data[self.split:]

        self.data_len = [len(var) for var in [self.train_data, self.valid_data]]

    def build_vocab(self, data):

        # unique characters with blind symbol
        chars = list(set(data)) + [self.pad_token, self.go_token, self.stop_token]
        chars_vocab_size = len(chars)

        # mappings itself
        idx_to_char = chars
        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        return chars_vocab_size, idx_to_char, char_to_idx

    def next_batch(self, batch_size, target: str, use_cuda=False):

        target = 0 if target == 'train' else 1

        indexes = np.array(np.random.randint(self.data_len[target], size=batch_size))

        encoder_input = [np.copy([self.train_data, self.valid_data][target][idx]).tolist() for idx in indexes]

        return self._wrap_tensor(encoder_input, use_cuda)

    def _wrap_tensor(self, input, use_cuda: bool):
        """
        :param input: An list of batch size len filled with lists of input indexes
        :param use_cuda: whether to use cuda
        :return: encoder_input, decoder_input and decoder_target tensors of Long type
        """

        batch_size = len(input)

        encoder_input = [np.concatenate(([self.char_to_idx[self.go_token]], line)) for line in np.copy(input)]
        decoder_input = [np.concatenate(([self.char_to_idx[self.go_token]], line)) for line in np.copy(input)]
        decoder_target = [np.concatenate((line, [self.char_to_idx[self.stop_token]])) for line in np.copy(input)]

        to_add = [self.max_seq_len - len(input[i]) for i in range(batch_size)]

        for i in range(batch_size):
            encoder_input[i] = np.concatenate((encoder_input[i], [self.char_to_idx[self.pad_token]] * to_add[i]))
            decoder_input[i] = np.concatenate((decoder_input[i], [self.char_to_idx[self.pad_token]] * to_add[i]))
            decoder_target[i] = np.concatenate((decoder_target[i], [self.char_to_idx[self.pad_token]] * to_add[i]))

        result = [np.array(var) for var in [encoder_input, decoder_input, decoder_target]]
        result = [Variable(t.from_numpy(var)).long() for var in result]
        if use_cuda:
            result = [var.cuda() for var in result]

        return tuple(result)