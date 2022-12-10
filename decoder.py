import torch as t
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, vocab_size, latent_variable_size, embed_size, max_seq_len):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.latent_variable_size = latent_variable_size
        self.embed_size = embed_size
        self.rnn_size = 1000

        if max_seq_len == 50:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, 512, 4, 2, 0),
                nn.BatchNorm1d(512),
                nn.ELU(),


                nn.ConvTranspose1d(512, 256, 4, 2, 0, output_padding=1),
                nn.BatchNorm1d(256),
                nn.ELU(),


                nn.ConvTranspose1d(256, 128, 4, 2, 0),
                nn.BatchNorm1d(128),
                nn.ELU(),

                nn.ConvTranspose1d(128, self.vocab_size, 4, 2, 0, output_padding=1)
            )
        elif max_seq_len == 209:
            self.cnn = nn.Sequential(
                nn.ConvTranspose1d(self.latent_variable_size, 512, 4, 2, 0),
                nn.BatchNorm1d(512),
                nn.ELU(),

                nn.ConvTranspose1d(512, 512, 4, 2, 0, output_padding=1),
                nn.BatchNorm1d(512),
                nn.ELU(),

                nn.ConvTranspose1d(512, 256, 4, 2, 0),
                nn.BatchNorm1d(256),
                nn.ELU(),

                nn.ConvTranspose1d(256, 256, 4, 2, 0, output_padding=1),
                nn.BatchNorm1d(256),
                nn.ELU(),

                nn.ConvTranspose1d(256, 128, 4, 2, 0),
                nn.BatchNorm1d(128),
                nn.ELU(),

                nn.ConvTranspose1d(128, self.vocab_size, 4, 2, 0)
            )
        else:
            raise ValueError("max_seq_len must be either 50 or 209 for now")

        self.rnn = nn.GRU(input_size=self.vocab_size + self.embed_size,
                          hidden_size=self.rnn_size,
                          num_layers=1,
                          batch_first=True)

        self.hidden_to_vocab = nn.Linear(self.rnn_size, self.vocab_size)

    def forward(self, latent_variable, decoder_input):
        aux_logits = self.conv_decoder(latent_variable)

        logits, _ = self.rnn_decoder(aux_logits, decoder_input, initial_state=None)

        return logits, aux_logits

    def conv_decoder(self, latent_variable):
        return t.transpose(self.cnn(latent_variable.unsqueeze(2)), 1, 2).contiguous()

    def rnn_decoder(self, cnn_out, decoder_input, initial_state=None):
        logits, final_state = self.rnn(t.cat([cnn_out, decoder_input], 2), initial_state)

        [batch_size, seq_len, _] = logits.size()

        logits = self.hidden_to_vocab(logits.contiguous().view(-1, self.rnn_size)).view(batch_size, seq_len, self.vocab_size)

        return logits, final_state
