from torch import transpose
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_size, latent_size, max_seq_len):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.latent_size = latent_size

        if max_seq_len == 50:
            self.layered_layers = [
                [nn.Conv1d(self.embed_size, 128, 4, 2),
                 nn.BatchNorm1d(128),
                 nn.ELU()],

                [nn.Conv1d(128, 256, 4, 2),
                 nn.BatchNorm1d(256),
                 nn.ELU()],

                [nn.Conv1d(256, 512, 4, 2),
                 nn.BatchNorm1d(512),
                 nn.ELU()],

                [nn.Conv1d(512, self.latent_size, 4, 2),
                 nn.BatchNorm1d(self.latent_size),
                 nn.ELU()]
            ]

        elif max_seq_len == 209:
            self.layered_layers = [
                [nn.Conv1d(self.embed_size, 128, 4, 2),
                 nn.BatchNorm1d(128),
                 nn.ELU()],

                [nn.Conv1d(128, 256, 4, 2),
                 nn.BatchNorm1d(256),
                 nn.ELU()],

                [nn.Conv1d(256, 256, 4, 2),
                 nn.BatchNorm1d(256),
                 nn.ELU()],

                [nn.Conv1d(256, 512, 4, 2),
                 nn.BatchNorm1d(512),
                 nn.ELU()],

                [nn.Conv1d(512, 512, 4, 2),
                 nn.BatchNorm1d(512),
                 nn.ELU()],

                [nn.Conv1d(512, self.latent_size, 4, 2),
                 nn.BatchNorm1d(self.latent_size),
                 nn.ELU()]
            ]
        else:
            raise ValueError("max_seq_len must be 50 or 209 for now")
        self.cnn = nn.Sequential(*[x for layer in self.layered_layers for x in layer])

    def forward(self, input):
        return self.cnn(transpose(input, 1, 2)).squeeze(2)