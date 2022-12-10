import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
from encoder import Encoder
from decoder import Decoder


class VAE(nn.Module):
    def __init__(self, vocab_size, embed_size, latent_size, max_seq_len):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.embed.weight = xavier_normal_(self.embed.weight)
        self.encoder = Encoder(self.embed_size, self.latent_size, max_seq_len)
        self.context_to_mu = nn.Linear(self.latent_size, self.latent_size)
        self.context_to_logvar = nn.Linear(self.latent_size, self.latent_size)
        self.decoder = Decoder(self.vocab_size, self.latent_size, self.embed_size,max_seq_len)

    def forward(self, drop_prob,encoder_input=None,decoder_input=None,z=None):

        if z is None:
            [batch_size, _] = encoder_input.size()
            encoder_input = self.embed(encoder_input)
            context = self.encoder(encoder_input)

            mu = self.context_to_mu(context)
            logvar = self.context_to_logvar(context)
            std = t.exp(0.5 * logvar)

            z = Variable(t.randn([batch_size, self.latent_size]))
            if encoder_input.is_cuda:
                z = z.cuda()
            z = z * std + mu
            z = F.dropout(z, drop_prob, training=True)

            kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean()
        else:
            kld = None
        decoder_input = self.embed(decoder_input)
        logits, aux_logits = self.decoder(z, decoder_input)

        return logits, aux_logits, kld

    def inference(self, input):
        context = self.encoder(self.embed(input))
        return self.context_to_mu(context), self.context_to_logvar(context)

    def sample(self, batch_loader, use_cuda, z=None):
        if z is None:
            z = Variable(t.randn([1, self.latent_size]))
            if use_cuda:
                z = z.cuda()

        final_state = None
        cnn_out = self.decoder.conv_decoder(z)

        x = np.array([[batch_loader.char_to_idx[batch_loader.go_token]]])
        x = Variable(t.from_numpy(x)).long()

        if use_cuda:
            x = x.cuda()

        x = self.embed(x)

        result = []

        for var in t.transpose(cnn_out, 0, 1)[:150]:
            out, final_state = self.decoder.rnn_decoder(var.unsqueeze(1), decoder_input=x, initial_state=final_state)
            out = F.softmax(out.squeeze(), dim=-1)
            out = out.data.cpu().numpy()
            idx = np.random.choice(len(out), p=out.ravel())

            x = batch_loader.idx_to_char[idx]

            if x == batch_loader.stop_token:
                break
            result += [x]
            x = Variable(t.from_numpy(np.array([[idx]]))).long()
            if use_cuda:
                x = x.cuda()
            x = self.embed(x)

        return ''.join(result)
