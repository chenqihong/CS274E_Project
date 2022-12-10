import torch as t
import torch.nn.functional as F
from batchloader import BatchLoader
from vae import VAE
from torch.autograd import Variable
from torch.optim import Adam
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO, filename="log.txt")
logger.addHandler(logging.StreamHandler())


def train(filename, num_iterations, n_epochs, batch_size, use_cuda,learning_rate, learning_rate_scale, dropout, aux,kld_weight, embed_size, max_len):

    encoder_sizes = [[embed_size, 128, 4, 2],
                     [128, 256, 4, 2],
                     [256, 256, 4, 2],
                     [256, 512, 4, 2],
                     [512, 512, 4, 2],
                     [512, 512, 4, 2]]

    latent_size = encoder_sizes[-1][1]

    batch_loader = BatchLoader(filename, max_seq_len=max_len - 1)

    vae = VAE(batch_loader.vocab_size, embed_size, latent_size, max_len - 1)
    if use_cuda:
        vae = vae.cuda()
    optimizer = Adam(vae.parameters(), learning_rate)

    iteration = -1
    start_epoch = 0

    for epoch in range(start_epoch, start_epoch + n_epochs):
        scheduler = t.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iterations,
                                                           learning_rate / learning_rate_scale)
        for iteration in range(iteration + 1, iteration + num_iterations):
            if kld_weight:
                kld_w = min(iteration / (num_iterations * kld_weight), 1.0)
            else:
                kld_w = 0

            '''Train step'''
            vae.train()
            input, decoder_input, target = batch_loader.next_batch(batch_size, 'train', use_cuda)

            target = target.view(-1)

            logits, aux_logits, kld = vae(dropout, input, decoder_input)

            logits = logits.view(-1, batch_loader.vocab_size)
            cross_entropy = F.cross_entropy(logits, target, reduction='sum')

            aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
            aux_cross_entropy = F.cross_entropy(aux_logits, target, reduction='sum')

            trainloss = cross_entropy + aux * aux_cross_entropy + kld_w * kld

            lstmloss = cross_entropy + kld

            optimizer.zero_grad()
            trainloss.backward()
            optimizer.step()
            scheduler.step()

            '''Validation'''
            vae.eval()
            input, decoder_input, target = batch_loader.next_batch(batch_size, 'valid', use_cuda)
            target = target.view(-1)

            logits, aux_logits, valid_kld = vae(dropout, input, decoder_input)

            logits = logits.view(-1, batch_loader.vocab_size)
            valid_cross_entropy = F.cross_entropy(logits, target, reduction='sum')

            aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
            valid_aux_cross_entropy = F.cross_entropy(aux_logits, target, reduction='sum')

            validloss = valid_cross_entropy + aux * valid_aux_cross_entropy + valid_kld

            if iteration % 50 == 0:
                logger.info("\niteration: %d", iteration)
                logger.info("trainloss: %d", trainloss)
                logger.info("validloss: %d", validloss)
                logger.info("lstmloss: %d", lstmloss)

                logger.info("train ce: %s, aux-ce:%s, kld:%s",
                            cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                            aux_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                            kld.data.cpu().numpy())
                logger.info("valid ce: %s, aux-ce:%s, kld:%s",
                            valid_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                            valid_aux_cross_entropy.data.cpu().numpy() / (max_len * batch_size),
                            valid_kld.data.cpu().numpy())

                inputv, _, _ = batch_loader.next_batch(2, 'valid', use_cuda)
                mu, logvar = vae.inference(inputv)
                mu = mu[0]
                logvar = logvar[0]
                std = t.exp(0.5 * logvar)
                z = Variable(t.randn([1, latent_size]))
                if use_cuda:
                    z = z.cuda()
                z = z * std + mu
                logger.info(''.join([batch_loader.idx_to_char[idx] for idx in inputv.data.cpu().numpy()[0]]))
                logger.info("sample: " + vae.sample(batch_loader, use_cuda, z))


train(filename="ptb.train.txt",
      num_iterations=35000,
      n_epochs=1,
      batch_size=300,
      use_cuda=True,
      learning_rate=0.0005,
      learning_rate_scale=100,
      dropout=0.12,
      aux=0.2,
      kld_weight=4,
      embed_size=32,
      max_len=210)