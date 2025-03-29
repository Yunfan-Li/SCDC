import torch
import torch.nn.functional as F
from torch import nn, optim
from loss import ZINBLoss
from tqdm import tqdm
from evaluate import inference, evaluate


class SCDC(nn.Module):
    def __init__(self, n_cell, n_gene, n_type, n_batch):
        super(SCDC, self).__init__()
        self.n_cell = n_cell
        self.n_type = n_type
        self.n_batch = n_batch
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_gene, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.cluster = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_type),
        )
        self.batch_encoder = nn.Sequential(
            nn.Linear(n_gene, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.batch_discriminator = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_batch),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32 + n_type, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
        )
        self.zinb_pi = nn.Sequential(
            nn.Linear(256, n_gene),
            nn.Sigmoid(),
        )
        self.zinb_disp = nn.Sequential(
            nn.Linear(256, n_gene),
            nn.Softplus(),
        )
        self.zinb_mean = nn.Linear(256, n_gene)
        self.pred_cluster_num = 1.0
        self.pseudo_labels = -torch.ones(self.n_cell, dtype=torch.long).cuda()
        self.cluster_centers = torch.ones((self.n_type, 32)).cuda()

    def gumbel_softmax(self, c):
        cur_temperature = (1 - self.pred_cluster_num / self.n_type) * (
            1.0 - 0.33) + 0.33
        return F.gumbel_softmax(c, hard=False, tau=cur_temperature)

    def encode(self, x, batch_indices):
        feature = self.feature_encoder(x)
        # gumbel-softmax
        c = self.gumbel_softmax(self.cluster(feature))
        # within-batch shuffling
        z_ = torch.zeros((x.shape[0], 32)).cuda()
        for b in range(self.n_batch):
            batch_index = (batch_indices == b)
            random_row = torch.randperm(batch_index.sum())
            z_[batch_index] = self.batch_encoder(x[batch_index][random_row])
        z = self.batch_encoder(x)
        return c, z, z_

    def decode(self, z):
        decode = self.decoder(z)
        pi = self.zinb_pi(decode)
        disp = torch.clamp(self.zinb_disp(decode), 1e-4, 1e4)
        mean = torch.clamp(torch.exp(self.zinb_mean(decode)), 1e-5, 1e6)
        return pi, disp, mean

    def encode_feature_cluster(self, x):
        feature = self.feature_encoder(x)
        c = self.cluster(feature)
        return feature, c

    def encode_batch(self, x):
        z = self.batch_discriminator(self.batch_encoder(x))
        return z

    def run(self, epochs, train_dataloader, test_dataloader, gamma):
        optimizer = optim.Adam(self.parameters())
        zinb_loss = ZINBLoss()
        ce_loss = nn.CrossEntropyLoss()
        for epoch in tqdm(range(epochs)):
            pred_type = [False for i in range(self.n_type)]
            self.train()
            zinb_loss_epoch = ce_loss_epoch = zinb_loss_shuffle_epoch = 0.
            for (x, sf, rc, t, b), index in train_dataloader:
                x_ = (x + gamma * torch.randn_like(x)).cuda()
                sf = sf.cuda()
                rc = rc.cuda()
                b = b.cuda()

                loss_zinb = loss_zinb_shuffle = loss_ce = 0
                c, z, z_ = self.encode(x_, b)
                pi, disp, mean = self.decode(torch.cat((c, z), dim=1))
                loss_zinb += zinb_loss(x=rc,
                                       mean=mean,
                                       disp=disp,
                                       pi=pi,
                                       scale_factor=sf)

                pi, disp, mean = self.decode(torch.cat((c, z_), dim=1))
                loss_zinb_shuffle += zinb_loss(x=rc,
                                               mean=mean,
                                               disp=disp,
                                               pi=pi,
                                               scale_factor=sf)

                batch_pred = self.encode_batch(x_)
                loss_ce += ce_loss(batch_pred, b)
                zinb_loss_epoch += loss_zinb.item()
                zinb_loss_shuffle_epoch += loss_zinb_shuffle.item()
                ce_loss_epoch += loss_ce.item()

                loss = loss_zinb_shuffle + loss_zinb + 0.01 * loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pred = torch.unique(torch.argmax(c, dim=1))
                    for p in pred:
                        pred_type[p] = True

            zinb_loss_epoch /= len(train_dataloader)
            ce_loss_epoch /= len(train_dataloader)
            zinb_loss_shuffle_epoch /= len(train_dataloader)
            tqdm.write(
                "Epoch [%d/%d] ZINB Loss: %.5f, ZINB(shuffle) Loss: %.5f, CE Loss: %.5f"
                % (epoch, epochs, zinb_loss_epoch, zinb_loss_shuffle_epoch,
                   ce_loss_epoch))
            self.pred_cluster_num = sum(pred_type)
            if (epoch + 1) % 10 == 0:
                feature_vec, type_vec, batch_vec, pred_vec, soft_vec = inference(
                    self, test_dataloader)
                evaluate(feature_vec, pred_vec, type_vec, batch_vec, soft_vec)