import os
from collections import defaultdict

# pytorch-lightning
import paddle
from paddle.io import DataLoader

from datasets import dataset_dict
# losses
from losses import loss_dict

from models.nerf import Embedding, NeRF
from models.rendering import render_rays
from opt import get_opts
# optimizer, scheduler, visualization
from utils import *


# metrics


class NeRFSystem(paddle.nn.Layer):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10)  # 10 is the default number
        self.embedding_dir = Embedding(3, 4)  # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays']  # (B, 8)
        rgbs = batch['rgbs']  # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i + self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = paddle.concat(v, 0)
        return results


hparams = get_opts()
system = NeRFSystem(hparams)
checkpoint_callback = paddle.callbacks.ModelCheckpoint(save_dir=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                             '{epoch:d}'))


def prepare_data():
    dataset = dataset_dict['llff']
    kwargs = {'root_dir': hparams.root_dir,
              'img_wh': tuple(hparams.img_wh)}
    if hparams.dataset_name == 'llff':
        kwargs['spheric_poses'] = hparams.spheric_poses
        kwargs['val_num'] = hparams.num_gpus
    train_dataset = dataset(split='train', **kwargs)
    val_dataset = dataset(split='val', **kwargs)
    return train_dataset, val_dataset


nerf_coarse = NeRF()
models = [nerf_coarse]
if hparams.N_importance > 0:
    nerf_fine = NeRF()
    models += [nerf_fine]
print(models)


def configure_optimizers():
    eps = 1e-8
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    optimizer = Adam(parameters=parameters, epsilon=eps,
                     weight_decay=hparams.weight_decay)
    scheduler = get_scheduler(hparams, optimizer)

    return [optimizer], [scheduler]


optimizer = get_optimizer(hparams, models)
scheduler = get_scheduler(hparams, optimizer)

train_dataset, _ = prepare_data()


def train_dataloader():
    return DataLoader(train_dataset,
                      shuffle=True,
                      num_workers=4,
                      batch_size=hparams.batch_size,
                      pin_memory=True)


def decode_batch(self, batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs


dataset, _ = prepare_data()
loader = train_dataloader()


def training_step(batch, batch_nb):
    log = {'lr': get_learning_rate(optimizer)}
    rays, rgbs = decode_batch(batch)
    results = system(rays)
    loss = loss_dict[hparams.loss_type]()
    log['train/loss'] = loss = loss(results, rgbs)
    typ = 'fine' if 'rgb_fine' in results else 'coarse'

    return loss


def train():
    epochs = 2

    for epoch in range(epochs):
        for data in enumerate(train_dataloader()):
            loss = training_step(data)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()


# 启动训练
train()
