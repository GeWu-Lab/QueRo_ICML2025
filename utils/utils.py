import torch
import torch.nn as nn
import numpy as np
import random
import os
import csv

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # if m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def analyse_attn_matrix(matrix, av_dim):
    # print(matrix.shape)
    avg_AA = torch.mean(matrix[1:av_dim, 1:av_dim])
    avg_VA = torch.mean(matrix[av_dim:, 1:av_dim])
    avg_VV = torch.mean(matrix[av_dim:, av_dim:])
    avg_AV = torch.mean(matrix[1:av_dim, av_dim:])

    sum_AA = torch.sum(matrix[1:av_dim, 1:av_dim])
    sum_AV = torch.sum(matrix[1:av_dim, av_dim:])
    norm = sum_AA + sum_AV
    sum_AA = sum_AA / norm
    sum_AV = sum_AV / norm

    sum_VV = torch.sum(matrix[av_dim:, av_dim:])
    sum_VA = torch.sum(matrix[av_dim:, 1:av_dim])
    norm = sum_VV + sum_VA
    sum_VV = sum_VV / norm
    sum_VA = sum_VA / norm

    score_per_token = torch.sum(matrix, dim=0)
    # print(score_per_token)

    # CLS weight of each modality
    cls_weight_a = matrix[0, 1:av_dim]
    cls_weight_v = matrix[0, av_dim:]

    return avg_AA, avg_AV, avg_VV, avg_VA, score_per_token, sum_AA, sum_VA, sum_VV, sum_AV, cls_weight_a, cls_weight_v


def get_QKV_Feature(qkv, av_dim):
    q = qkv['q']
    k = qkv['k']

    q_cls = q[:, 0, :]
    k_a = k[:, 1:av_dim, :]
    k_v = k[:, av_dim:, :]

    return q_cls, k_a, k_v


def get_logger(path):
    train_logger = Logger(os.path.join(path, 'train.log'),
                            ['epoch', 'loss', 'lr'])

    val_logger = Logger(os.path.join(path, 'val.log'),
                ['epoch', 'loss', 'acc'])
    return train_logger, val_logger


class Logger(object):
    def __init__(self, path, header):
        # self.log_file = path.open('w')
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def write_to_batch_logger(batch_logger, epoch, i, data_loader, losses, accuracies, current_lr):
    if batch_logger is not None:
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr,
        })


def write_to_epoch_logger(epoch_logger, epoch, losses, accuracies, current_lr):
    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss': losses,
            'acc': accuracies,
            'lr': current_lr
        })


def load_unimodal_encoder(model, ckpt, modality):
    loaded_dict = torch.load(ckpt)
    state_dict = loaded_dict['model']

    new_dict = {}
    for k, v in state_dict.items():
        if modality == 'audio':
            new_dict['audio_net.' + k] = v
        elif modality == 'visual':
            new_dict['visual_net.' + k] = v
        else:
            raise UserWarning("Unknown modality")

    missing_keys, unexpected_keys = model.load_state_dict(new_dict, strict=False)
    
    print(f'loading {modality}, we got')
    print('Missing keys:')
    print(missing_keys)
    print('Unexpected keys:')
    print(unexpected_keys)


def compute_rotation_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    assert X.shape == Y.shape
    """
    Compute the exact rotation matrix W such that Y = WX using SVD.
    Args:
        X (torch.Tensor): Input vector of shape (d,)
        Y (torch.Tensor): Target vector of shape (d,)
    Pre-cond:
        norm(X) == norm(Y)
    Returns:
        W (torch.Tensor): Rotation matrix of shape (d, d)
    """
    # Ensure vectors are normalized
    X = X / X.norm(p=2)
    Y = Y / Y.norm(p=2)

    # Compute the outer product matrix
    M = torch.ger(Y, X)  # Equivalent to Y @ X^T for 1D vectors

    # Perform SVD on M
    U, S, Vt = torch.linalg.svd(M)

    # Construct the rotation matrix
    W = U @ Vt

    # Ensure det(W) = 1 (rotation matrix condition, not reflection)
    if torch.det(W) < 0:
        # Adjust to ensure positive determinant
        U[:, -1] *= -1  # Flip the sign of the last column of U
        W = U @ Vt
    
    # print('Origin is: ', X)
    # print('Target is: ', Y)
    # print("Error:", (Y.double() - W.double() @ X.double()).norm(p=2))                 # Should be close to zero
    # print("Error:", torch.mean(torch.abs( (Y.double() - W.double() @ X.double()) )))  # Should be close to zero
    return W
