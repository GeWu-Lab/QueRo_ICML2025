import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time
import json

from utils.utils import *
from utils.args import get_arguments
from transformers import get_cosine_schedule_with_warmup

from train_func import train
from validation import valid

from dataset.CramedDataset import *
from dataset.KSDataset import *

from models.mm_transformer import *


def prepare_exp(args):
    # setup exp
    experiment = '{}_{}_{}_{}'.format(args.fusion_method, args.modulation, args.backbone,
                                        time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
    sub_root = '{}'.format(args.dataset)
    print("NAME OF EXPERIMENT: {}_{}".format(sub_root, experiment))

    exp_dset_root = os.path.join('exp', sub_root)
    if not os.path.exists(exp_dset_root):
        os.makedirs(exp_dset_root)

    exp_root = os.path.join(exp_dset_root, experiment)
    args.exp_root = exp_root
    if not os.path.exists(exp_root):
        os.makedirs(exp_root)
    
    ckpt_path = os.path.join(exp_root, 'ckpt')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    with open(os.path.join(exp_root, 'config.json'), 'w') as file:
        file.write(json.dumps({**vars(args)}, separators=(',\n', ':')))

    return exp_root, ckpt_path


def get_loader(args):
    if args.dataset == 'KineticSound':
        train_dataset = KS_dataset(args, mode='train')
        test_dataset = KS_dataset(args, mode='test')
        valid_datset =  KS_dataset(args, mode='valid')
        n_classes = 31
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
        valid_datset =  CramedDataset(args, mode='valid')
        n_classes = 6
    
    args.n_classes = n_classes
    # print(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)
    
    valid_dataloader = DataLoader(valid_datset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True)

    return n_classes, train_dataloader, test_dataloader, valid_dataloader


def build_model(args, n_classes, gpu_ids, device):
    if args.fusion_method in ['attention']:
        model = MMTransformerAttn(args=args, n_classes=n_classes)
        print('Use MMTransformerAttn')
    else:
        model = MMTransformerBase(args=args, n_classes=n_classes)
        print('Use MMTransformerBase')
    
    model.apply(weight_init)

    # load pretrain here, after overall init
    loaded_dict = torch.load('pretrain/vit_pretrain.pth')
    
    keys_to_remove = [key for key in loaded_dict.keys() if key.startswith("bot")]
    for key in keys_to_remove:
        del loaded_dict[key]

    missing_keys, unexpected_keys = model.encoder.load_state_dict(loaded_dict, strict=False)
    print('ckpt loaded!')
    print('missing keys: ', missing_keys)
    print('unexpected keys: ', unexpected_keys)

    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    
    print(gpu_ids)

    model.cuda()

    return model


def main():
    args = get_arguments()
    print(args)
    
    # mkdir
    exp_root, ckpt_path = prepare_exp(args=args)
    
    # device & cuda
    print(torch.__version__)
    print(torch.cuda.is_available())
    setup_seed(args.random_seed)
    print(args.gpu_ids)
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0')

    # loader
    n_classes, train_dataloader, test_dataloader, valid_dataloader = get_loader(args=args)

    # model
    model = build_model(args=args, n_classes=n_classes, gpu_ids=gpu_ids, device=device)

    print(f'Use {args.optimizer} optimizer')
    # optimizer & scheduler
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(len(train_dataloader)*2.5), len(train_dataloader) * args.epochs)

    # logger
    train_logger, val_logger = get_logger(exp_root)

    # start train
    print("CORE TRAIN ARGS:")
    print(f'lr: {args.learning_rate}')
    print(f'bs: {args.batch_size}')
    print(f'epoch: {args.epochs}')

    best_acc = 0.0

    epochs = args.epochs

    for epoch in range(epochs):
        print('Epoch: {}: '.format(epoch))
        writer_path = os.path.join(exp_root, 'log')
        writer = SummaryWriter(writer_path)
        
        batch_loss, batch_loss_a, batch_loss_v = train(args, epoch, model, device, train_dataloader, optimizer, scheduler, writer, train_logger)
        acc, acc_a, acc_v = valid(args, model, device, test_dataloader, epoch, val_logger)
        writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                    'Audio Loss': batch_loss_a,
                                    'Visual Loss': batch_loss_v}, epoch)
        writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                            'Audio Accuracy': acc_a,
                                            'Visual Accuracy': acc_v}, epoch)

        print("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))
        print("Audio Acc: {:.3f}, Visual Acc: {:.3f} ".format(acc_a, acc_v))
        print("Previous Best Acc: {:.3f}".format(best_acc))
        if acc > best_acc:
            best_acc = float(acc)
            model_name = 'best.pth'
            best_log = os.path.join(ckpt_path, 'best.txt')
            with open(best_log, 'w') as file:
                file.write('epoch {}, acc {:.3f}'.format(epoch, best_acc))
            
            saved_dict = {
                        'saved_epoch': epoch,
                        'modulation': args.modulation,
                        'fusion': args.fusion_method,
                        'acc': acc,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        }
            
            save_dir = os.path.join(ckpt_path, model_name)

            torch.save(saved_dict, save_dir)
            print('The best model has been saved at {}.'.format(save_dir))
        

if __name__ == "__main__":
    main()