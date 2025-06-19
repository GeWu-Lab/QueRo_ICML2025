import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import *
from method import set_rotary_matrix


def get_data(data_dict):
    spec = data_dict['a']
    image = data_dict['v']
    label = data_dict['label']
    return spec, image, label

def get_data_with_idx(data_dict):
    spec = data_dict['a']
    image = data_dict['v']
    label = data_dict['label']
    idx = data_dict['idx']
    return spec, image, label, idx

def train(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None, logger=None):
    criterion = nn.CrossEntropyLoss()
    if args.fusion_method == 'attention':
            return train_attn(args, epoch, model, criterion, device, dataloader, optimizer, scheduler, writer, logger)
    elif args.fusion_method in ['concat', 'sum']:
        return train_base(args, epoch, model, criterion, device, dataloader, optimizer, scheduler, writer, logger)
    elif args.fusion_method in ['uni_A', 'uni_V']:
        return train_uni(args, epoch, model, criterion, device, dataloader, optimizer, scheduler, writer, logger)
    

def train_uni(args, epoch, model, criterion, device, dataloader, optimizer, scheduler, writer=None, logger=None):
    model.train()
    print("Start training ... ")

    _loss = 0

    for step, (data) in tqdm(enumerate(dataloader)):
        spec, image, label = get_data(data)
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(spec.unsqueeze(1).float(), image.float())
        out = output['out']

        loss = criterion(out, label)

        loss.backward()

        print("epoch: " + str(epoch) + ", iter " + str(step) + ", loss: %.2f, loss_a: %.2f, loss_v: %.2f" % (loss.item(), loss.item(), loss.item()))
        optimizer.step()
        if args.backbone not in ['resnet18']:
            scheduler.step()

        _loss += loss.item()

    current_lr = optimizer.param_groups[-1]['lr']
    scheduler.step()
    logger.log({'epoch': epoch, 'loss': loss.item() / len(dataloader), 'lr': current_lr})

    return _loss / len(dataloader), _loss / len(dataloader), _loss / len(dataloader)


def train_base(args, epoch, model, criterion, device, dataloader, optimizer, scheduler, writer=None, logger=None):
    model.train()
    softmax = nn.Softmax(dim=1)
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, (data) in tqdm(enumerate(dataloader)):
        spec, image, label = get_data(data)
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        output = model(spec.unsqueeze(1).float(), image.float())
        out = output['out']
        out_a = output['out_a']
        out_v = output['out_v']
        loss = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)

        # train_main
        loss.backward() 
         # train_probe
        loss_a.backward()
        loss_v.backward()

        print("epoch: " + str(epoch) + ", iter " + str(step) + ", loss: %.2f, loss_a: %.2f, loss_v: %.2f" % (loss.item(), loss_a.item(), loss_v.item()))
        optimizer.step()
        scheduler.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    current_lr = optimizer.param_groups[-1]['lr']
    scheduler.step()
    logger.log({'epoch': epoch, 'loss': loss.item() / len(dataloader), 'lr': current_lr})

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def train_attn(args, epoch, model, criterion, device, dataloader, optimizer, scheduler, writer=None, logger=None):
    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    ka = []
    kv = []
    q_cls_list = []

    for step, (data) in tqdm(enumerate(dataloader)):
        spec, image, label = get_data(data)
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(spec.unsqueeze(1).float(), image.float())

        out = output['out'][0]
        attn_matrix = output['out'][1]
        out_a = output['out_a']
        out_v = output['out_v']
        
        # RECORD
        token_a = output['token_a']
        seq_a_len = token_a.shape[1] # [B, L, edim]
        token_v = output['token_v']
        seq_v_len = token_v.shape[1]
        av_dim = seq_a_len + 1 # cls [0], token_a [1:seq_a_len+1], token_a [seq_a_len+1:]
        qkv = output['out'][2]

        qcls, k_a, k_v = get_QKV_Feature(qkv, av_dim)
        if step >= len(dataloader) - 5:
            q_cls_list.append(qcls.detach().clone())
            ka.append(k_a.detach().clone())
            kv.append(k_v.detach().clone())

        loss = criterion(out, label)
        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)

        # train_main
        loss.backward()
        loss_a.backward() # probe
        loss_v.backward()

        print("epoch: " + str(epoch) + ", iter " + str(step) + ", loss: %.2f, loss_a: %.2f, loss_v: %.2f" % (loss.item(), loss_a.item(), loss_v.item()))
        optimizer.step()
        scheduler.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    current_lr = optimizer.param_groups[-1]['lr']
    scheduler.step()
    logger.log({'epoch': epoch, 'loss': _loss / len(dataloader), 'lr': current_lr})

    with torch.no_grad():
        # ka[-1] [B, L, edim]
        avg_ka = torch.mean(ka[-1], dim=1) # -> [B, edim]
        avg_kv = torch.mean(kv[-1], dim=1) # -> [B, edim]
        query = q_cls_list[-1] # q_cls_list[-1] in [B, edim]

        cos_theta_a = F.cosine_similarity(query, avg_ka) # [B, 1]
        cos_theta_v = F.cosine_similarity(query, avg_kv) # [B, 1]
        print("cosine av: ", cos_theta_a, cos_theta_v)
        AIR = torch.mean(cos_theta_a - cos_theta_v)
        print("AIR: ", AIR)
        
        with open(os.path.join(args.exp_root, 'AIR.txt'), '+a') as file:
            file.write("%.2f\n" % AIR)
        print(args.beta)

        if args.modulation == "roll" and model.module.fusion_module.r_time < args.r_time:
            if torch.abs(AIR) > args.beta:
                print(f"Set Rotary Matrix at {epoch}")
                set_rotary_matrix(args=args, model=model, AIR=AIR, device=device, dataloader=dataloader)
                with open(os.path.join(args.exp_root, 'AIR.txt'), '+a') as file:
                    file.write(f"Set Rotary Matrix at {epoch}\n")


    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)

