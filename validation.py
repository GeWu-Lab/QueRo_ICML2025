import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os.path as osp


def get_data(data_dict):
    spec = data_dict['a']
    image = data_dict['v']
    label = data_dict['label']
    return spec, image, label


def valid(args, model, device, dataloader, epoch, logger):
    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        loss = 0.0

        with torch.no_grad():
            for step, (data) in tqdm(enumerate(dataloader)):
                spec, image, label = get_data(data)
                
                spec = spec.to(device)
                image = image.to(device)
                label = label.to(device)

                # print(image.shape)
                output = model(spec.unsqueeze(1).float(), image.float())
                out   = output['out']
                if isinstance(out,list):
                    out = out[0]
                out_v = output['out_v']
                out_a = output['out_a']

                loss += criterion(out, label)
                prediction = softmax(out)
                pred_v = softmax(out_v)
                pred_a = softmax(out_a)

                for i in range(image.shape[0]):
                    ma = np.argmax(prediction[i].cpu().data.numpy())
                    v = np.argmax(pred_v[i].cpu().data.numpy())
                    a = np.argmax(pred_a[i].cpu().data.numpy())

                    num[label[i]] += 1.0
                
                    #pdb.set_trace()
                    if np.asarray(label[i].cpu()) == ma:
                        acc[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == v:
                        acc_v[label[i]] += 1.0
                    if np.asarray(label[i].cpu()) == a:
                        acc_a[label[i]] += 1.0
        
        if logger is not None:
            logger.log({'epoch': epoch, 'loss': loss.item() / len(dataloader), 'acc': sum(acc) / sum(num)})

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)

