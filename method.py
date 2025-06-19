import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils import *

def set_rotary_matrix(args, model, AIR, device, dataloader):
    model.eval()

    with torch.no_grad():
        ka = []
        kv = []
        qcls_ls = []

        print("estimate the k space distribution")
        for step, data in tqdm(enumerate(dataloader)):
            spec = data['a'].to(device)
            image = data['v'].to(device)
            # label = data['label'].to(device)

            output = model(spec.unsqueeze(1).float(), image.float())

            token_a = output['token_a']
            seq_a_len = token_a.shape[1] # [B, L, edim]
            av_dim = seq_a_len + 1 # cls [0], token_a [1:seq_a_len+1], token_a [seq_a_len+1:]

            qkv = output['out'][2]

            qcls, k_a, k_v = get_QKV_Feature(qkv, av_dim)
            device = qcls.device

            batch_avg_ka = torch.mean(torch.mean(k_a.detach().clone(), dim=1), dim=0)
            ka.append(batch_avg_ka)
            batch_avg_kv = torch.mean(torch.mean(k_v.detach().clone(), dim=1), dim=0)
            kv.append(batch_avg_kv)
            batch_avg_qcls = torch.mean(qcls.detach().clone(), dim=0) # 
            qcls_ls.append(batch_avg_qcls)
            # break

        # Now Calculate Where it's
        if len(ka) == 1:
            src_ka = ka
            src_kv = kv
            query_ls = qcls_ls
        else:
            src_ka = torch.stack(ka, dim=0) # [batch_num, edim]
            src_kv = torch.stack(kv, dim=0) # [batch_num, edim]
            query_ls  = torch.stack(qcls_ls, dim=0)
        
        if len(src_kv.shape) > 1:
            avg_a = torch.mean(src_ka, dim=0)
            avg_v = torch.mean(src_kv, dim=0)
            query = torch.mean(query_ls, dim=0)
        else:
            avg_a = src_ka
            avg_v = src_kv
            query = query_ls

        # alpha given by equation
        tanh = nn.Tanh()
        alpha = (1 + tanh(- args.sigma * AIR)) / 2
        with open(os.path.join(args.exp_root, 'AIR.txt'), '+a') as file:
            file.write(f"weight for A={alpha}, weight for V={1-alpha}\n")
        
        target_cls = alpha * avg_a + (1-alpha) * avg_v

        v = query

        # Uniform
        tar_norm = torch.norm(target_cls)
        target_cls = target_cls * torch.norm(v) / tar_norm

        print('Calculating rotary matrix')
        rotary_matrix = compute_rotation_matrix(X=v, Y=target_cls)
        print('Done.')

        model.module.fusion_module.set_rotate(rotary_matrix.float(), target_cls)

