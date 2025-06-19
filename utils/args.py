import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    
    # model
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['uni_A', 'uni_V', 'sum', 'concat', 'attention'])
    parser.add_argument('--modulation', default='norm', type=str,
                        choices=['norm', 'roll'])
    parser.add_argument('--backbone', default='vit', type=str)
    parser.add_argument('--depth', default=4, type=int)
    parser.add_argument('--multi_depth', default=0, type=int)
    parser.add_argument('--edim', default=768, type=int)
    
    # dataset
    parser.add_argument('--dataset', default='KineticSound', type=str)
    parser.add_argument('--n_classes', default=-1, type=int)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)

    # train
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0, 1, 2, 3', type=str, help='GPU ids')

    # method_params
    parser.add_argument('--r_time', default=1, type=int, help='rotate time limits')
    parser.add_argument('--beta', default=0.0, type=float, help='bound of AIR')
    parser.add_argument('--sigma', default=0.0, type=float, help='rate for AIR')

    # record
    parser.add_argument('--exp_root', default='', type=str)
    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')

    # mode train/eval
    parser.add_argument('--ckpt', default='', type=str, help='where to load a new model')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    return parser.parse_args()
