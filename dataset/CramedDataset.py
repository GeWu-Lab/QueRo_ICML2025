import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb

class CramedDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './data/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = '/home/haotian_ni/CREMA-D'
        self.audio_feature_path  = '/home/haotian_ni/CREMA-D/AudioWAV'
        # In this case, I download directly from github repo with git-lfs

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv  = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])
                # visual_path = os.path.join(self.visual_feature_path, 'Image-10-FPS'.format(self.args.fps), item[0])
                # print(visual_path)
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    if not os.path.exists(audio_path):
                        print("Audio Path not found")
                    if not os.path.exists(visual_path):
                        print("Visual Path not found")
                    # break
                    continue
            print('sample_num: ', len(self.label))
        
        # self.replace = True
        self.replace = False
        if self.replace:
            print("DANGEROUS!!! 50% replace a by noise")
            print('Generating noise samples...')
            keys = np.random.choice(np.arange(0, len(self.label)), size=len(self.label) // 2, replace=False)
            self.replace_dict = {key : -7 + 5 * np.random.normal(size=(257, 188)) for key in keys}
            # self.replace_dict = {key :np.random.normal(size=(257, 188)) for key in keys}
        print('Done.')


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        if self.replace and idx in self.replace_dict:
            print('replace')
            spectrogram = self.replace_dict[idx]
        else:
            # print('No replace')
            samples, rate = librosa.load(self.audio[idx], sr=22050)
            resamples = np.tile(samples, 3)[:22050*3]
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.

            spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
            # mean = np.mean(spectrogram)
            # std = np.std(spectrogram)
            # print('mean & std', mean, std)
            #spectrogram = np.divide(spectrogram - mean, std + 1e-9)


        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        # pick_num = self.args.fps
        pick_num = self.args.use_video_frames
        # select_index = np.random.choice(len(image_samples), size=pick_num, replace=False)
        select_index = np.random.choice(len(image_samples), size=pick_num, replace=True)
        select_index.sort()

        images = torch.zeros((pick_num, 3, 224, 224))
        for i in range(pick_num):
            img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
            img = transform(img)
            images[i] = img

        # images = torch.permute(images, (1,0,2,3))
        '''
            ORIGINAL ERROR
        '''
        images = images.permute((1,0,2,3)) # (C, T, H, W)
        # print(images.shape)

        # label
        label = self.label[idx]
        
        return {
            'a': spectrogram, 
            'v': images, 
            'label': label,
            'idx': idx
            }