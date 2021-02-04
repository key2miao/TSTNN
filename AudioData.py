import torch
from torch.utils.data import Dataset
from preprocess import SignalToFrames, ToTensor
import soundfile as sf
import numpy as np
import random
import h5py
import glob
import os


class TrainingDataset(Dataset):
    r"""Training dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256, nsamples=64000):
        # file_path is the path of training dataset
        # option1: '/data/KaiWang/pytorch_learn/pytorch_for_speech/dataset/timit_mix/trainset/two_data'
        # option2 : .txt file format  file_path='/data/KaiWang/pytorch_learn/pytorch_for_speech/DDAEC/train_file_list'

        #self.file_list = glob.glob(os.path.join(file_path, '*'))

        with open(file_path, 'r') as train_file_list:
            self.file_list = [line.strip() for line in train_file_list.readlines()]

        self.nsamples = nsamples
        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        #print(len(self.file_list))
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')
        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]
        reader.close()

        size = feature.shape[0]
        start = random.randint(0, max(0, size + 1 - self.nsamples))
        feature = feature[start:start + self.nsamples]
        label = label[start:start + self.nsamples]

        feature = np.reshape(feature, [1, -1])  # [1, sig_len]
        label = np.reshape(label, [1, -1])  # [1, sig_len]

        feature = self.get_frames(feature)  # [1, num_frames, sig_len]

        feature = self.to_tensor(feature)  # [1, num_frames, sig_len]
        label = self.to_tensor(label)  # [1, sig_len]

        return feature, label


class EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256):

        #self.filename = filename
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]

        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')

        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]

        feature = np.reshape(feature, [1, 1, -1])  # [1, 1, sig_len]

        feature = self.get_frames(feature)  # [1, 1, num_frames, frame_size]

        feature = self.to_tensor(feature)  # [1, 1, num_frames, frame_size]
        label = self.to_tensor(label)  # [sig_len, ]

        return feature, label


class Company_EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256):

        #self.filename = filename
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]

        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')

        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]

        #feature = np.reshape(feature, [1, 1, -1])  # [1, 1, sig_len]

        #feature = self.get_frames(feature)  # [1, 1, num_frames, frame_size]

        #feature = self.to_tensor(feature)  # [sig_len, ]
        label = self.to_tensor(label)  # [sig_len, ]

        return feature, label



# testing 中 clean 和 noisy分不同的noisy和dB
class TestDataset(Dataset):
    def __init__(self, clean_file_path, noisy_file_path, frame_size, frame_shift):
        self.clean_test_name = os.listdir(clean_file_path)
        self.noisy_test_name = os.listdir(noisy_file_path)
        self.noisy_file_path = noisy_file_path
        self.clean_file_path = clean_file_path

        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.clean_test_name)

    def __getitem__(self, index):
        noisy_name = '%s_%s.wav' % (self.clean_test_name[index].split('.')[0], os.path.basename(self.noisy_file_path))
        if noisy_name in self.noisy_test_name:
            noisy_audio, sr = sf.read(os.path.join(self.noisy_file_path, noisy_name))
            clean_audio, sr1 = sf.read(os.path.join(self.clean_file_path, self.clean_test_name[index]))
            if sr != 16000 and sr1 != 16000:
                raise ValueError('Invalid sample rate')

            feature = np.reshape(noisy_audio, [1, 1, -1])  # [1, 1, sig_len]

            feature = self.get_frames(feature)  # [1, 1, num_frames, frame_size]
            feature = self.to_tensor(feature)   # [1, 1, num_frames, frame_size]
            label = self.to_tensor(clean_audio)  # [sig_len, ]

        else:
            raise TypeError('Invalid noisy audio file')

        return feature, label


class TrainCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            feat_dim = batch[0][0].shape[-1]  # frame_size
            label_dim = batch[0][1].shape[-1]  # sig_len

            feat_nchannels = batch[0][0].shape[0]  # 1
            label_nchannels = batch[0][1].shape[0]  # 1

            # sorted by sig_len for label
            sorted_batch = sorted(batch, key=lambda x: x[1].shape[1], reverse=True)
            # (num_frames, sig_len)
            lengths = list(map(lambda x: (x[0].shape[1], x[1].shape[1]), sorted_batch))

            padded_feature_batch = torch.zeros((len(lengths), feat_nchannels, lengths[0][0], feat_dim))
            padded_label_batch = torch.zeros((len(lengths), label_nchannels, lengths[0][1]))
            lengths1 = torch.zeros((len(lengths),), dtype=torch.int32)

            for i in range(len(lengths)):
                padded_feature_batch[i, :, 0:lengths[i][0], :] = sorted_batch[i][0]
                padded_label_batch[i, :, 0:lengths[i][1]] = sorted_batch[i][1]
                lengths1[i] = lengths[i][1]

            return padded_feature_batch, padded_label_batch, lengths1
        else:
            raise TypeError('`batch` should be a list.')


class EvalCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')

class Company_EvalCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')



class TestCollate(object):

    def __init__(self):
        self.name = 'collate'

    def __call__(self, batch):
        if isinstance(batch, list):
            # testdataloder 中的batch_size = 1; 因此就返回仅有的一个(feature, label)
            return batch[0][0], batch[0][1]
        else:
            raise TypeError('`batch` should be a list.')