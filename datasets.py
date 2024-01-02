# /dehazyDeepFusionNetwork/datasets.py
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import args

# yang's hazy datasets
class YHD_Datasets(Dataset):
    def __init__(self, file_path, data_process_way, transform=None):
        self.hazy_path = []
        self.clear_path = []
        self.t_path = []
        self.data_process_way = data_process_way

        with open(file_path, 'r') as f:
            for x in f.readlines():
                r = x.strip().split('|')
                self.hazy_path.append(r[0])
                self.clear_path.append(r[1])
                self.t_path.append(r[2])
        self.transform = transform

    def __getitem__(self, idx):
        clear = cv2.imread(self.clear_path[idx], cv2.IMREAD_COLOR)
        hazy = cv2.imread(self.hazy_path[idx], cv2.IMREAD_COLOR)
        t = cv2.imread(self.hazy_path[idx], cv2.IMREAD_GRAYSCALE)

        # 选择图像处理方式，用以符合图像输入尺寸
        if self.data_process_way == 1:
            clear = cv2.copyMakeBorder(clear, 4,4,0,0, borderType= cv2.BORDER_CONSTANT, value=[0, 0, 0])
            hazy = cv2.copyMakeBorder(hazy, 4, 4, 0, 0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            t = cv2.copyMakeBorder(t, 4, 4, 0, 0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        t[t<=130] = 0
        t[t>130] = 1

        # 归一化处理
        clear = clear /255.0
        hazy = hazy / 255.0

        clear = torch.from_numpy(clear).float().permute(2, 0, 1)
        hazy = torch.from_numpy(hazy).float().permute(2, 0, 1)
        t = torch.from_numpy(t).unsqueeze(0)
        t_hazy = {'t': t, 'hazy': hazy}
        return t_hazy, clear

    def __len__(self):
        return len(self.clear_path)


if __name__ == '__main__':
    args = args.parse_args()
    file_path = r'D:\experiment_data\small_data\catalog.txt'
    test_data = YHD_Datasets(file_path, args, transform=None)
    t_hazy, clear = test_data[0]
    t = t_hazy['t']
    hazy = t_hazy['hazy']
    print(f'clear shape: {clear.shape}, unique: {np.unique(clear)}')
    print(f'hazy shape: {hazy.shape}, unique: {np.unique(hazy)}')
    print(f't shape: {t.shape}, unique: {np.unique(t)}')
