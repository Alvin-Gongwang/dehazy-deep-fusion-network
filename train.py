# /dehazyDeepFusionNetwork/train.py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import random

import args
from datasets import YHD_Datasets
# from model.ydffn import Ydffn
from model.unet import UNet

# 设置相同随机种子数
def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)  # Numpy model
    random.seed(seed)  # Python random model
    torch.backends.cudnn.deterministic = True  # cuda随机种子固定
    torch.backends.cudnn.benchmark = True  # 提升训练速度

# 显示设定参数
def show_args(args):
    print('-' * 18 + 'args' + '-' * 18)
    for key, value in args.__dict__.items():
        print(f'{key}:{value}')
    print('-' * 40)

def create_model(model_name):
    model = None
    if model_name == 'unet':
        return UNet()
    return model
def train(model, train_loader, critertion, optimizer, device):
    model.train()
    loss = 0
    tl = tqdm(train_loader)
    for X, y in tl:
        t = X['t']
        hazy = X['hazy']
        hazy, y = hazy.to(device), y.to(device)
        t = t.to(device)
        pred = model(hazy)
        loss = critertion(pred, y)
        tl.desc = 'loss: {:.3f}'.format(loss.item())
        loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss = loss / len(train_loader)
    return {'loss': loss}


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备为cuda或cpu

    train_dataset = YHD_Datasets(args.train_catalog, args.data_process_way)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    test_datasets = YHD_Datasets(args.test_catalog, args.data_process_way)
    test_loader = DataLoader(
        dataset=test_datasets,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    model = create_model(args.model)
    model.to(device)

    critertion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for e in range(args.epochs):
        train_metric = train(model, train_loader, critertion, optimizer, device)
        print(train_metric)


# 主入口
if __name__ == '__main__':
    args = args.parse_args()
    show_args(args)
    main(args)

