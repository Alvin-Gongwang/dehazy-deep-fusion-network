# /dehazyDeepFusionNetwork/args.py

import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)

    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--data_process_way', type=int, default=1)

    parser.add_argument('--train_catalog', type=str, default=r'D:\experiment_data\small_data\train_catalog.txt')
    parser.add_argument('--test_catalog', type=str, default=r'D:\experiment_data\small_data\test_catalog.txt')
    parser.add_argument('--valid_catalog', type=str, default=r'D:\experiment_data\small_data\valid_catalog.txt')

    return parser.parse_args()