import os
import sys
import time
import argparse

import torch

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import write_csv

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_datadir', help='test dataset directory', type=str, default='C:\\Users\\chengan_huang\\Desktop\\computervision\\hw2\\p2\\hw2_data\\p2_data\\val\\')
    parser.add_argument('--model_type', help='mynet or resnet18', type=str, default='resnet18')
    parser.add_argument('--output_path', help='output csv file path', type=str, default='C:\\Users\\chengan_huang\\Desktop\\computervision\\hw2\\p2\\hw2_data\\pred.csv')
    args = parser.parse_args()

    model_type = args.model_type
    test_datadir = args.test_datadir
    output_path = args.output_path

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    ##### NOTE: Put your best trained models to checkpoint/ #####
    if model_type == 'mynet':
        model = MyNet()
        model.load_state_dict(torch.load('C:\\Users\\chengan_huang\\Desktop\\computervision\\hw2\\p2\\checkpoint\\mynet_best.pth', map_location=torch.device('cpu')))
    elif model_type == 'resnet18':
        model = ResNet18()
        model.load_state_dict(torch.load('C:\\Users\\chengan_huang\\Desktop\\computervision\\hw2\\p2\\checkpoint\\resnet18_best.pth', map_location=torch.device('cpu')))
    else:
        raise NameError('Unknown model type')
    model.to(device)

    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    test_loader = get_dataloader(test_datadir, batch_size=1, split='test')

    ##### INFERENCE #####
    predictions = []
    model.eval()
    with torch.no_grad():
        test_start_time = time.time()


        for batch_id, (inputs, _) in enumerate(test_loader):
            print('here')
            print(inputs,type(inputs))
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
            sys.stdout.write('\rProgreming...: {:.1f}%'.format(100.0 * (batch_id + 1) / len(test_loader)))
            sys.stdout.flush()

    test_time = time.time() - test_start_time
    print()
    print(f'Finish testing {test_time:.2f} sec(s), dumps result to {output_path}')

    ##### WRITE RESULT #####
    write_csv(output_path, predictions, test_loader)
    
if __name__ == '__main__':
    main()
