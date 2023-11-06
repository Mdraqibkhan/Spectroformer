import argparse
import os
import cv2
from math import log10
import time
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import get_test_set

from utils import is_image_file, load_img, save_img
import time
torch.backends.cudnn.benchmark = True
from thop import profile

# Testing settings
parser = argparse.ArgumentParser(description='phaseformer-implementation')
parser.add_argument('--dataset', default='D:/Raqib/underwaterTresting/testing_code/dataset/UCCS/green/', required=False, help='facades')
parser.add_argument('--save_path', default='results/gopro/', required=False, help='facades')
parser.add_argument('--checkpoints_path', default='./checkpoints/blur/', required=False, help='facades')
parser.add_argument('--epoch_count', type=int, default=189,help='the starting epoch count')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--cuda', action='store_false', help='use cuda')
parser.add_argument('--show_flops_params',type=bool, default=True, help='Show number of flops and parameter of model')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")
criterionMSE = nn.MSELoss().to(device)


# G_path = "D:/Raqib/checkpoint/uw_data/netG_model_epoch_{}.pth".format(opt.epoch_count)
G_path = './checkpoints/best.pth'
my_net = torch.load(G_path).to(device)
if opt.show_flops_params:
    input_ = torch.randn (1, 3, 256, 256).cuda()
    flops, params = profile (my_net, inputs = (input_,))
    print("FLOPS of the network ARE::::::::::::::",flops)
    print("FLOPS of the network ARE::::::::::::::",params)    

image_dir = "{}/".format(opt.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

test_set = get_test_set(opt.dataset)
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.test_batch_size, shuffle=False)

start = time.time()
avg_psnr=0
a = 0
times = []
for iteration_test, batch in enumerate(testing_data_loader,1):
    input1, filename = batch[0].to(device), batch[1]     
    final_l = my_net(input1)[0]    
    final_l = final_l.detach().squeeze(0).cpu()
    # print(filename[0])
    save_img(final_l, opt.save_path+filename[0])
