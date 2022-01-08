import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt
from Rainformer import Net
import torch.nn.functional as F
from tqdm import tqdm, trange
import sys
sys.path.append('..')
from tool import *
from skimage import measure
import time

# -----------------------------------------------
cuda_idx = 1
file_idx = 2
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)
test_seq = np.load('../test_seq.npy')

# -----------------------------------------------
train_data, test_data = get_data()
epoch_size, batch_size = 200, 28
in_channel = 9
out_channel = 9
net = Net(
    input_channel=9,
    hidden_dim=96,
    downscaling_factors=(4, 2, 2, 2),
    layers=(2, 2, 2, 2),
    heads=(3, 6, 12, 24),
    head_dim=32,
    window_size=9,
    relative_pos_embedding=True,).cuda()
min_test_loss, out_count = 1e10, 0
min_mae = 1e10
net.load_state_dict(torch.load('model1_1_100.pt'))

# -----------------------------------------------
net.eval()
# 测试数据
with torch.no_grad():
    ran = np.arange(batch_size, test_seq.shape[0], batch_size)
    pbar = tqdm(ran)
    threshold = [0.5, 2, 5, 10, 30]
    CSI, HSS, mse, mae = [], [], [], []
    for i in range(5):
        CSI.append([])
        HSS.append([])

    for batch in pbar:
        x, y = data_2_cnn(test_data, batch, batch_size, test_seq)
        x = x.cuda()
        y = y.cuda()
        y_hat = net(x)

        y_hat = to_np(y_hat)
        y = to_np(y)
        for i in range(batch_size):
            for j in range(9):
                a, b = y[i, j], y_hat[i, j]
                mse.append(B_mse(a, b))
                mae.append(B_mae(a, b))
                csi_result = csi(a, b)
                hss_result = hss(a, b)

                for t in range(5):
                    CSI[t].append(csi_result[t])
                    HSS[t].append(hss_result[t])

    for i in range(5):
        CSI[i] = np.array(CSI[i]).mean()
        HSS[i] = np.array(HSS[i]).mean()
    mse = np.array(mse).mean()
    mae = np.array(mae).mean()

    print('CSI: ')
    for i in range(5):
        print('r >=', threshold[i], ':', CSI[i], end=' ')
    print()
    print('HSS:')
    for i in range(5):
        print('r >=', threshold[i], ':', HSS[i], end=' ')
    print()
    print('MSE:', mse, 'MAE:', mae)

