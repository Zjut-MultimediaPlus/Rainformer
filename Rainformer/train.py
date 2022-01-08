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
file_idx = 1
device = torch.device('cuda:' + str(cuda_idx))
torch.cuda.set_device(device)
train_seq, test_seq = np.load('../train_seq.npy'), np.load('../test_seq.npy')
val_seq = train_seq[5000:]
train_seq = train_seq[:5000]

# -----------------------------------------------
train_data, test_data = get_data()
epoch_size, batch_size = 200, 24
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

# -----------------------------------------------
opt = optim.Adam(net.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=5, verbose=True)
criterion = BMAEloss().cuda()

def MSE(y_hat, y):
    sub = y_hat - y
    return np.sum(sub * sub)

# -----------------------------------------------
for epoch in range(1, epoch_size + 1):
    f = open('log_' + str(file_idx) + '.txt', 'a+')
    train_l_sum, test_l_sum, n = 0.0, 0.0, 0
    net.train()
    np.random.shuffle(train_seq)
    ran = np.arange(batch_size, train_seq.shape[0], batch_size)
    pbar = tqdm(ran)
    for batch in pbar:
        x, y = data_2_cnn(train_data, batch, batch_size, train_seq)
        x = x.cuda()
        y = y.cuda()
        y_hat = net(x)
        loss = criterion(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_num = loss.detach().cpu().numpy()
        pbar.set_description('Train MSE Loss: ' + str(loss_num / batch_size))
        train_l_sum += loss_num
        n += batch_size
    train_loss = train_l_sum / n
    n = 0
    net.eval()

    with torch.no_grad():
        np.random.shuffle(val_seq)
        ran = np.arange(batch_size, val_seq.shape[0], batch_size)
        pbar = tqdm(ran)

        for batch in pbar:
            # 因为验证集是从训练集中得到的，所以还是train_data
            x, y = data_2_cnn(train_data, batch, batch_size, val_seq)
            x = x.cuda()
            y = y.cuda()
            y_hat = net(x)

            loss = criterion(y_hat, y)
            loss_num = loss.detach().cpu().numpy()
            test_l_sum += loss_num
            pbar.set_description('Test MSE Loss: ' + str(loss_num / batch_size))
            n += batch_size

        f.write('Iter: ' + str(epoch) + ' ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
        print('Iter:', epoch, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        test_loss = test_l_sum / n
        lr_scheduler.step(test_loss)

        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'model' + str(cuda_idx) + '_' + str(file_idx) + '_' + str(epoch) + '.pt')

    f.write('Train loss: ' + str(train_loss) + ' Test loss: ' + str(test_loss) + '\n')
    print('Train loss:', train_loss, ' Test loss:', test_loss)
    print('==='*20)
    seg_line = '=======================================================================' + '\n'
    f.write(seg_line)
    f.close()
