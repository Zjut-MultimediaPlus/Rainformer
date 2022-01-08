import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from PIL import Image
import os

def show(a):
    plt.imshow(a)
    plt.show()

def to_np(a):
    return a.cpu().detach().numpy()

def get_data():
    root = '../../../../data/sf/KNMI/KNMI.h5'
    f = h5py.File(root, mode='r')
    train = f['train']
    test = f['test']
    train_image = train['images']
    test_image = test['images']
    return train_image, test_image


def psi(a, scale=4):
    # a shape is [B, S, H, W]
    B, S, H, W = a.shape
    C = scale ** 2
    new_H = int(H // scale)
    new_W = int(W // scale)
    a = np.reshape(a, (B, S, new_H, scale, new_W, scale))
    a = np.transpose(a, (0, 1, 3, 5, 2, 4))
    a = np.reshape(a, (B, S, C, new_H, new_W))
    return a

def inverse(a, scale=4):
    B, S, C, new_H, new_W = a.shape
    H = int(new_H * scale)
    W = int(new_W * scale)
    a = np.reshape(a, (B, S, scale, scale, new_H, new_W))
    a = np.transpose(a, (0, 1, 4, 2, 5, 3))
    a = np.reshape(a, (B, S, H, W))
    return a

def get_mask(eta, shape, test=False):
    B, S, C, H, W = shape
    if test:
        return torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0
    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1
    return eta, torch.tensor(mask, dtype=torch.float)

def data_2_rnn_mask(data, batch, batch_size, sequence, scale, eta, test=False):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)

    B, S, C, H, W = result.shape

    if test:
        return result, torch.zeros((B, int(S // 2), C, H, W))
    eta -= 0.00002
    if eta < 0:
        eta = 0

    mask = np.random.random_sample((B, int(S // 2), C, H, W))
    mask[mask < eta] = 0
    mask[mask > eta] = 1

    return result, torch.tensor(mask, dtype=torch.float), eta


def data_2_rnn(data, batch, batch_size, sequence, scale):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    result = psi(result, scale=scale)
    return result

def data_2_cnn(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    x = result[:, :9]
    y = result[:, 9:]
    return x, y

def data_2_cnn2(data, batch, batch_size, sequence):
    sequence = sequence[batch - batch_size:batch]
    result = []
    for i in sequence:
        tmp = data[i] * 4783 / 100 * 12
        result.append(torch.tensor(tmp, dtype=torch.float))
    result = torch.stack(result, dim=0)
    return result

def inverse_cnn2(x, y):
    x = torch.unsqueeze(x, dim=1)
    y = torch.unsqueeze(y, dim=1)
    x = to_np(x)
    y = to_np(y)
    x = inverse(x, scale=3)
    y = inverse(y, scale=3)

    x2 = np.zeros((x.shape[0], 9, 288, 288))
    y2 = np.zeros((y.shape[0], 9, 288, 288))

    index = 0

    for i in range(0, 864, 288):
        for j in range(0, 864, 288):
            x2[:, index] = x[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            y2[:, index] = y[:, 0, i:i+x2.shape[2], j:j+x2.shape[2]]
            index += 1
    return x2, y2


def _draw_color(t, flag, color):
    r = t[:, :, 0]
    g = t[:, :, 1]
    b = t[:, :, 2]
    r[flag] = color[0]
    g[flag] = color[1]
    b[flag] = color[2]
    return t



def draw_color_single(y):
    t = np.ones((y.shape[0], y.shape[1], 3)) * 255
    tt1 = []
    index = 0.5
    for i in range(30):
        tt1.append(index)
        index += 1
    color = [[28, 230, 180], [39, 238, 164], [58, 245, 143], [74, 248, 128], [97, 252, 108],
             [121, 254, 89], [143, 255, 73], [159, 253, 63], [173, 251, 56], [190, 244, 52],
             [203, 237, 52], [215, 229, 53], [227, 219, 56], [238, 207, 58], [246, 195, 58],
             [251, 184, 56], [254, 168, 51], [254, 153, 44], [253, 138, 38], [249, 120, 30],
             [244, 103, 23], [239, 88, 17], [231, 73, 12], [221, 61, 8], [212, 51, 5],
             [202, 42, 4], [188, 32, 2], [172, 23, 1], [158, 16, 1], [142, 10, 1]]

    for i in range(30):
        rain = y >= tt1[i]
        _draw_color(t, rain, color[i])
    #
    # rain_1 = y >= 0.5
    # rain_2 = y >= 2
    # rain_3 = y >= 5
    # rain_4 = y >= 10
    # rain_5 = y >= 30
    # _draw_color(t, rain_1, [156, 247, 144])
    # _draw_color(t, rain_2, [55, 166, 0])
    # _draw_color(t, rain_3, [103, 180, 248])
    # _draw_color(t, rain_4, [0, 2, 254])
    # _draw_color(t, rain_5, [250, 3, 240])
    t = t.astype(np.uint8)
    return t

def fundFlag(a, n, m):
    flag_1 = np.uint8(a >= n)
    flag_2 = np.uint8(a < m)
    flag_3 = flag_1 + flag_2
    return flag_3 == 2

def B_mse(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mse = np.sum(mask * ((a - b) ** 2)) / n
    return mse

def B_mae(a, b):
    mask = np.zeros(a.shape)
    mask[a < 2] = 1
    mask[fundFlag(a, 2, 5)] = 2
    mask[fundFlag(a, 5, 10)] = 5
    mask[fundFlag(a, 10, 30)] = 10
    mask[a > 30] = 30
    n = a.shape[0] * b.shape[0]
    mae = np.sum(mask * np.abs(a - b)) / n
    return mae

def draw_color(data):
    B, C, H, W = data.shape
    result = torch.zoers((B, C, H, W, 3))
    for i in range(B):
        for j in range(C):
            result[B, C] = draw_color_single(data[B, C])
    return result

def tp(pre, gt):
    return np.sum(pre * gt)

def fn(pre, gt):
    a = pre + gt
    flag = (gt == 1) & (a == 1)
    return np.sum(flag)

def fp(pre, gt):
    a = pre + gt
    flag = (pre == 1) & (a == 1)
    return np.sum(flag)

def tn(pre, gt):
    a = pre + gt
    flag = a == 0
    return np.sum(flag)

def _csi(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    return TP / (TP + FN + FP + eps)


def _hss(pre, gt):
    eps = 1e-9
    TP, FN, FP, TN = tp(pre, gt), fn(pre, gt), fp(pre, gt), tn(pre, gt)
    a = TP * TN - FN * FP
    b = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN) + eps
    if a / b < 0:
        return 0
    return a / b

def csi(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_csi(a, b))
    return result

def hss(pred, gt):
    threshold = [0.5, 2, 5, 10, 30]
    result = []
    for i in threshold:
        a = np.zeros(pred.shape)
        b = np.zeros(gt.shape)
        a[pred >= i] = 1
        b[gt >= i] = 1
        result.append(_hss(a, b))
    return result

import torch
from torch import nn

class BMAEloss(nn.Module):
    def __init__(self):
        super(BMAEloss, self).__init__()

    def fundFlag(self, a, n, m):
        flag_1 = (a >= n).int()
        flag_2 = (a < m).int()
        flag_3 = flag_1 + flag_2
        return flag_3 == 2

    def forward(self, pred, y):
        mask = torch.zeros(y.shape).cuda()
        mask[y < 2] = 1
        mask[self.fundFlag(y, 2, 5)] = 2
        mask[self.fundFlag(y, 5, 10)] = 5
        mask[self.fundFlag(y, 10, 30)] = 10
        mask[y > 30] = 30
        return torch.sum(mask * torch.abs(y - pred))


# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.figure(dpi=200)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# curve_type = 'mae'
#
# x = np.arange(5, 50, 5)
#
# conv_mae = np.load('ConvLSTM/' + curve_type + '.npy')
# rainformer_mae = np.load('Rainformer_2/' + curve_type + '.npy')
# pfst_mae = np.load('PFST/' + curve_type + '.npy')
# mim_mae = np.load('MIM/' + curve_type + '.npy')
# predrnn_mae = np.load('PredRNN/' + curve_type + '.npy')
# predrnnpp_mae = np.load('PredRNN++/' + curve_type + '.npy')
# causal_mae = np.load('CausalLSTM/' + curve_type + '.npy')
# sa_mae = np.load('SAConvLSTM/' + curve_type + '.npy')
#
# plt.xlabel('Prediction interval (min)')
# plt.ylabel(curve_type)
#
# line1, = plt.plot(x, conv_mae, label='ConvLSTM')
# line2, = plt.plot(x, predrnn_mae, label='PredRNN')
# line3, = plt.plot(x, predrnnpp_mae, label='PredRNN++')
# line4, = plt.plot(x, causal_mae, label='CausalLSTM')
# line5, = plt.plot(x, mim_mae, label='MIM')
# line6, = plt.plot(x, pfst_mae, label='PFST')
# line7, = plt.plot(x, sa_mae, label='SA-ConvLSTM')
# line8, = plt.plot(x, rainformer_mae, color='black', label='Rainformer')
#
# plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8], labels=['ConvLSTM', 'PredRNN', 'PredRNN++',
#                                                                               'CausalLSTM', 'MIM', 'PFST', 'SA-ConvLSTM', 'Rainformer'], loc='best')
# plt.show()
