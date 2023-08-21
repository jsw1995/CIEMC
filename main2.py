"""
先压缩再比特加密
采样率为 0.25，封面图像与加密图像一样大小
以256和512两个尺寸作为实验对象
测量加密时间时把中将密文有关输出信息去掉
"""

import argparse
import math
import os
# import cv2 as cv
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import skimage
from net import net1, net2, net3, net4, net21
from matplotlib import pyplot as plt
import hashlib
import time

parser = argparse.ArgumentParser(description="ComRecon")
args = parser.parse_args()

class MyCSNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.config = opt
        self.device = args.cuda if torch.cuda.is_available() else 'cpu'
        self.len1 = [1024, 819, 616, 411, 207, 104, 21, 512]
        self.len2 = [256, 205, 153, 102, 50, 25, 5, 128]

        self.comp = net1(128,128,512).to(self.device)
        self.recon = net2(128,128,512).to(self.device)
        self.se_recon = net3().to(self.device)
        self.se_recon2 = net3().to(self.device)


    def compress(self, img):
        return self.comp(img)

    def recons(self, CSMat):

        img11 = self.recon(CSMat,128,128,512)

        return img11

    def se_recons1(self, img11):

        img21 = self.se_recon(img11)

        return img21

    def se_recons2(self, img21):

        img31 = self.se_recon2(img21)

        return img31



    def loaddict(self, root):
        if self.device == 'cpu':
            self.comp.load_state_dict(
                torch.load(root + '/' + '7abs_comp2_net2.pth', map_location=torch.device('cpu')))
            self.recon.load_state_dict(
                torch.load(root + '/' + '7abs_recon2_net2.pth', map_location=torch.device('cpu')))
            self.se_recon.load_state_dict(
                torch.load(root + '/' + '7abs_se_recon21_net2.pth', map_location=torch.device('cpu')))
            self.se_recon2.load_state_dict(
                torch.load(root + '/' + '7abs_se_recon22_net2.pth', map_location=torch.device('cpu')))


        else:
            self.comp.load_state_dict(torch.load(root + '/' +  '7abs_comp2_net2.pth'))
            self.recon.load_state_dict(torch.load(root + '/' +  '7abs_recon2_net2.pth'))
            self.se_recon.load_state_dict(torch.load(root + '/' + '7abs_se_recon21_net2.pth'))
            self.se_recon2.load_state_dict(torch.load(root + '/' + '7abs_se_recon22_net2.pth'))


    def Device(self):
        return self.device

def npcr(a, b):
    # 像素变化率

    if len(a.shape) == 3:
        [rows, columns, pages] = a.shape
        counter = 0
        for k in range(pages):
            for j in range(columns):
                for i in range(rows):
                    if a[i, j, k] != b[i, j, k]:
                        counter += 1
        c = counter / (rows * columns * pages)
    else:
        [rows, columns] = a.shape
        counter = 0
        for j in range(columns):
            for i in range(rows):
                if a[i, j] != b[i, j]:
                    counter += 1
        c = counter / (rows * columns)
    return c


def uaci(image1, image2):
    # 归一化变化强度
    if len(image1.shape) == 3:
        [rows, columns, pages] = image1.shape
        counter = 0
        for k in range(pages):
            for j in range(columns):
                for i in range(rows):
                    counter = (math.fabs(int(image1[i, j, k]) - int(image2[i, j, k]))) / 256 + counter
        c = counter / (rows * columns * pages)
    else:
        [rows, columns] = image1.shape
        counter = 0
        for j in range(columns):
            for i in range(rows):
                counter = (math.fabs(int(image1[i, j]) - int(image2[i, j]))) / 256 + counter
        c = counter / (rows * columns)
    return c


def LSM(initial, parameters, N):  # 混沌
    x = np.zeros([N + 1001, 1])
    y = np.zeros([N + 1001, 1])
    x[0] = initial[0]
    y[0] = initial[1]
    a = parameters[0]
    b = parameters[1]
    for i in range(N + 1000):
        x[i + 1] = (1 - b * y[i] * y[i]) * math.sin(a / x[i])
        y[i + 1] = (1 - b * x[i] * x[i]) * math.sin(a / y[i])

    return x[1001:N + 1001], y[1001:N + 1001]


def hex_dec(str2):  # 十六转十
    b = eval(str2)
    return b


def dec_hex(str1):  # 十转十六 （输入为str类型）
    a = str(hex(eval(str1)))
    return a


def dec2bin(a):
    if a.ndim == 3:
        [pages, rows, columns] = a.shape
    else:
        [rows, columns] = a.shape
        pages = 1

    a = a.reshape(pages * rows * columns, 1)

    b = a // 128
    a = a - b * 128
    b1 = a // 64
    a = a - b1 * 64
    b2 = a // 32
    a = a - b2 * 32
    b3 = a // 16
    a = a - b3 * 16
    b4 = a // 8
    a = a - b4 * 8
    b5 = a // 4
    a = a - b5 * 4
    b6 = a // 2
    a = a - b6 * 2
    c = np.concatenate((b, b1, b2, b3, b4, b5, b6, a), axis=1)
    return c


def bin2dec(tem13):
    tem11 = np.array([[128], [64], [32], [16], [8], [4], [2], [1]])
    tem12 = np.dot(tem13, tem11)
    return tem12


def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x = x.reshape(-1, )
    x_value_list = set([x[i] for i in range(x.shape[0])])
    # for i in range(x.shape[0])：
    #    x_value_list = set(x[i])

    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]  # x[x == x_value].shape[0] 求取x中等于x_value的个数
        logp = np.log2(p)
        ent -= p * logp

    return ent


def information_entropy(image):
    B = image[0, :, :]
    G = image[1, :, :]
    R = image[2, :, :]
    B_ENT = calc_ent(B)
    G_ENT = calc_ent(G)
    R_ENT = calc_ent(R)
    T1 = np.array([B_ENT, G_ENT, R_ENT])
    T = np.array(T1.argsort(), int)

    return T


def imgssim(img1, img2):
    if len(img1.shape) == 3:
        ssim1 = SSIM(img1[0, :, :], img2[0, :, :])
        ssim2 = SSIM(img1[1, :, :], img2[1, :, :])
        ssim3 = SSIM(img1[2, :, :], img2[2, :, :])
        ssim = (ssim1 + ssim2 + ssim3) / 3
    else:
        ssim = SSIM(img1, img2)
    return ssim


def mssim(im1, im2, win):
    """
    :param im1: 第一张图像(输入为长宽高形式)
    :param im2: 第二张图像
    :param win: 窗口大小
    :return: 平均结构相似性
    """
    c1 = 0.0001
    c2 = 0.0009
    if len(im1.shape) == 3:
        m, n, p = im1.shape
    else:
        im1 = np.expand_dims(im1, axis=2)
        im2 = np.expand_dims(im2, axis=2)
        m, n, p = im1.shape

    sz = math.floor(win / 2)

    mer = []
    for k in range(p):
        for i in range(sz, m - sz):
            for j in range(sz, n - sz):
                six = im1[i - sz:i + sz, j - sz:j + sz, k].ravel()
                siy = im2[i - sz:i + sz, j - sz:j + sz, k].ravel()
                meux = np.mean(six)
                meuy = np.mean(siy)
                sigx = np.std(six)
                sigy = np.std(siy)
                sigxy = np.sum((six - meux) * (siy - meuy)) / (six.shape[0] - 1)
                er = ((2 * meux * meuy + c1) * (2 * sigxy + c2)) / (
                            (meux ** 2 + meuy ** 2 + c1) * (sigx ** 2 + sigy ** 2 + c2))
                mer.append(er)

    return sum(mer) / (len(mer))


def im_and_time_sha256(im, sec_key):
    # 当前时间与输入图像的哈希值
    # 动态密钥产生
    time1 = np.array(list(time.localtime(time.time())))
    # time1 = np.array([2021, 11, 30, 9, 15, 21, 1, 334, 0])

    H1 = hashlib.sha256(im.transpose([1, 2, 0])).hexdigest()
    H2 = hashlib.sha256(time1).hexdigest()
    H = np.zeros([32, ])
    for i in range(32):
        t1 = eval('0x' + H1[2 * i:2 * (i + 1)])
        t2 = eval('0x' + H2[2 * i:2 * (i + 1)])
        H[i] = int(t1) ^ int(t2)
    H = np.array(H, np.uint8)
    h = np.sum(H) / 8192
    h2 = 0
    for i in range(32):
        h2 = h2 ^ H[i]
    h2 = h2 / 255

    x0 = math.sin(h * math.pi * (sec_key[0] + h2))
    y0 = math.sin(h * math.pi * (sec_key[1] + h2))
    a = h / h2 * 10 + sec_key[2]
    b = (h / h2 * 2 + sec_key[3]) % 2

    return [x0, y0, a, b]


def choose_plaintext(im, sec_key, time1):
    # 当前时间与输入图像的哈希值
    # 动态密钥产生
    # time1 = np.array(list(time.localtime(time.time())))
    # time1 = np.array([2021, 11, 30, 9, 15, 21, 1, 334, 0])

    H1 = hashlib.sha256(im.transpose([1, 2, 0])).hexdigest()
    H2 = hashlib.sha256(time1).hexdigest()
    H = np.zeros([32, ])
    for i in range(32):
        t1 = eval('0x' + H1[2 * i:2 * (i + 1)])
        t2 = eval('0x' + H2[2 * i:2 * (i + 1)])
        H[i] = int(t1) ^ int(t2)
    H = np.array(H, np.uint8)
    h = np.sum(H) / 8192
    h2 = 0
    for i in range(32):
        h2 = h2 ^ H[i]
    h2 = h2 / 255

    x0 = math.sin(h * math.pi * (sec_key[0] + h2))
    y0 = math.sin(h * math.pi * (sec_key[1] + h2))
    a = h / h2 * 10 + sec_key[2]
    b = (h / h2 * 2 + sec_key[3]) % 2

    return H1, H2, [x0, y0, a, b]


def encry(plaintext, T1, V1, V2, cover):
    time2222 = time.time()
    [hight, rows, columns] = cover.shape
    hight, rows, columns = hight, int(rows / 2), int(columns / 2)

    T1 = np.floor(np.mod(T1[0: rows + columns] * 10E12, 8 * hight))  # 高置乱
    V1 = np.floor(np.mod(V1[0: 8 * hight * rows] * 10E12, columns))  # 每页行置乱
    V2 = np.floor(np.mod(V2[0: 8 * hight * columns] * 10E12, rows))  # 每页行置乱

    bit_plaint = np.reshape(dec2bin(plaintext), [8 * hight, rows, columns])
    tem10 = np.zeros([8 * hight, rows, columns], int)
    for i in range(rows):
        tem10[:, i, :] = np.roll(bit_plaint[:, i, :], int(T1[i]), axis=0)
    for i in range(columns):
        tem10[:, :, i] = np.roll(tem10[:, :, i], int(T1[i + rows]), axis=0)

    for i in range(8 * hight):
        for j in range(rows):
            tem10[i, j, :] = np.roll(tem10[i, j, :], int(V1[i * rows + j]))
        for k in range(columns):
            tem10[i, :, k] = np.roll(tem10[i, :, k], int(V2[i * columns + k]))

    TCIP = np.reshape(bin2dec(np.reshape(tem10, [hight * rows * columns, 8])), [hight, rows, columns])
    tcip = np.reshape(tem10, [hight * 2 * rows * 2 * columns, 2])
    bit_cover = dec2bin(cover)
    print(time.time() - time2222)
    time333 = time.time()
    cip_bit = np.concatenate((bit_cover[:, 0:6], tcip ^ bit_cover[:, 6:8]), axis=1)
    cip = np.reshape(bin2dec(cip_bit), [hight, 2 * rows, 2 * columns])

    cip1 = np.copy(cip).astype(int)
    cip1[cip - cover == 3] = cip[cip - cover == 3] - 4
    cip1[cip - cover == -3] = cip[cip - cover == -3] + 4
    cip1[cip1 > 255] = cip[cip1 > 255]
    cip1[cip1 < 0] = cip[cip1 < 0]

    print('嵌入时间', time.time() - time333)

    return TCIP, cip1


def dencry(cip, T1, V1, V2, cover):
    time2222 = time.time()

    [hight, rows, columns] = cip.shape
    hight, rows, columns = hight, int(rows / 2), int(columns / 2)

    T1 = -np.floor(np.mod(T1[0: rows + columns] * 10E12, 8 * hight))  # 高置乱
    V1 = -np.floor(np.mod(V1[0: 8 * hight * rows] * 10E12, columns))  # 每页行置乱
    V2 = -np.floor(np.mod(V2[0: 8 * hight * columns] * 10E12, rows))  # 每页行置乱

    time2222 = time.time()
    bit_cip = dec2bin(cip)
    bit_cover = dec2bin(cover)
    tcip_bit = bit_cip[:, 6:8] ^ bit_cover[:, 6:8]
    tcip_bit = np.reshape(tcip_bit, [8 * hight, rows, columns])
    print('提取时间', time.time() - time2222)

    tem10 = np.zeros([8 * hight, rows, columns], int)
    for i in range(8 * hight):
        for k in range(columns):
            tem10[i, :, k] = np.roll(tcip_bit[i, :, k], int(V2[i * columns + k]))
        for j in range(rows):
            tem10[i, j, :] = np.roll(tem10[i, j, :], int(V1[i * rows + j]))

    for i in range(columns):
        tem10[:, :, i] = np.roll(tem10[:, :, i], int(T1[i + rows]), axis=0)
    for i in range(rows):
        tem10[:, i, :] = np.roll(tem10[:, i, :], int(T1[i]), axis=0)

    tem10 = np.reshape(tem10, [hight * rows * columns, 8])
    rim = np.reshape(bin2dec(tem10), [hight, rows, columns])

    return rim


def encryption(img, sec_key, cover, model):
    time111 = time.time()
    [hight, rows, columns] = cover.shape
    hight, rows, columns = hight, int(rows / 2), int(columns / 2)
    # 动态密钥产生
    dynamic_key = im_and_time_sha256(img, sec_key)
    # 伪随机序列
    ch_x, ch_y = LSM(dynamic_key[0:2], dynamic_key[2:4], N=24 * (rows + columns))
    T1 = ch_x
    V1 = ch_y[0:24 * rows]
    V2 = ch_y[24 * rows:24 * (rows + columns)]
    # 压缩
    model.eval()
    sort = information_entropy(img)
    img = np.concatenate(
        (img[sort[0]:sort[0] + 1, :, :], img[sort[1]:sort[1] + 1, :, :], img[sort[2]:sort[2] + 1, :, :]), axis=0)

    date = np.expand_dims(img / 255.0, axis=0)
    date = torch.Tensor(date)
    with torch.no_grad():
        compress = model.compress(date)
        compress = torch.relu(compress)
    compress_max, compress_min = torch.max(compress), torch.min(compress)
    compress = torch.round((compress - compress_min) / (compress_max - compress_min) * 255)
    compress = np.squeeze(np.array(compress))
    print(time.time() - time111)
    # # 加密
    TCIP, cip = encry(compress, T1, V1, V2, cover)
    return np.array(cip, np.uint8), np.array(TCIP, np.uint8), compress_max, compress_min, sort, dynamic_key


def dencryption(cip, dynamic_key, cover, compress_max, compress_min, sort, model):
    time111 = time.time()
    [hight, rows, columns] = cover.shape
    hight, rows, columns = hight, int(rows / 2), int(columns / 2)
    # 伪随机序列
    ch_x, ch_y = LSM(dynamic_key[0:2], dynamic_key[2:4], N=24 * (rows + columns))
    T1 = ch_x
    V1 = ch_y[0:24 * rows]
    V2 = ch_y[24 * rows:24 * (rows + columns)]
    # 解密
    compress = dencry(cip, T1, V1, V2, cover)

    # 重构
    model.eval()
    compress = (compress / 255.0) * np.array(compress_max - compress_min) + np.array(compress_min)
    date = np.expand_dims(compress, axis=0)

    date = torch.Tensor(date)
    with torch.no_grad():
        rim1 = model.recons(date)
        rim2 = model.se_recons1(rim1)

        com1 = date - torch.relu(model.compress(rim2))
        rim3 = model.recons(com1) + rim2
        rim4 = model.se_recons2(rim3)

        # com2 = squat - torch.relu(model.compress(rim4))
        # rim5 = model.recons(com2) + rim4
        # rim6 = model.se_recons3(rim5)

        # rim7 = torch.cat((rim2, rim4, rim6), dim=1)
        # output2 = model.thr_recons(rim7)

    output2 = np.array(torch.clip(rim4, 0, 1).cpu()).squeeze() * 255

    rim = np.copy(output2)
    rim[sort[0]], rim[sort[1]], rim[sort[2]] = output2[0], output2[1], output2[2]

    print(time.time() - time111)
    return np.array(rim, np.uint8)


if __name__ == '__main__':
    model = MyCSNet(args)
    model.loaddict(root='model/vel')

    plaintext = cv.imread("D:\python\color com\com_and_encry\image\pepper.bmp")  # 读取图片
    cover = cv.imread("D:\python\date\Encryption_standard_image\misc/4.2.06.tiff")


    m1, n1, k1 = plaintext.shape
    print(plaintext.shape)
    plaintext = np.transpose(plaintext, [2, 0, 1])
    cover = np.transpose(cover, [2, 0, 1])
    sec_key = [0.1, 0.1, 21, 0.75]
    time1 = time.time()
    cip, tcip, compress_max, compress_min, sort, dynamic_key = encryption(plaintext, sec_key, cover, model)
    rim = dencryption(cip, dynamic_key, cover, compress_max, compress_min, sort, model)
    print(time.time() - time1)

    cv.imshow('im', plaintext.transpose([1, 2, 0]))
    cv.imshow('cover', cover.transpose([1, 2, 0]))
    cv.imshow('tcip', tcip.transpose([1, 2, 0]))
    cv.imshow('cip', cip.transpose([1, 2, 0]))
    cv.imshow('rim', rim.transpose([1, 2, 0]))

    ## 测试加密解密效果（ssim与psnr）
    # """"
    psnr_cip = PSNR(cip / 255.0, cover / 255.0)
    mssim_cip = mssim(cip.transpose([1, 2, 0]), cover.transpose([1, 2, 0]), m1-8)  ####
    ssim_cip = imgssim(cip, cover)
    psnr_rim = PSNR(rim / 255.0, plaintext / 255.0)
    mssim_rim = mssim(rim.transpose([1, 2, 0]), plaintext.transpose([1, 2, 0]), m1-8)  ####
    ssim_rim = imgssim(rim, plaintext)
    print('psnr_cip', 'mssim_cip', 'ssim_cip', psnr_cip, mssim_cip, ssim_cip)
    print('psnr_rim', 'mssim_rim', 'ssim_rim', psnr_rim, mssim_rim, ssim_rim)
    # """


    plt.show()
    cv.waitKey(0)








