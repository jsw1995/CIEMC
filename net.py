import math
import torch.nn as nn
import torch
import torch.nn.functional as F


class net1(nn.Module):

    def __init__(self, out_chnl1 = 1024,out_chnl2 = 1024,out_chnl3 = 1024):

        super(net1, self).__init__()

        self.compress1 = nn.Conv2d(1, out_chnl1, kernel_size=32, stride=32, padding=0, bias=False)
        self.compress2 = nn.Conv2d(1, out_chnl2, kernel_size=32, stride=32, padding=0, bias=False)
        self.compress3 = nn.Conv2d(1, out_chnl3, kernel_size=32, stride=32, padding=0, bias=False)
        self.pixelshuffle1 = nn.PixelShuffle(16)


    def forward(self, image):

        cip0 = self.compress1(image[:, 0:1, :, :])
        cip1 = self.compress2(image[:, 1:2, :, :])
        cip2 = self.compress3(image[:, 2:3, :, :])
        cip = torch.cat((cip0, cip1, cip2),1)

        cip = self.pixelshuffle1(cip)

        return cip


class net2(nn.Module):

    def __init__(self, out_chnl1,out_chnl2,out_chnl3):

        super(net2, self).__init__()

        self.PixelUnshuffle0 = nn.PixelUnshuffle(16)

        self.recon1 = nn.Conv2d(out_chnl1, 1024, 1, 1, 0, bias=False)
        self.recon2 = nn.Conv2d(out_chnl2, 1024, 1, 1, 0, bias=False)
        self.recon3 = nn.Conv2d(out_chnl3, 1024, 1, 1, 0, bias=False)

        self.pixelshuffle1 = nn.PixelShuffle(32)


    def forward(self, image, out_chnl1, out_chnl2, out_chnl3):

        a = out_chnl1
        b = out_chnl1 + out_chnl2
        c = out_chnl1 + out_chnl2 + out_chnl3
        image = self.PixelUnshuffle0(image)
        rim1 = self.recon1(image[:, 0: a, :, :])
        rim1 = self.pixelshuffle1(rim1)
        rim2 = self.recon2(image[:, a: b, :, :])
        rim2 = self.pixelshuffle1(rim2)
        rim3 = self.recon3(image[:, b: c, :, :])
        rim3 = self.pixelshuffle1(rim3)
        rim = torch.cat((rim1, rim2, rim3),1)

        return rim


class net21(nn.Module):

    def __init__(self, out_chnl1,out_chnl2,out_chnl3):

        super(net21, self).__init__()
        self.PixelUnshuffle0 = nn.PixelUnshuffle(16)

        self.recon1 = nn.Conv2d(out_chnl1+out_chnl3, 1024, 1, 1, 0, bias=False)
        self.recon2 = nn.Conv2d(out_chnl2+out_chnl3, 1024, 1, 1, 0, bias=False)
        self.recon3 = nn.Conv2d(out_chnl3, 1024, 1, 1, 0, bias=False)

        self.pixelshuffle1 = nn.PixelShuffle(32)


    def forward(self, image, out_chnl1, out_chnl2, out_chnl3):

        a = out_chnl1
        b = out_chnl1 + out_chnl2
        c = out_chnl1 + out_chnl2 + out_chnl3
        image = self.PixelUnshuffle0(image)
        image1 = torch.cat((image[:, 0: a, :, :],image[:, b: c, :, :]),dim=1)
        image2 = torch.cat((image[:, a: b, :, :],image[:, b: c, :, :]),dim=1)
        image3 = image[:, b: c, :, :]
        rim1 = self.recon1(image1)
        rim1 = self.pixelshuffle1(rim1)
        rim2 = self.recon2(image2)
        rim2 = self.pixelshuffle1(rim2)
        rim3 = self.recon3(image3)
        rim3 = self.pixelshuffle1(rim3)
        rim = torch.cat((rim1, rim2, rim3),1)

        return rim


class net3(nn.Module):

    def __init__(self, ):

        super(net3, self).__init__()
        self.Residual_net1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.Residual_net2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.Residual_net3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
        )



    def forward(self, image):

        cip = self.Residual_net1(image)
        cip = self.Residual_net2(cip)
        cip = self.Residual_net3(cip)

        return cip+image


class net4(nn.Module):

    def __init__(self, input_chnls = 9):

        super(net4, self).__init__()
        self.Residual_net1 = nn.Sequential(
            nn.Conv2d(input_chnls, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.Residual_net3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, image):

        cip = self.Residual_net1(image)
        cip = self.Residual_net3(cip)

        return cip


if __name__ == '__main__':

    a = torch.rand((2,3,128,128))

    model1 = net1(128,128,512)
    model2 = net2(128,128,512)
    model3 = net3()
    model4 = net3()
    model5 = net3()
    model6 = net4(9)

    compress = model1(a)
    rim1 = model2(compress,128,128,512)
    rim2 = model3(rim1)
    compress1 = compress-model1(rim2)
    rim3 = model2(compress1,128,128,512) + rim2
    rim4 = model4(rim3)
    compress2 = compress - model1(rim4)
    rim5 = model2(compress2,128,128,512) + rim4
    rim6 = model3(rim5)
    rim7 = torch.cat((rim2,rim4,rim6),dim=1)
    rim = model6(rim7)

    print(rim.shape)