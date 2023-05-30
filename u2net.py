import torch
import torch.nn as nn
import torch.nn.functional as F



def upsample(src, tar):
    src = F.upsample(src, size= tar.shape[2:], mode='bilinear')
    return(src)
'''In U2NET we save params by upsampling instead of convTranspose2d '''

class RBC(nn.Module):
    def __init__(self, in_ch, out_ch, dirate = 1):
        super(RBC, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, padding = 1*dirate, dilation = 1*dirate)
        self.ReLu = nn.ReLU(True)
        self.batch_norm = nn.BatchNorm2d(out_ch)
    def forward(self,x):
        hx = x
        x_out = self.ReLu(self.batch_norm(self.conv1(x)))
        return x_out
    

class Block1(nn.Module):
    def __init__(self):
        super(Block1, self).__init__()
        self.convin = RBC(1, 64)
        self.rbc1 = RBC(64, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64,32)
        self.rbc1d = RBC(64,64)
    
    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)


        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx5h = self.maxpool56(hx5)

        hx6 = self.rbc6(hx5h)
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))

        hx = upsample(hx5d, hx4)
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))

        hx = upsample(hx4d, hx3)
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))

        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))

        hx = upsample(hx2d, hx1)
        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))
        
        return hx1d + hxin
    
class Block2(nn.Module):
    def __init__(self):
        super(Block2, self).__init__()
        self.rbcin = RBC(64, 128)
        self.rbc1 = RBC(128, 64)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(64,64)
        self.maxpoool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True)
        self.rbc3 = RBC(64, 64)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(64, 64)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode = True)
        self.rbc5 = RBC(64, 64)
        self.rbc6 = RBC(64, 64, dirate = 2)

        self.rbc5d = RBC(128, 64)
        self.rbc4d = RBC(128, 64)
        self.rbc3d = RBC(128, 64)
        self.rbc2d = RBC(128, 64)
        self.rbc1d = RBC(128, 128)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpoool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx6 = self.rbc6(hx5)

        hx5d = self.rbc5d(torch.cat((hx5, hx6), 1))
        hx = upsample(hx5d, hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)

        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))

        return hxin + hx1d

class Block3(nn.Module):
    def __init__(self):
        super(Block3, self).__init__()
        self.rbcin = RBC(128, 256)
        self.rbc1 = RBC(256, 128, dirate = 2)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(128, 128, dirate = 2)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(128, 128, dirate = 2)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(128, 128, dirate = 2)
        self.rbc5 = RBC(128, 128, dirate = 2)

        self.rbc4d = RBC(256, 128, dirate =2)
        self.rbc3d = RBC(256, 128, dirate =2)
        self.rbc2d = RBC(256, 128, dirate =2)
        self.rbc1d = RBC(256, 256, dirate =2)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)
        
        hx4 = self.rbc4(hx3h)
        hx5 = self.rbc5(hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx5), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)

        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))

        return hxin + hx1d

class Block4(nn.Module):
    def __init__(self):
        super(Block4, self).__init__()
        self.rbcin = RBC(256, 512)
        self.rbc1 = RBC(512, 256, dirate=3)
        self.maxpol12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(256, 256, dirate= 3)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(256, 256, dirate=3)
        self.rbc4 = RBC(256, 256, dirate=3)

        self.rbc3d = RBC(512, 256, dirate = 3)
        self.rbc2d = RBC(512, 256, dirate = 3)
        self.rbc1d = RBC(512, 512, dirate = 3)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpol12(hx1)
        hx2  = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)
        hx3 = self.rbc3(hx2h)
        hx4 = self.rbc4(hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx4), 1))
        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx1d = self.rbc1d(torch.cat((hx1, hx), 1))

        return hxin + hx1d
    
class Block5(nn.Module):
    def __init__(self):
        super(Block5, self).__init__()
        self.conv1 = nn.Conv2d(512, 1024, kernel_size = 3, padding = 1)
        self.ReLu = nn.ReLU(True)
        self.batch_norm1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size = 3, padding =1)
        self.batch_norm2 = nn.BatchNorm2d(512)
    
    def forward(self, x):
        hx = x
        x_out = self.batch_norm2(self.conv2(self.ReLu(self.batch_norm1(self.conv1(hx)))))
        x_model = self.ReLu(x_out)

        return {'model_input': x_model, 'model_output':x_model}
    
class Block6(nn.Module):
    def __init__(self):
        super(Block6, self).__init__()
        self.rbcin = RBC(1024, 256)
        self.rbc1 = RBC(256, 256, dirate=3)
        self.maxpol12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(256, 256, dirate= 3)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(256, 256, dirate=3)
        self.rbc4 = RBC(256, 256, dirate=3)

        self.rbc3d = RBC(512, 256, dirate = 3)
        self.rbc2d = RBC(512, 256, dirate = 3)
        self.c1d = nn.Conv2d(512, 256, kernel_size = 3, padding =3, dilation = 3)
        self.b1d = nn.BatchNorm2d(256)
        self.ReLu = nn.ReLU(True)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpol12(hx1)
        hx2  = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)
        hx3 = self.rbc3(hx2h)
        hx4 = self.rbc4(hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx4), 1))
        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx1d_output = self.b1d(self.c1d(torch.cat((hx1, hx), 1)))
        hx1d_input = self.ReLu(hx1d_output)

        return {'model_input': hx1d_input, 'model_output':hx1d_output}

        
class Block7(nn.Module):
    def __init__(self):
        super(Block7, self).__init__()
        self.rbcin = RBC(512, 128)
        self.rbc1 = RBC(128, 128, dirate = 2)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc2 = RBC(128, 128, dirate = 2)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc3 = RBC(128, 128, dirate = 2)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(128, 128, dirate = 2)
        self.rbc5 = RBC(128, 128, dirate = 2)

        self.rbc4d = RBC(256, 128, dirate =2)
        self.rbc3d = RBC(256, 128, dirate =2)
        self.rbc2d = RBC(256, 128, dirate =2)
        self.c1d = nn.Conv2d(256, 128, kernel_size = 3, padding =2, dilation = 2)
        self.b1d = nn.BatchNorm2d(128)
        self.ReLu = nn.ReLU(True)

    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)
        
        hx4 = self.rbc4(hx3h)
        hx5 = self.rbc5(hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx5), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        hx1d_output = self.b1d(self.c1d(torch.cat((hx1, hx), 1)))
        hx1d_input = self.ReLu(hx1d_output)

        return {'model_input': hx1d_input, 'model_output':hx1d_output}
    

class Block8(nn.Module):
    def __init__(self):
        super(Block8, self).__init__()
        self.rbcin = RBC(256, 64)
        self.rbc1 = RBC(64, 64)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(64,64)
        self.maxpoool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True)
        self.rbc3 = RBC(64, 64)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride= 2, ceil_mode = True)
        self.rbc4 = RBC(64, 64)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode = True)
        self.rbc5 = RBC(64, 64)
        self.rbc6 = RBC(64, 64, dirate = 2)

        self.rbc5d = RBC(128, 64)
        self.rbc4d = RBC(128, 64)
        self.rbc3d = RBC(128, 64)
        self.rbc2d = RBC(128, 64)
        self.c1d = nn.Conv2d(128, 64, kernel_size = 3)
        self.b1d = nn.BatchNorm2d(64)
        self.ReLu = nn.ReLU(True)


    def forward(self, x):
        hx = x
        hxin = self.rbcin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)

        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpoool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx6 = self.rbc6(hx5)

        hx5d = self.rbc5d(torch.cat((hx5, hx6), 1))
        hx = upsample(hx5d, hx4)

        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))
        hx = upsample(hx4d, hx3)

        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))
        hx = upsample(hx3d, hx2)

        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))
        hx = upsample(hx2d, hx1)
        
        hx1d_output = self.b1d(self.c1d(torch.cat((hx1, hx), 1)))
        hx1d_input = self.ReLu(hx1d_output)

        return {'model_input': hx1d_input, 'model_output':hx1d_output}
    

class Block9(nn.Module):
    def __init__(self):
        super(Block9, self).__init__()
        self.convin = RBC(128, 3)
        self.rbc1 = RBC(3, 32)
        self.maxpool12 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc2 = RBC(32, 32)
        self.maxpool23 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc3 = RBC(32, 32)
        self.maxpool34 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc4 = RBC(32,32)
        self.maxpool45 = nn.MaxPool2d(kernel_size = 2, stride =2, ceil_mode=True)
        self.rbc5 = RBC(32,32)
        self.maxpool56 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        self.rbc6 = RBC(32, 32)
        self.rbc7 = RBC(32,32, dirate = 2)

        self.rbc6d = RBC(64,32)
        self.rbc5d = RBC(64,32)
        self.rbc4d = RBC(64,32)
        self.rbc3d = RBC(64,32)
        self.rbc2d = RBC(64,32)
        self.c1d = nn.Conv2d(64, 3, kernel_size = 3)
        self.b1d = nn.BatchNorm2d(3)
        self.ReLu = nn.ReLU(True)
        
    
    def forward(self, x):
        hx = x
        hxin = self.convin(hx)
        hx1 = self.rbc1(hxin)
        hx1h = self.maxpool12(hx1)


        hx2 = self.rbc2(hx1h)
        hx2h = self.maxpool23(hx2)

        hx3 = self.rbc3(hx2h)
        hx3h = self.maxpool34(hx3)

        hx4 = self.rbc4(hx3h)
        hx4h = self.maxpool45(hx4)

        hx5 = self.rbc5(hx4h)
        hx5h = self.maxpool56(hx5)

        hx6 = self.rbc6(hx5h)
        hx7 = self.rbc7(hx6)
        
        hx6d = self.rbc6d(torch.cat((hx6, hx7),1))
        hx = upsample(hx6d, hx5)
        hx5d = self.rbc5d(torch.cat((hx5, hx), 1))

        hx = upsample(hx5d, hx4)
        hx4d = self.rbc4d(torch.cat((hx4, hx), 1))

        hx = upsample(hx4d, hx3)
        hx3d = self.rbc3d(torch.cat((hx3, hx), 1))

        hx = upsample(hx3d, hx2)
        hx2d = self.rbc2d(torch.cat((hx2, hx), 1))

        hx = upsample(hx2d, hx1)
        hx1d_output = self.b1d(self.c1d(torch.cat((hx1, hx), 1)))
        hx1d_input = self.ReLu(hx1d_output)

        return {'model_input': hx1d_input, 'model_output':hx1d_output}
        
     
    

class U2NET(nn.module):
    def __init__(self):
        super(U2NET, self).__init__()
        maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode=True)
        block1 = Block1() 
        block2 = Block2()
        block3 = Block3()
        block4 = Block4()
        block5 = Block5()
        block6 = Block6()
        block7 = Block7()
        block8 = Block8()
        block9 = Block9()



