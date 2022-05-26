import torch
import torch.nn.functional as torch_F
import math

class wrap:
    class Conv2D_Norm_Act(torch.nn.Module):
        def __init__(self, in_channels:int, out_channels:int, kernel_size:tuple, activate:bool=True):
            super(wrap.Conv2D_Norm_Act, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.activate = activate

            self.pad = torch.nn.ConstantPad2d((math.floor((self.kernel_size[0]-1)/2), math.ceil((self.kernel_size[0]-1)/2), math.floor((self.kernel_size[1]-1)/2), math.ceil((self.kernel_size[1]-1)/2)), 0.0)
            self.conv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size)
            self.norm = torch.nn.BatchNorm2d(self.out_channels)

        def forward(self, input):
            x = self.pad(input)
            x = self.conv(x)
            x = self.norm(x)
            if self.activate: x = torch_F.relu(x)
            return x

    class Linear_Norm_Drop_Act(torch.nn.Module):
        def __init__(self, in_features:int, out_features:int, bias:bool=True, drop:bool=True, drop_rate:float=0.2, activate:bool=True):
            super(wrap.Linear_Norm_Drop_Act, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            self.drop = drop
            self.drop_rate = drop_rate
            self.activate = activate

            self.linear = torch.nn.Linear(self.in_features, self.out_features, self.bias)
            self.norm = torch.nn.BatchNorm1d(self.out_features)
            if self.drop: self.dropout = torch.nn.Dropout(self.drop_rate)

        def forward(self, input):
            x = self.linear(input)
            x = self.norm(x)
            if self.drop: x = self.dropout(x)
            if self.activate: x = torch_F.relu(x)
            return x

class models:
    class CNN(torch.nn.Module):
        def __init__(self, input_shape:tuple, filter_num:int):
            #Input : (batch_size, length, dim) (None, 28, 28)
            super(models.CNN, self).__init__()
            self.batch, self.length, self.dim = input_shape
            self.filter = filter_num

            self.conv_list = [wrap.Conv2D_Norm_Act(1, self.filter, (10, 10))]
            self.conv_list.append(wrap.Conv2D_Norm_Act(self.filter, self.filter, (6, 6)))
            self.conv_list.append(wrap.Conv2D_Norm_Act(self.filter, self.filter, (4, 4)))
            self.conv = torch.nn.Sequential(*self.conv_list)
            self.maxpool = torch.nn.Sequential(torch.nn.MaxPool2d((2, 2)), torch.nn.BatchNorm2d(self.filter), torch.nn.ReLU())
            self.flatten = torch.nn.Flatten(1)
            self.linear = torch.nn.Sequential(wrap.Linear_Norm_Drop_Act(self.filter*math.floor((self.length-2)/2+1)*math.floor((self.dim-2)/2+1), 200, drop_rate=0.2), wrap.Linear_Norm_Drop_Act(200, 10, drop_rate=0.0))
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, input):
            input = input.reshape((-1, 1, self.length, self.dim))
            x = self.conv(input)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.linear(x)
            p = self.softmax(x)
            output = p.argmax(dim=1)
            return output, p

