import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
    
        x = x.view(-1, 32 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class BasicResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(BasicResnetBlock, self).__init__()
        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3), stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3,3), stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        if stride!=1 or in_channels!=out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, (1,1), stride=stride)

    def forward(self, x):

        if self.stride!=1 or self.in_channels != self.out_channels:
            x = self.ReLU(self.conv3(x) + self.bnorm2(self.conv2(self.ReLU(self.bnorm1(self.conv1(x))))))
        else:
            x = self.ReLU(x + self.bnorm2(self.conv2(self.ReLU(self.bnorm1(self.conv1(x))))))
        return x
class resnet18(nn.Module):

    def __init__(self, num_classes=10):
        super(resnet18, self).__init__()
        self.conv_input = nn.Conv2d(3, 64, (7,7), stride=2, padding=3)
        self.pool1 = nn.MaxPool2d((2,2), stride=2)
        self.block1 = BasicResnetBlock(64, 64, stride=1)
        self.block2 = BasicResnetBlock(64, 64, stride=1)
        self.block3 = BasicResnetBlock(64, 128, stride=2)
        self.block4 = BasicResnetBlock(128, 128, stride=1)
        self.block5 = BasicResnetBlock(128, 256, stride=2)
        self.block6 = BasicResnetBlock(256, 256, stride=1)
        self.block7 = BasicResnetBlock(256, 512, stride=2)
        self.block8 = BasicResnetBlock(512, 512, stride=1)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # Calculate the correct input features for FC
        self._fc_in_features = 512 * 1 * 1
        self.FC = nn.Linear(self._fc_in_features, num_classes)
    
    def forward(self, x):
        x = self.conv_input(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.FC(x)
        return x