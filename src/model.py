import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        return self.relu(out)
    
    
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super().__init__()
        self.globpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(channel, channel // reduction_ratio)
        self.fc2 = nn.Linear(channel // reduction_ratio, channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        f = self.globpool(x)
        f = f.view(f.size(0), -1)  # Flatten operation
        f = self.relu(self.fc1(f))
        f = self.sigmoid(self.fc2(f))
        f = f.view(f.size(0), f.size(1), 1, 1)  # Reshape back to match input dimensions
        return x * f
    
    
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MyModel(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.7):
        super(MyModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, padding=3, stride=2),  # Initial downsample with stride 2
            nn.MaxPool2d(3, 2, padding=1),  # Additional downsample
            BasicBlock(64, 64), 
            BasicBlock(64, 64),
            nn.MaxPool2d(2, 2),
            
            BasicBlock(64, 128, downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=1),
                nn.BatchNorm2d(128))),
            BasicBlock(128, 128),
            nn.MaxPool2d(2, 2),
            SEBlock(128),
            
            BasicBlock(128, 256, downsample=nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256))),
            BasicBlock(256, 256),
            nn.MaxPool2d(2, 2),
            
            SEBlock(256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model.forward(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
