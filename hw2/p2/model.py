import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=4*4*128, out_features=512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        conv_x1 = self.conv1(x)              
        bn_x1 = self.bn1(conv_x1)             
        relu_x1 = self.relu1(bn_x1)           
        pool_x1 = self.pool1(relu_x1)  
        conv_x2 = self.conv2(pool_x1)              
        bn_x2 = self.bn2(conv_x2)             
        relu_x2 = self.relu2(bn_x2)           
        pool_x2 = self.pool2(relu_x2)  
        conv_x3 = self.conv3(pool_x2)              
        bn_x3 = self.bn3(conv_x3)             
        relu_x3 = self.relu3(bn_x3)           
        pool_x3 = self.pool3(relu_x3)  
        x = pool_x3.view(-1, 4*4*128)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)
        


    def forward(self, x):
        conv_x1 = self.conv1(x)              
        bn_x1 = self.bn1(conv_x1)             
        relu_x1 = self.relu1(bn_x1)           
        pool_x1 = self.pool(relu_x1)  

        x = self.resnet.layer1(pool_x1)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
