from torch import nn

# Model definition
class ConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.convr = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1)
    def forward(self, input):
        residual = self.convr(input)
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu3(x)
        return x

class IdentityBlock(nn.Module):
    def __init__(self, in_channels):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
    def forward(self, input):
        
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu3(x)
        return x

class ResNet24(nn.Module):
    def __init__(self):
        super(ResNet24, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=9, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(2)
        
        self.convblk1 = ConvBlock(64)
        self.convblk2 = ConvBlock(64)
        self.identityblk1 = IdentityBlock(64)
        self.convblk3 = ConvBlock(64)
        self.identityblk2 = IdentityBlock(64)
        self.convblk4 = ConvBlock(64)
        self.identityblk3 = IdentityBlock(64)
        
        self.pool2 = nn.AvgPool1d(2)
        
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(in_features=704, out_features=2),
        #                        nn.Softmax()
                                )
    
    def forward(self, x):
        x = x.transpose(1, 2)  # Transpose to have the correct dimensions for Conv1d (batch, channels, length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.convblk1(x)
        x = self.convblk2(x)
        x = self.identityblk1(x)
        x = self.convblk3(x)
        x = self.identityblk2(x)
        x = self.convblk4(x)
        x = self.identityblk3(x)
        x = self.pool2(x)
        x = self.fc(x)
        return x