import torch
import torch.nn as nn

# 1,3,580,770
batch_size = 25
class ConvRNN(nn.Module):
    def __init__(self, num_classes):
        super(ConvRNN, self).__init__()
        
        # Activation functions sgmoid and LeakyRelu
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()

        # Convolutional layers
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)        # Convolutional layer 1
        self.pool_1 = nn.MaxPool2d((2, 2), stride=2)                              # Max-pool layer 1 
        self.bn_1 = nn.BatchNorm2d(64)
        
        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)       # Convolutional layer 2
        
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)      # Convolutional layer 3
        self.bn_3 = nn.BatchNorm2d(256)                                            # Batch normalization layer
        self.pool_3 = nn.MaxPool2d((2, 2), stride=2)                               # Max-pool layer 2
        
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=2, padding=1, stride=1)      # Convolutional layer 4
        
        
        # Dense layer                                  
        self.dense1 = nn.Linear(7213568, 64) `# 256*256*3                          # 7213568 will be the output of the last conv layer                

        # Recurrent layer
        self.rnn_1 = nn.LSTM(64, 128, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
                
        # Output layer
        self.dense2 = nn.Linear(128*2,25)
        
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool_1(x)
        x = self.bn_1(x)

        x = self.conv_2(x)
        x = self.relu(x)
        
        x = self.conv_3(x)
        x = self.relu(x)
        x = self.pool_3(x)
        x = self.bn_3(x)
        
        x = self.conv_4(x)
        x = self.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.relu(x)

        x, _ = self.rnn_1(x)
        x = self.dense2(x)

        return x

# DEBUGGING
batch_size = 1
input = torch.randn(batch_size, 3, 580, 770)