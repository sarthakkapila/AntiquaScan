import torch
import torch.nn as nn

# 1,3,580,770
batch_size = 25
class ConvRNN(nn.Module):
    def __init__(self, num_classes):
        super(ConvRNN, self).__init__()
        
        # Activation function
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()

        # Convolutional layers
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)        # Convolutional layer 1
        self.pool_1 = nn.MaxPool2d((2, 2), stride=2)                              # Max-pool layer 1 
        self.bn_1 = nn.BatchNorm2d(64)
        
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)       # Convolutional layer 2
        
        self.conv_6 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)      # Convolutional layer 3
        self.bn_6 = nn.BatchNorm2d(256)                                            # Batch normalization layer
        self.pool_6 = nn.MaxPool2d((2, 2), stride=2)                               # Max-pool layer 2
        
        self.conv_7 = nn.Conv2d(256, 256, kernel_size=2, padding=1, stride=1)       # Convolutional layer 4
        
        
        # Recurrent layers
        self.dense1 = nn.Linear(7213568, 64)

#         RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x256 and 128x256)

        self.rnn_1 = nn.LSTM(64, 128, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
#         self.rnn_2 = nn.LSTM(256, 64, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
        
        # Output layer
        self.dense2 = nn.Linear(128*2,25)
        


    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool_1(x)
        x = self.bn_1(x)


#         x = self.conv_2(x)
#         x = self.relu(x)
#         x = self.pool_2(x)

        x = self.conv_3(x)
        x = self.relu(x)

#         x = self.conv_4(x)
#         x = self.relu(x)
#         x = self.pool_4(x)

#         x = self.conv_5(x)
#         x = self.relu(x)
#         x = self.bn_5(x)

        x = self.conv_6(x)
        x = self.relu(x)
        x = self.pool_6(x)
        x = self.bn_6(x)
        
        x = self.conv_7(x)
#         print(x.shape)
        x = self.relu(x)
#         print(x.shape)

# # SQUEEZE
#         print(x.shape, "1")
#         x_squeezed = torch.squeeze(x)
#         print(x_squeezed.shape, "2")
#         x = x_squeezed
#         x = x.permute(0, 2, 1)

#         print(x.shape, "3")
        x = torch.flatten(x, 1)
#         print(x.shape)
        x = self.dense1(x)
#         print(x.shape)
        x = self.relu(x)

        x, _ = self.rnn_1(x)
#         x, _ = self.rnn_2(x)

        x = self.dense2(x)
#         x = x.view(-1, 1750, max_length)
#         x = torch.squeeze(x, dim=0)

        return x
