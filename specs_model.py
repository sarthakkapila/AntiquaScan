from model import ConvRNN

batch_size = 1
input = torch.randn(batch_size, 3, 580, 770)
model = ConvRNN(input)
print(model)

""" ConvRNN(
  (relu): LeakyReLU(negative_slope=0.2, inplace=True)
  (sig): Sigmoid()
  (conv_1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool_1): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (bn_1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv_3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv_6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn_6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool_6): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv_7): Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
  (dense1): Linear(in_features=7213568, out_features=64, bias=True)
  (rnn_1): LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
  (dense2): Linear(in_features=256, out_features=25, bias=True)
) """