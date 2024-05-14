# AntiqueScan

AntiqueScan is a OCR model implemented using PyTorch specifically for manuscripts and old texts.

## Overview

The CRNN architecture consists of convolutional layers for extracting features from input images followed by recurrent layers for processing sequential text data. The architecture is designed to handle image-text tasks such as image captioning, where the goal is to generate a textual description of an input image.

## Dependencies

- Python 3
- PyTorch
- Torchvision
- Matplotlib
- Pillow (PIL)

## Dataset
![Screenshot 2024-05-14 at 11 00 51 PM](https://github.com/sarthakkapila/AntiquaScan/assets/112886451/b98f64ac-18b9-4bbb-b90d-6236ba37629e)

The dataset consists of 31 pages of a scanned early modern printed text. The dataset also includes a transcription of 25 pages. The last 6 pages of transcriptions have been used as test.

Each image file should have a corresponding text file containing the textual description of the image.

## Model Architecture

The CRNN model architecture consists of the following components:

- Activation Functions:
  - ReLU: LeakyReLU(negative_slope=0.2, inplace=True)
  - Sigmoid

- Convolutional Layers:
  - Conv1: Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - Conv3: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - Conv6: Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  - Conv7: Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))

- Pooling Layers:
  - Pool1: MaxPool2d(kernel_size=(2, 2), stride=2)
  - Pool6: MaxPool2d(kernel_size=(2, 2), stride=2)

- Batch Normalization Layers:
  - BatchNorm1: BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  - BatchNorm6: BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

- Fully Connected Layers:
  - Dense1: Linear(in_features=7213568, out_features=64, bias=True)
  - Dense2: Linear(in_features=256, out_features=25, bias=True)

- Recurrent Layers:
  - RNN1: LSTM(64, 128, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)

## License

This project is licensed under the MIT License.
