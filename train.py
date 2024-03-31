import torch
import torch.nn as nn
import torch.optim as optim

from model import ConvRNN
from text_to_tensor import text_to_char_ids
from laoding import dataloader

max_epochs = 11
losses = []
accuracies = []

label_tensor = []

for epoch in range(max_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    conv_rnn_model.train()
    
    for images, labels in dataloader:
        label_tensor = []
        labels = ''.join(labels)
#         labels = labels.replace('\n', '') 

        char_ids = text_to_char_ids(labels, char_to_id, max_length)
        label_tensor.append(char_ids)

        label_tensor = torch.tensor(label_tensor, dtype=torch.float)
        print(label_tensor, "\nAs tensor\n")       

        optimizer.zero_grad()

        outputs = conv_rnn_model(images)
        print(outputs.shape)
        print(label_tensor.shape)

        loss = criterion(outputs, label_tensor)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 0)
        total += label_tensor.size(0)
        correct += (predicted == label_tensor).sum().item()

        epoch_loss += loss.item()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{max_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

epoch_loss /= len(dataloader)
accuracy = correct / total
losses.append(epoch_loss)
accuracies.append(accuracy)
print(f'Epoch [{epoch+1}/{max_epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')