from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model.eval()  # Set the model to evaluation mode

predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get the index of the maximum value

        predictions.extend(predicted.cpu().numpy())  # Convert to NumPy for scikit-learn
        true_labels.extend(labels.numpy())

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')