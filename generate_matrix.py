import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

unseen_dataset = datasets.ImageFolder('processed_dataset/test_unseen', transform=transform)
unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)
class_names = unseen_dataset.classes

model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('best_cnn_model.pth'))
model = model.to(device)
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in unseen_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Generate and save the plot directly to your system
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Unseen Subject (Person 09)')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Confusion matrix saved locally as 'confusion_matrix.png'")

# Also print the raw array just in case
print("\nRaw Matrix Array:")
print(repr(cm))
