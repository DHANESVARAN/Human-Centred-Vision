# evaluate_unseen.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Same transform as validation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

unseen_dataset = datasets.ImageFolder('processed_dataset/test_unseen', transform=transform)
unseen_loader = DataLoader(unseen_dataset, batch_size=32, shuffle=False)

# Load Model
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load('best_cnn_model.pth'))
model = model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in unseen_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on UNSEEN subject (Person 09): {100 * correct / total:.2f}%")
