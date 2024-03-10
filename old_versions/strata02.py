import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root='train/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root='validation/', transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Output layer with 4 classes (left, right, up, down)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if there is a saved model
if os.path.exists('arrow_model.pth'):
    print("Loading model from checkpoint...")
    model.load_state_dict(torch.load('arrow_model.pth'))
    print("Model loaded successfully!")
    train_model = False  # Do not train if model was loaded
else:
    print("No checkpoint found, starting training from scratch...")
    train_model = True

if train_model:
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}")

    # Save the model after training
    torch.save(model.state_dict(), 'arrow_model.pth')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on validation set: {(correct / total) * 100}%")

from PIL import Image

def predict_arrow_direction(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        if predicted.item() == 0:
            return "Left"
        elif predicted.item() == 1:
            return "Right"
        elif predicted.item() == 2:
            return "Up"
        elif predicted.item() == 3:
            return "Down"

def predict_arrow_direction_for_folder(folder_path):
    for filename in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, filename)
            direction = predict_arrow_direction(image_path)
            print(f"{filename}: The arrow is pointing {direction}.")
        except Exception as e:
            print(f"{filename}: An error occurred - {e}")

def predict_arrow_direction_for_folder(folder_path):
    for filename in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, filename)
            direction = predict_arrow_direction(image_path)
            print(f"{filename}: The arrow is pointing {direction}.")
        except Exception as e:
            print(f"{filename}: An error occurred - {e}")

# Call the function to predict directions for all images in the test folder
test_folder_path = "test/"
predict_arrow_direction_for_folder(test_folder_path)