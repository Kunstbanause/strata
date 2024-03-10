import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import winsound

def play_system_sound(ht=500 ,t=50):
    winsound.Beep(ht, t)  # Frequency = 1000Hz, Duration = 1000ms

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
    num_epochs = 255
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

def predict_category(model, image_path, confidence_threshold=0.6):
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        if confidence.item() < confidence_threshold:
            return None, int(confidence.item()*100)
        else:
            predicted_directions = ["s", "a", "d", "w"]
            predicted_category = predicted_directions[predicted.item()]
            # Display the image with predicted direction as the title
            # plt.imshow(image)
            # plt.title(f"{predicted_category} {int(confidence.item()*100)}%")
            # plt.axis('off')
            # plt.show()
            return predicted_category, int(confidence.item()*100)

# Example usage: predict_category
# image_path = "test/test.png"  # Replace with the path to your image
# predicted_category = predict_category(model, image_path)
# print(f"The predicted category for the image is: {predicted_category}")


def predict_categories_for_test_folder(model, folder_path):
    model.eval()
    directionsArray = []
    for filename in os.listdir(folder_path):
        try:
            image_path = os.path.join(folder_path, filename)
            #direction = predict_category_and_show(model, image_path)
            direction, confidence = predict_category(model, image_path)
            print(f"{filename}: {direction} ({confidence})")
            directionsArray.append(direction)
        except Exception as e:
            print(f"{filename}: An error occurred - {e}")
    return directionsArray

# Call the function to predict categories for all images in the test folder
#test_folder_path = "test/"
test_folder_path = "combo1/"
#predict_categories_for_test_folder(model, test_folder_path)


## Screenshot
import mss.tools
def Screenshot(area,i):
    with mss.mss() as sct:
        # The screen part to capture
        path = "combo1/" + i + ".png"
        # Grab the data
        sct_img = sct.grab(area)
        # Save to the picture file
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=path)

play_system_sound(1000)
area = {"top": 208, "left": 115, "width": 22, "height": 22}
Screenshot(area,"1")
area = {"top": 208, "left": 115+22, "width": 22, "height": 22}
Screenshot(area,"2")
area = {"top": 208, "left": 115+22*2, "width": 22, "height": 22}
Screenshot(area,"3")
area = {"top": 208, "left": 115+22*3, "width": 22, "height": 22}
Screenshot(area,"4")
area = {"top": 208, "left": 115+22*4, "width": 22, "height": 22}
Screenshot(area,"5")

result = predict_categories_for_test_folder(model, test_folder_path)

import keyboard
import time

for element in result:
    if element != None:
        keyboard.press(element)
        play_system_sound()
        time.sleep(0.05)

