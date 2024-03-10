import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import winsound
import time
import keyboard

import vgamepad as vg
gamepad = vg.VX360Gamepad()

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
            #          kekymap:  'w': 0x57,'a': 0x41,'s': 0x53,'d': 0x44
            predicted_directions = ["down", "left", "right", "up"]
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

def run(gamepad):

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

    play_system_sound(1500)
    area = {"top": 0, "left": 0, "width": 1920, "height": 1080}
    Screenshot(area,"full")
    area = {"top": 191, "left": 103, "width": 22, "height": 22}
    Screenshot(area,"1")
    area = {"top": 191, "left": 103+22, "width": 22, "height": 22}
    Screenshot(area,"2")
    area = {"top": 191, "left": 103+22*2, "width": 22, "height": 22}
    Screenshot(area,"3")
    area = {"top": 191, "left": 103+22*3, "width": 22, "height": 22}
    Screenshot(area,"4")
    area = {"top": 191, "left": 103+22*4, "width": 22, "height": 22}
    Screenshot(area,"5")

    result = predict_categories_for_test_folder(model, test_folder_path)

    # # import keyboard
    # import ctypes
    # from ctypes import wintypes

    # user32 = ctypes.WinDLL('user32', use_last_error=True)
    # INPUT_KEYBOARD = 1
    # KEYEVENTF_EXTENDEDKEY = 0x0001
    # KEYEVENTF_KEYUP       = 0x0002
    # KEYEVENTF_UNICODE     = 0x0004
    # MAPVK_VK_TO_VSC = 0
    # # msdn.microsoft.com/en-us/library/dd375731
    # wintypes.ULONG_PTR = wintypes.WPARAM
    # class MOUSEINPUT(ctypes.Structure):
    #     _fields_ = (("dx",          wintypes.LONG),
    #                 ("dy",          wintypes.LONG),
    #                 ("mouseData",   wintypes.DWORD),
    #                 ("dwFlags",     wintypes.DWORD),
    #                 ("time",        wintypes.DWORD),
    #                 ("dwExtraInfo", wintypes.ULONG_PTR))
    # class KEYBDINPUT(ctypes.Structure):
    #     _fields_ = (("wVk",         wintypes.WORD),
    #                 ("wScan",       wintypes.WORD),
    #                 ("dwFlags",     wintypes.DWORD),
    #                 ("time",        wintypes.DWORD),
    #                 ("dwExtraInfo", wintypes.ULONG_PTR))
    #     def __init__(self, *args, **kwds):
    #         super(KEYBDINPUT, self).__init__(*args, **kwds)
    #         if not self.dwFlags & KEYEVENTF_UNICODE:
    #             self.wScan = user32.MapVirtualKeyExW(self.wVk,
    #                                                  MAPVK_VK_TO_VSC, 0)
    # class HARDWAREINPUT(ctypes.Structure):
    #     _fields_ = (("uMsg",    wintypes.DWORD),
    #                 ("wParamL", wintypes.WORD),
    #                 ("wParamH", wintypes.WORD))
    # class INPUT(ctypes.Structure):
    #     class _INPUT(ctypes.Union):
    #         _fields_ = (("ki", KEYBDINPUT),
    #                     ("mi", MOUSEINPUT),
    #                     ("hi", HARDWAREINPUT))
    #     _anonymous_ = ("_input",)
    #     _fields_ = (("type",   wintypes.DWORD),
    #                 ("_input", _INPUT))
    # LPINPUT = ctypes.POINTER(INPUT)
    # def PressKey(hexKeyCode):
    #     x = INPUT(type=INPUT_KEYBOARD,
    #               ki=KEYBDINPUT(wVk=hexKeyCode))
    #     user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
    # def ReleaseKey(hexKeyCode):
    #     x = INPUT(type=INPUT_KEYBOARD,
    #               ki=KEYBDINPUT(wVk=hexKeyCode,
    #                             dwFlags=KEYEVENTF_KEYUP))
    #     user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))

    # for element in result:
    #     if element != None:
    #         play_system_sound()
    #         PressKey(element)
    #         time.sleep(0.05)
    #         ReleaseKey(element)
    #         time.sleep(0.05)

    for element in result:
        if element != None:
            play_system_sound()
            gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
            gamepad.update()
            time.sleep(0.5)
            gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP)
            gamepad.update()    
            time.sleep(0.05)


keyboard.add_hotkey('ctrl+1', lambda: run(gamepad))
keyboard.wait('f10')