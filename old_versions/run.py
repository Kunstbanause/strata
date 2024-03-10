import keyboard
import subprocess

def run():
    subprocess.run(["python", "strata09.py"])

keyboard.add_hotkey('F10', run)

# Keep the script running
keyboard.wait('esc')