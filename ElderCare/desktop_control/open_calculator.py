import os
import platform

def open_calculator():
    if platform.system() == "Windows":
        os.system("start calc")
    elif platform.system() == "Darwin":  # macOS
        os.system("open -a Calculator")
    elif platform.system() == "Linux":
        os.system("gnome-calculator")
