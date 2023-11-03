import json
import tkinter as tk
from tkinter import filedialog

# Function to update JSON fields
def update_json_field(key):
    file_path = filedialog.askopenfilename(title="Select a file")
    if file_path:
        config_data[key] = file_path
        with open('config.json', 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

# Load the JSON config file
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

# Create a basic GUI window
window = tk.Tk()
window.title("JSON Config Updater")

# Create buttons for each field in the JSON
for key in config_data:
    button = tk.Button(window, text=f"Update {key}", command=lambda k=key: update_json_field(k))
    button.pack()

window.mainloop()
