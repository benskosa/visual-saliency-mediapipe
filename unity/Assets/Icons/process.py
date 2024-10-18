import os
from PIL import Image

# Set the target size
TARGET_SIZE = (384, 384)

def process_png(filename):
    # Open the image file
    with Image.open(filename) as img:
        # Ensure it's in RGBA mode to preserve the alpha channel
        img = img.convert("RGBA")

        # Resize the image
        img = img.resize(TARGET_SIZE, Image.LANCZOS)

        # Get the pixel data
        data = img.getdata()

        # Create a new list to store modified pixels
        new_data = []
        for item in data:
            # item is a tuple (R, G, B, A)
            r, g, b, a = item

            # Change black (0, 0, 0) to white (255, 255, 255), preserving alpha
            if r == 0 and g == 0 and b == 0:
                new_data.append((255, 255, 255, a))  # Change to white but keep alpha
            else:
                new_data.append(item)  # Keep other colors as they are

        # Update the image with new pixel data
        img.putdata(new_data)

        # Overwrite the original file
        img.save(filename)
        print(f"Processed and overwritten: {filename}")

def find_and_process_pngs():
    # Get all files in the current directory
    for file in os.listdir('.'):
        # Check if the file is a PNG
        if file.lower().endswith('.png'):
            print(f"Processing file: {file}")
            process_png(file)

if __name__ == "__main__":
    find_and_process_pngs()
