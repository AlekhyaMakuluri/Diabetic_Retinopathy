import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def create_sample_oct_image():
    # Create a blank image with a dark background
    img = Image.new('RGB', (224, 224), color='black')
    draw = ImageDraw.Draw(img)
    
    # Add some random noise to simulate OCT scan
    noise = np.random.normal(128, 30, (224, 224, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    noise_img = Image.fromarray(noise)
    
    # Blend the noise with the background
    img = Image.blend(img, noise_img, 0.7)
    
    # Add some horizontal lines to simulate retinal layers
    for i in range(5):
        y = 50 + i * 30
        draw.line([(0, y), (224, y)], fill='white', width=2)
    
    return img

def create_sample_fundus_image():
    # Create a blank image with a dark background
    img = Image.new('RGB', (224, 224), color='black')
    draw = ImageDraw.Draw(img)
    
    # Add a circular region to simulate the optic disc
    draw.ellipse([(80, 80), (144, 144)], fill='white')
    
    # Add some blood vessels
    for i in range(4):
        x1 = 112
        y1 = 112
        angle = i * 90
        length = 50
        x2 = x1 + length * np.cos(np.radians(angle))
        y2 = y1 + length * np.sin(np.radians(angle))
        draw.line([(x1, y1), (x2, y2)], fill='red', width=3)
    
    # Add some noise to simulate fundus image
    noise = np.random.normal(128, 20, (224, 224, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    noise_img = Image.fromarray(noise)
    
    # Blend the noise with the background
    img = Image.blend(img, noise_img, 0.5)
    
    return img

def generate_test_images():
    # Create output directory if it doesn't exist
    os.makedirs('data/processed/test/test_data', exist_ok=True)
    
    # Generate 5 pairs of images
    for i in range(5):
        # Generate OCT image
        oct_img = create_sample_oct_image()
        oct_path = f'data/processed/test/test_data/oct_{i+1}.png'
        oct_img.save(oct_path)
        
        # Generate fundus image
        fundus_img = create_sample_fundus_image()
        fundus_path = f'data/processed/test/test_data/fundus_{i+1}.png'
        fundus_img.save(fundus_path)
        
        print(f"Generated image pair {i+1}")

if __name__ == "__main__":
    generate_test_images() 