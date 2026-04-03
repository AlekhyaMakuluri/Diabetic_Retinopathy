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

"""create_sample_oct_image function description."""