from PIL import Image, ImageFilter
import numpy as np
import os
from tqdm import tqdm
import porespy as ps
from torchvision import transforms


transform = transforms.Compose([
transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
transforms.RandomRotation(degrees=360, expand=True),
transforms.RandomPerspective(distortion_scale=0.2, p=0.6),
transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.01, 2))], p = 0.3),
# Custom_resize_transform(),
])


def make_cow_coat_realistic(image, id, directory_path):
    """
    # Convert the image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    """
    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Define base colors
    white_color = np.array([255, 255, 250])  # Slightly off-white/cream color
    black_variation = np.array([40, 40, 40])  # Variation for black regions
    #white_variation = np.array([15, 15, 10])  # Subtle variation for white regions

    # Create masks for white and black regions
    white_mask = np.all(image_array == [255, 255, 255], axis=-1)
    black_mask = np.all(image_array == [0, 0, 0], axis=-1)

    # Apply color changes to the white areas
    noise_white = np.random.randint(-10, 10, image_array[white_mask].shape)
    image_array[white_mask] = np.clip(white_color  + noise_white, 0, 255)

    # Apply noise and variation to the black regions
    noise_black = np.random.randint(-20, 20, image_array[black_mask].shape)
    image_array[black_mask] = np.clip(image_array[black_mask] + black_variation + noise_black, 0, 255)

    # Convert the NumPy array back to a PIL image
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')


    # Optionally, apply a slight blur to smooth transitions
    image = image.filter(ImageFilter.GaussianBlur(radius=1)).convert('RGBA')

    os.makedirs(f"{directory_path}/{id}", exist_ok=True)
    save_path = f"{directory_path}/{id}/0.png"
    image.save(save_path)

    for augment_number in range(1, augments_per_sample + 1):
        save_path = f"{directory_path}/{id}/{augment_number}.png"
        transform(image).save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str)
    args = parser.parse_args()

    augments_per_sample = 500

    # Generate a random blob pattern
    # Set up the style for visualization
    ps.visualization.set_mpl_style()
    shape = [107, 224]  # Size of the image
    porosity = 0.5  # Adjust to control the amount of black vs white

    for id in tqdm(range(1860, 2000)):
        blobs = ps.generators.blobs(shape=shape, porosity=porosity, blobiness=0.25)
        blobs = (blobs * 255).astype('uint8')
        blobs = np.stack((blobs,) * 3, axis=-1)
        make_cow_coat_realistic(blobs, id, args.directory_path)

