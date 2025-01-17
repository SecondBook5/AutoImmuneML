import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from skimage.transform import resize

def visualize_large_images(sdata, image_pairs, output_dir="visualizations", figsize=(20, 10),
                           chunk_size=(2048, 2048), downsample_factor=4, max_workers=6):
    """
    Efficiently visualize large images using chunking, downsampling, and parallel processing.

    Args:
        sdata (SpatialData): The spatial data object containing the images.
        image_pairs (list of tuples): List of (image_keys, titles) pairs to visualize.
        output_dir (str): Directory to save intermediate results.
        figsize (tuple): Figure size for the plots.
        chunk_size (tuple): Size of chunks for processing large images.
        downsample_factor (int): Factor to downsample the images for visualization.
        max_workers (int): Maximum number of threads for parallel processing.
    """
    # Ensure the output directory exists for saving visualizations
    os.makedirs(output_dir, exist_ok=True)

    def load_chunked_image(image_key):
        """
        Load a large image in chunks, downsample it, and reconstruct the full image.

        Args:
            image_key (str): Key of the image to process.

        Returns:
            np.ndarray: Fully reconstructed image from chunks.
        """
        # Access the lazy image from the SpatialData object
        lazy_image = sdata.images[image_key]
        channels, height, width = lazy_image.shape

        # Initialize an empty array for the downsampled image
        full_image = np.zeros((channels, height // downsample_factor, width // downsample_factor), dtype=np.float32)

        # Calculate total number of chunks for progress tracking
        total_chunks = (height // chunk_size[0] + (height % chunk_size[0] != 0)) * \
                       (width // chunk_size[1] + (width % chunk_size[1] != 0))

        # Load and process each chunk, updating the progress bar
        with tqdm(total=total_chunks, desc=f"Loading and downsampling {image_key}") as pbar:
            for y_start in range(0, height, chunk_size[0]):
                for x_start in range(0, width, chunk_size[1]):
                    # Define chunk boundaries
                    y_end = min(y_start + chunk_size[0], height)
                    x_end = min(x_start + chunk_size[1], width)
                    try:
                        # Load the chunk from the lazy image
                        chunk = lazy_image[:, y_start:y_end, x_start:x_end].to_numpy()

                        # Downsample the chunk
                        downsampled_chunk = resize(chunk,
                                                   (chunk.shape[0],
                                                    (y_end - y_start) // downsample_factor,
                                                    (x_end - x_start) // downsample_factor),
                                                   anti_aliasing=True)

                        # Place the downsampled chunk into the full image
                        full_image[:, y_start // downsample_factor:(y_end // downsample_factor),
                                   x_start // downsample_factor:(x_end // downsample_factor)] = downsampled_chunk

                        # Update progress bar
                        pbar.update(1)
                    except Exception as e:
                        print(f"[ERROR] Failed to process chunk: {y_start}:{y_end}, {x_start}:{x_end} - {e}")
                        pbar.update(1)
        return full_image

    def process_pair(image_keys, titles):
        """
        Process and visualize a pair of images.

        Args:
            image_keys (list): Keys of the images to visualize.
            titles (list): Titles corresponding to the images.
        """
        # Validate that each image key has a corresponding title
        if len(image_keys) != len(titles):
            raise ValueError("Image keys and titles must have the same length.")

        # Create subplots for side-by-side visualization
        fig, axes = plt.subplots(1, len(image_keys), figsize=figsize)

        # Iterate over each image key and its corresponding title
        for ax, key, title in zip(axes, image_keys, titles):
            try:
                # Ensure the image exists in the SpatialData object
                if key in sdata.images.keys():
                    print(f"[INFO] Loading and visualizing {key}")

                    # Load the full image using chunked processing
                    image = load_chunked_image(key)

                    print(f"[INFO] Image {key} loaded successfully.")

                    # Display the image in the subplot
                    ax.imshow(image.transpose(1, 2, 0) if image.shape[0] == 3 else image[0], cmap="gray")
                    ax.set_title(title, fontsize=14)
                else:
                    # Handle case where the image key is not found
                    ax.set_title(f"{title}: Not Found", fontsize=12)
                    ax.text(0.5, 0.5, "Image not found", ha="center", va="center", color="red")
                ax.axis("off")
            except Exception as e:
                # Handle any errors during processing
                ax.set_title(f"{title}: Error", fontsize=12)
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha="center", va="center", color="red")
                ax.axis("off")

        # Adjust layout and save the visualization
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{image_keys[0]}_visualization.png")
        plt.savefig(output_path)
        print(f"[INFO] Visualization saved to {output_path}")
        plt.show()

    # Process each image pair in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(lambda pair: process_pair(*pair), image_pairs),
                  desc="Processing Image Pairs", total=len(image_pairs)))

# Extract the specific dataset from preloaded data
single_zarr_key = "UC1_NI.zarr"
single_dataset = all_zarr_data[single_zarr_key]

# Define image pairs and their titles for visualization
image_pairs = [
    (["HE_registered", "HE_nuc_registered"], ["H&E Registered", "H&E Nucleus Registered"]),
]

# Visualize images with optimized chunking, downsampling, and logging
visualize_large_images(
    single_dataset,
    image_pairs=image_pairs,
    chunk_size=(2048, 2048),  # Larger chunks for faster processing
    downsample_factor=4,      # Downsample to 1/4th resolution
    max_workers=6             # Use 6 parallel workers
)
