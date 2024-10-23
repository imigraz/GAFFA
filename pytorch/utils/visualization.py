import math
import os
import numpy as np
from PIL import Image, ImageDraw
from random import choice
import random  # Import random module for seeding
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def save_images_with_landmarks(images, targets, img_ids, predictions=None, prefix=''):
    output_dir = 'visualization_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colors = ['blue', 'green', 'yellow', 'magenta', 'cyan', 'black', 'orange', 'purple', 'pink',
              'lime', 'deepskyblue', 'goldenrod', 'mediumspringgreen', 'darkviolet', 'lightseagreen',
              'navy', 'saddlebrown', 'grey', 'darkslateblue', 'olive', 'chocolate', 'teal', 'coral',
              'steelblue', 'forestgreen', 'mediumslateblue']

    # Process each image individually
    for idx, (image, landmarks, img_id) in enumerate(zip(images, targets, img_ids)):
        # Use img_id as seed for random color choice
        random.seed(img_id.item() if torch.is_tensor(img_id) else img_id)  # Ensure img_id is integer if it's a tensor
        # Convert tensor to PIL image
        image = image.cpu().numpy()
        image = ((image + 1) / 2 * 255).clip(0, 255).astype(np.uint8).squeeze(0)
        img_pil = Image.fromarray(image)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        draw = ImageDraw.Draw(img_pil)

        # Draw landmarks and predictions if available
        if predictions is not None:
            for landmark, pred in zip(landmarks, predictions[idx]):
                _, y_true, x_true = landmark

                if len(pred) > 2:
                    _, y_pred, x_pred = pred
                else:
                    y_pred, x_pred = pred
                color = choice(colors)
                radius = 2
                draw.ellipse((x_pred - radius, y_pred - radius, x_pred + radius, y_pred + radius), fill=color)
                draw.line((x_true, y_true, x_pred, y_pred), fill="red", width=1)
        else:
            for landmark in landmarks:
                _, y_true, x_true = landmark
                color = choice(colors)
                radius = 2
                draw.ellipse((x_true - radius, y_true - radius, x_true + radius, y_true + radius), fill=color)

        # Save image with drawn landmarks
        filename = f"{prefix}_image_{img_id}.png"
        img_pil.save(os.path.join(output_dir, filename))


def draw_circle_on_landmark(tensor, coords, landmark_index, img_id=None, filename_prefix='', radius=5):
    """
    Draws a circle at the specified landmark coordinates on the heatmap.

    :param tensor: The input tensor representing the heatmap.
    :param coords: The coordinates of the landmark (2-element tuple or list).
    :param landmark_index: The index of the landmark for title annotation.
    :param img_id: Optional. An identifier for the image, used in the filename.
    :param filename_prefix: Optional. A prefix for the filename when saving the image.
    :param radius: The radius of the circle to be drawn.
    """
    # Convert tensor to numpy array and detach it from the current graph
    tensor_array = tensor.cpu().detach().numpy()

    # Extract x and y coordinates
    _, y, x = coords.cpu().detach().numpy()

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(tensor_array.squeeze(0), cmap='hot', interpolation='nearest')

    # Draw a circle at the specified coordinates
    circle = Circle((x, y), radius, color='blue', fill=False)
    ax.add_patch(circle)

    # Remove axes
    ax.axis('off')  # Turn off the axis

    # Set title for the plot
    #plt.title(f"Image {img_id} Landmark {landmark_index + 1} Location", pad=20)

    # Save the plot to a file if img_id is provided
    if img_id is not None:
        filename = f"{filename_prefix}_{img_id}_landmark_{landmark_index + 1}.png"
        path = os.path.join("visualization_output", filename)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    # Display the plot after saving it
    #plt.show()
    plt.close(fig)  # Close the figure to free up memory


def visualize_cond_heatmaps(conditional_heatmaps: torch.Tensor):
    """
    Visualizes a batch of conditional distribution heatmaps stored in a tensor.

    This function takes a 3D tensor containing conditional heatmaps and displays them in a grid.
    Each heatmap is displayed as a grayscale image. The layout of the grid is calculated to be
    approximately square based on the number of heatmaps.

    Parameters:
    conditional_heatmaps (torch.Tensor): A 3D tensor of shape (N, H, W),
                                         where N is the number of heatmaps,
                                         and H, W are the height and width of each heatmap.

    Returns:
    None: This function displays the heatmaps but does not return any values.
    """
    # Use the first dimension of the tensor to get the number of images
    N = conditional_heatmaps.shape[0]

    # Calculate the number of rows and columns using native Python math
    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    # Set the size of the plotting window, scaling by 2 units for each row and column
    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(N):
        plt.subplot(rows, cols, i + 1)  # Create a subplot for each image
        plt.imshow(conditional_heatmaps[i], cmap='gray')  # Display the image in grayscale
        plt.axis('off')  # Turn off axis numbers and ticks

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()


def imshow_batch(images, title=None):
    """
    Display a batch of images.

    Args:
        images (torch.Tensor): The batch of images tensor, expected shape (batch_size, channels, height, width).
        title (str): Optional title for the plot.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Create a grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=len(images), figsize=(15, 5))

    for i, img in enumerate(images):
        # Check if it's a single channel image (grayscale)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)  # Convert to 3 channel by repeating the single channel

        img = img.numpy().transpose((1, 2, 0))  # Rearrange dimensions to (height, width, channels)
        img = std * img + mean  # Denormalize
        img = np.clip(img, 0, 1)  # Clip values to be between 0 and 1

        axs[i].imshow(img)
        axs[i].axis('off')

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()
