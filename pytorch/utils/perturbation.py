import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.utils import save_image

plt.rcParams["savefig.bbox"] = 'tight'

little_finger = [13, 33, 34, 35, 36]
ring_finger = [14, 29, 30, 31, 32]
middle_finger = [15, 25, 26, 27, 28]
index_finger = [16, 21, 22, 23, 24]
thumb_finger = [17, 18, 19, 20, -1]

all_fingers = torch.tensor([little_finger, ring_finger, middle_finger, index_finger, thumb_finger])


def get_neighbors(lst):
    if len(lst) < 2:
        return torch.tensor([])

    neighbors = [[lst[i], lst[i + 1]] for i in range(len(lst) - 1)]
    return torch.tensor(neighbors)


def add_gaussian_noise(tensor, mean=0, std=1):
    return tensor + torch.randn(tensor.size(), device="cuda") * std + mean


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def draw_circle(image_tensor, center_x, center_y, radius):
    """
    Draw a circle on the input image.

    Parameters:
    - image_tensor (torch.Tensor): Input image tensor.
    - center_x (int): x-coordinate of the circle's center.
    - center_y (int): y-coordinate of the circle's center.
    - radius (int): Radius of the circle.
    - color (tuple): RGB color tuple for the circle (default is red).

    Returns:
    - torch.Tensor: Image tensor with the circle drawn.
    """
    image_tensor = torch.squeeze(image_tensor)

    # Generate coordinates for the circle
    aranges = [torch.arange(s, device='cuda') for s in image_tensor.shape]
    x, y = torch.meshgrid(*aranges, indexing='ij')

    # Use the circle equation to set pixels inside the circle to 1
    circle_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2

    # Create image containing Gaussian noise
    noisy_image = torch.zeros_like(image_tensor)
    noisy_image = add_gaussian_noise(noisy_image, 0, 0.02)

    # Fill the circle in the original image with Gaussian noise
    image_tensor[circle_mask] = noisy_image[circle_mask]

    return torch.unsqueeze(image_tensor, 0)


def simulate_occlusion_fingers(image_tensor, landmark_locations, radius=10):
    for img_id in range(0, image_tensor.shape[0]):
        rand_fingers = torch.randint(low=0, high=len(all_fingers), size=(len(all_fingers),))
        rand_fingers = list(all_fingers[rand_fingers == 0])
        for i in range(0, len(rand_fingers)):
            rand_finger = rand_fingers[i]
            rand_finger = get_neighbors(rand_finger)
            rand_parts = torch.randint(low=0, high=len(rand_finger)-1, size=(4,))
            rand_fingers[i] = rand_finger[rand_parts == 0]
        for finger in rand_fingers:
            if len(finger) == 0:
                continue
            for first_landmark_id, second_landmark_id in finger:
                if -1 in second_landmark_id:
                    continue
                image = image_tensor[img_id]
                # get center coordinates of neighbouring finger landmarks
                first_x = landmark_locations[img_id][first_landmark_id][1]
                first_y = landmark_locations[img_id][first_landmark_id][2]
                second_x = landmark_locations[img_id][second_landmark_id][1]
                second_y = landmark_locations[img_id][second_landmark_id][2]

                vector = torch.tensor([first_x - second_x, first_y - second_y], device='cuda')
                distance = torch.norm(vector)
                amount_circles = ((distance / radius) + 1).to(dtype=torch.int)

                part_way = vector * (1.0 / amount_circles)
                draw_circle(image, first_x, first_y, radius)
                draw_circle(image, second_x, second_y, radius)
                for i in range(0, amount_circles):
                    image = draw_circle(image, second_x + part_way[0] * i, second_y + part_way[1] * i, radius)
                image_tensor[img_id] = image
                # save_image(image_tensor[img_id], str(img_id) + ".png")
    return image_tensor
