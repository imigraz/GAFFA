import nibabel as nib
import numpy as np
import torch


def flip_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Flips a tensor along its last two dimensions.

    Parameters:
    t (torch.Tensor): Input tensor of at least two dimensions.

    Returns:
    torch.Tensor: The flipped tensor with the last two dimensions reversed.
    """
    flipped = torch.flip(t, dims=[-2, -1])
    return flipped


def compute_meshgrid(height, width):
    """
    Computes meshgrid for the given heatmap size.
    """
    x = torch.linspace(0, width - 1, steps=width)
    y = torch.linspace(0, height - 1, steps=height)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return xx, yy


def save_as_nifti(heatmaps, file_name):
    # Assuming heatmaps is a 4D numpy array: (num_landmarks, num_landmarks, height, width)
    nifti_img = nib.Nifti1Image(heatmaps, affine=np.eye(4))
    nib.save(nifti_img, f"{file_name}.nii.gz")


def load_nii_to_tensor(file_path):
    nii_data = nib.load(file_path)
    array_data = nii_data.get_fdata()
    tensor_data = torch.from_numpy(array_data).float()  # Convert to float tensor
    return tensor_data
