import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import torch.nn.functional as F
import numpy as np
from fft_conv_pytorch import fft_conv

from pytorch.utils.heatmaps import generate_gmm_heatmaps
from pytorch.utils.helper import flip_tensor, save_as_nifti, load_nii_to_tensor
from pytorch.utils.visualization import draw_circle_on_landmark, visualize_cond_heatmaps


def conv_fft(A, B, num_landmarks):
    """
    Perform group convolution in fourier space.

    :param A: Convolution kernel for priors.
    :param B: Input heatmaps (likelihoods).
    :param num_landmarks: Number of landmarks.
    :return: Convolved output.
    """
    A = A.reshape(num_landmarks, 1, A.shape[2], A.shape[3])
    padding = (A.shape[2] - 1) // 2
    B = F.pad(B, (padding, padding + 1, padding, padding + 1), mode="constant", value=0)
    # flip so that we have convolution not correlation
    A = flip_tensor(A)
    # C = F.conv2d(input=B, weight=A, groups=num_landmarks)
    C = fft_conv(signal=B, kernel=A, groups=num_landmarks)
    return C


class GAFFA(nn.Module):
    def __init__(self, num_landmarks=37, heatmap_size=None, cur_cv=1, pretrained=False, direct_train_cond=False):
        super(GAFFA, self).__init__()
        if heatmap_size is None:
            heatmap_size = [512, 512]

        self.num_landmarks = num_landmarks
        self.direct_train_cond = direct_train_cond

        # Load the dictionary with valid cond heatmap combinations (mandatory for training and inference!)
        self.heatmap_dict = np.load(f"conditional_heatmaps_dic_cv{cur_cv}.npy", allow_pickle=True).item()

        self.scale_factor_cond = 0.125
        self.scale_factor_heatmap = 0.25

        # Load all conditional heatmaps, resize
        if not pretrained:
            self.conditional_heatmaps = load_nii_to_tensor(f'conditional_heatmaps_cv{cur_cv}.nii.gz')
            self.conditional_heatmaps = F.interpolate(self.conditional_heatmaps, scale_factor=self.scale_factor_cond,
                                                      mode='bicubic',
                                                      align_corners=False).to('cuda')
        else:
            # loading the model will fill the heatmaps with values
            self.conditional_heatmaps = torch.empty(num_landmarks, num_landmarks,
                                                    int(heatmap_size[0] * self.scale_factor_heatmap),
                                                    int(heatmap_size[1] * self.scale_factor_heatmap)).to('cuda')

        # Set specified conditional heatmaps as learnable parameters
        self.learnable_heatmaps = nn.ParameterDict()
        for i, valid_indices in self.heatmap_dict.items():
            for j in valid_indices:
                param_key = f"{i}_{j}"
                self.learnable_heatmaps[param_key] = nn.Parameter(self.conditional_heatmaps[i, j])

        self.delta = 10 ** -6

        # Initialize learnable bias for each valid conditional heatmap combination
        self.bias = nn.ParameterDict()
        for i in range(num_landmarks):
            valid_indices = self.heatmap_dict[i]
            bias_size = len(valid_indices)
            shape_x = int(heatmap_size[0] * self.scale_factor_heatmap)
            shape_y = int(heatmap_size[1] * self.scale_factor_heatmap)
            self.bias[f'bias_{i}'] = nn.Parameter(torch.zeros(bias_size, shape_x, shape_y) + self.delta)

        # Temperature scaling was not used, should be investigated in future work
        self.temperature = torch.tensor(1.0)
        self.softplus_beta = 5
        self.hm_height = heatmap_size[0]
        self.hm_width = heatmap_size[1]
        self.softplus = nn.Softplus(beta=self.softplus_beta)
        self.bn_heatmaps = nn.BatchNorm2d(num_landmarks)
        # Precompute the meshgrid based on the heatmap size
        self.precompute_meshgrid(heatmap_size[0], heatmap_size[1])

    def precompute_meshgrid(self, height, width):
        """
        Precomputes and stores the meshgrid for the given heatmap size.
        Compared to DSNT, we use the native height and width (no back transformation necessary)
        """
        x = torch.linspace(0, width - 1, steps=width)
        y = torch.linspace(0, height - 1, steps=height)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.precomputed_xx = xx
        self.precomputed_yy = yy

    def differentiable_argmax2d(self, batch_heatmaps):
        """
        Extracts the 2D coordinates of the maximum value in a 2D heatmap in a differentiable way with temperature scaling.

        Args:
        batch_heatmaps (Tensor): A batch of 2D heatmaps. Shape: [batch_size, height, width]

        Returns:
        Tensor: The 2D coordinates of the maximum value for each heatmap. Shape: [batch_size, 2]
        """
        temperature = self.temperature
        # Flatten and apply softmax with temperature scaling
        softmax_heatmaps = torch.softmax(batch_heatmaps.view(batch_heatmaps.shape[0], -1) / temperature, dim=1)

        # Reshape back to original shape
        softmax_heatmaps = softmax_heatmaps.view_as(batch_heatmaps)

        # Use the precomputed meshgrid, ensuring it's on the same device as the input
        xx = self.precomputed_xx.to(batch_heatmaps.device)
        yy = self.precomputed_yy.to(batch_heatmaps.device)

        # Calculating expected values for x and y coordinates
        expected_x = torch.sum(softmax_heatmaps * xx, dim=(1, 2))
        expected_y = torch.sum(softmax_heatmaps * yy, dim=(1, 2))

        # Stacking the coordinates
        coordinates = torch.stack([torch.tensor([1] * len(expected_x)).to("cuda"), expected_x, expected_y], dim=1)
        return coordinates

    def forward(self, heatmaps, img_ids=None, debug=False):
        landmark_coords = []
        heatmaps = self.bn_heatmaps(heatmaps)
        heatmaps_resized = F.interpolate(heatmaps, scale_factor=self.scale_factor_heatmap,
                                         mode='bicubic',
                                         align_corners=False)

        for i in range(self.num_landmarks):
            # Get valid heatmaps indices from dictionary for the current landmark
            valid_indices = self.heatmap_dict[i]
            # Get the corresponding conditional heatmaps
            conditional_heatmaps = self.conditional_heatmaps[i, valid_indices]
            if self.direct_train_cond:
                # Instead of only setting up the precomputed conditional_heatmaps as learnable network parameters
                # directly access them for correct gradient tracking
                # leads to catastrophic forgetting of prior anatomical knowledge, future work
                learnable_heatmaps = []
                for j in valid_indices:
                    key = f"{i}_{j}"
                    learnable_heatmaps.append(self.learnable_heatmaps[key])

                # Stack the learnable heatmaps along a new dimension to match the original tensor structure
                learnable_heatmaps = torch.stack(learnable_heatmaps)
                conditional_heatmaps = learnable_heatmaps
            # visualize_cond_heatmaps(conditional_heatmaps.cpu())
            conditional_heatmaps = F.interpolate(conditional_heatmaps.unsqueeze(1),
                                                 scale_factor=(self.scale_factor_heatmap / self.scale_factor_cond),
                                                 mode='bicubic',
                                                 align_corners=False)
            prior = self.softplus(conditional_heatmaps)
            # For each conditional probability p(i|j), we have p(j)
            likelihood = self.softplus(heatmaps_resized[:, valid_indices])
            # Get the bias for the current landmark
            current_bias = self.softplus(self.bias[f'bias_{i}'])
            convolved = conv_fft(prior, likelihood, len(valid_indices))
            marginal_energy = torch.sum(torch.log(convolved + current_bias + self.delta), dim=1)
            marginal_energy = marginal_energy.unsqueeze(1)

            marginal_energy = F.interpolate(marginal_energy, scale_factor=1 / self.scale_factor_heatmap,
                                            mode='bicubic',
                                            align_corners=False)
            # Get the original heatmap from UNet and add marginal energy
            hm = heatmaps[:, i, :, :].unsqueeze(1)
            init_marginal_energy = torch.log(self.softplus(hm) + self.delta)
            marginal_energy += init_marginal_energy
            landmark_coords_tensor = self.differentiable_argmax2d(marginal_energy.squeeze(1))

            # Can be used to save spatial energy heatmaps of specific image to disk
            debug_image_id = "3771"
            if debug and (debug_image_id in img_ids):
                bias = torch.sum(torch.log(current_bias + self.delta), dim=0).unsqueeze(0)
                conv = torch.sum(torch.log(convolved + self.delta), dim=1).unsqueeze(1)
                bias = F.interpolate(bias.unsqueeze(0), scale_factor=1 / self.scale_factor_heatmap,
                                     mode='bicubic',
                                     align_corners=False)
                conv = F.interpolate(conv, scale_factor=1 / self.scale_factor_heatmap,
                                     mode='bicubic',
                                     align_corners=False)
                for j in range(0, len(marginal_energy)):
                    if img_ids[j] == debug_image_id:
                        draw_circle_on_landmark(init_marginal_energy[j], landmark_coords_tensor[j], i,
                                                img_ids[j], "loc")
                        draw_circle_on_landmark(bias[0], landmark_coords_tensor[j], i,
                                                img_ids[j], "bias")
                        draw_circle_on_landmark(conv[j], landmark_coords_tensor[j], i,
                                                img_ids[j], "conv")
                        draw_circle_on_landmark(marginal_energy[j] - init_marginal_energy[j],
                                                landmark_coords_tensor[j], i, img_ids[j], "marginal_energy")
                        draw_circle_on_landmark(marginal_energy[j], landmark_coords_tensor[j], i, img_ids[j],
                                                "final_marginal_energy")
            landmark_coords.append(landmark_coords_tensor)

        # Convert the list of coordinates to a tensor
        landmark_coords = torch.stack(landmark_coords, dim=1)
        return landmark_coords


def learn_conditional_distr(config_dic, data_cur_cv, cur_cv):
    output_heatmap_size = 2 * np.array(list(reversed(config_dic["output_size_model"])))

    train_loader = data_cur_cv.train_dataloader()
    file1 = open('configs/MRF_connectivity/xray_hand_edges.txt', 'r')
    Lines = file1.readlines()
    edges_indices = set()
    for line in Lines:
        a, b = map(int, line.split())
        edges_indices.add((a - 1, b - 1))

    landmarks_batch_list = []
    # repeat sampling 3x, to have more samples for GMM
    for idx, sample in enumerate(train_loader):
        landmarks_heatmap_space_batch = sample[2]
        landmarks_batch_list.append(landmarks_heatmap_space_batch)
    for idx, sample in enumerate(train_loader):
        landmarks_heatmap_space_batch = sample[2]
        landmarks_batch_list.append(landmarks_heatmap_space_batch)
    for idx, sample in enumerate(train_loader):
        landmarks_heatmap_space_batch = sample[2]
        landmarks_batch_list.append(landmarks_heatmap_space_batch)
    landmarks = torch.cat(landmarks_batch_list, dim=0)
    heatmaps, heatmaps_dic = generate_gmm_heatmaps(np.array(landmarks), output_heatmap_size, 10, 0, edges_indices)
    # Visualize one of the heatmaps
    #plt.imshow(heatmaps[1, 5], cmap='hot', interpolation='nearest')
    # Mark the center frame pixel
    #center_x, center_y = output_heatmap_size[0] // 2, output_heatmap_size[1] // 2
    #plt.scatter(center_x, center_y, c='red', marker='x', s=30)  # blue x at the center
    #plt.show()
    np.save(f"conditional_heatmaps_dic_cv{cur_cv}.npy", heatmaps_dic)
    # save as nifti to save disk space
    save_as_nifti(heatmaps, f"conditional_heatmaps_cv{cur_cv}")
