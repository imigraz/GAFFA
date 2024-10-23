import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

from pytorch.utils.helper import compute_meshgrid


def generate_heatmap_target(heatmap_size, landmarks, sigmas, scale=1.0, normalize=False, data_format='channels_first',
                            distribution='gaussian'):
    landmarks_shape = landmarks.shape
    sigmas_shape = sigmas.shape
    batch_size = landmarks_shape[0]
    num_landmarks = landmarks_shape[1]
    dim = landmarks_shape[2] - 1
    assert len(heatmap_size) == dim, 'Dimensions do not match.'
    assert sigmas_shape[0] == num_landmarks, 'Number of sigmas does not match.'

    if data_format == 'channels_first':
        heatmap_axis = 1
        landmarks_reshaped = torch.reshape(landmarks[..., 1:], [batch_size, num_landmarks] + [1] * dim + [dim])
        is_valid_reshaped = torch.reshape(landmarks[..., 0], [batch_size, num_landmarks] + [1] * dim)
        sigmas_reshaped = torch.reshape(sigmas, [1, num_landmarks] + [1] * dim)
    else:
        heatmap_axis = dim + 1
        landmarks_reshaped = torch.reshape(landmarks[..., 1:], [batch_size] + [1] * dim + [num_landmarks, dim])
        is_valid_reshaped = torch.reshape(landmarks[..., 0], [batch_size] + [1] * dim + [num_landmarks])
        sigmas_reshaped = torch.reshape(sigmas, [1] + [1] * dim + [num_landmarks])

    aranges = [torch.arange(s, device='cuda') for s in heatmap_size]
    grid = torch.meshgrid(*aranges, indexing='ij')

    grid_stacked = torch.stack(grid, dim=dim)
    grid_stacked = torch.stack([grid_stacked] * batch_size, dim=0)
    grid_stacked = torch.stack([grid_stacked] * num_landmarks, dim=heatmap_axis)

    if normalize:
        if distribution == 'gaussian':
            scale /= torch.pow(torch.sqrt(torch.tensor(2) * torch.pi) * sigmas_reshaped, dim)
        elif distribution == 'laplacian':
            assert("normalization with laplacian not supported yet!")

    distances = None
    if distribution == 'gaussian':
        distances = torch.sum(torch.pow(grid_stacked - landmarks_reshaped, 2.0), dim=-1)
    elif distribution == 'laplacian':
        distances = torch.sum(torch.abs(grid_stacked - landmarks_reshaped), dim=-1)
    else:
        assert "blob distribution not supported!"

    heatmap = scale * torch.exp(-distances / (2 * torch.pow(sigmas_reshaped, 2)))
    heatmap_or_zeros = torch.where((is_valid_reshaped + torch.zeros_like(heatmap)) > 0,
                                   heatmap, torch.zeros_like(heatmap))

    return heatmap_or_zeros


def fit_gmm(data, max_components=10):
    lowest_score = np.inf
    best_gmm = None
    best_scores = None

    for n_components in range(1, max_components):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(data)
        aic = gmm.aic(data)
        bic = gmm.bic(data)
        avg_score = (aic + bic) / 2  # Use the average of AIC and BIC

        if avg_score < lowest_score:
            lowest_score = avg_score
            best_gmm = gmm
            best_scores = (aic, bic)

    return best_gmm, best_scores

# Global landmarks for the hand (adjusted to zero-based indexing)
manual_defined_landmarks = np.array([2, 6, 14, 15, 16, 17, 18, 37, 33, 29, 25, 21]) - 1


def generate_gmm_heatmaps(landmarks, image_size, max_gmm_components, top_n_combinations, edges_indices):
    num_samples, num_landmarks, _ = landmarks.shape
    frame_center = np.array([image_size[0] // 2, image_size[1] // 2])
    heatmaps = np.zeros((num_landmarks, num_landmarks, image_size[1], image_size[0]), dtype=np.float32)

    x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
    all_pixels = np.stack([x.ravel(), y.ravel()], axis=1)

    landmark_pairs_dict = {i: [] for i in range(num_landmarks)}
    scores_dict = {i: [] for i in range(num_landmarks)}

    # Compute scores for each pair
    for i in range(num_landmarks):
        for j in range(num_landmarks):
            if i != j:
                shifted_positions = []
                for sample in landmarks:
                    if sample[i, 0] > 0 and sample[j, 0] > 0:
                        T_i_cond_i = frame_center - sample[j, 1:3][[1, 0]]
                        shifted_i = sample[i, 1:3][[1, 0]] + T_i_cond_i
                        shifted_positions.append(shifted_i)

                # Fit GMM
                gmm, scores = fit_gmm(np.array(shifted_positions), max_gmm_components)
                avg_score = np.mean(scores)
                scores_dict[i].append((avg_score, j, gmm))

    # Sort and select top N combinations for each landmark
    for i in range(num_landmarks):
        scores_dict[i].sort(key=lambda x: x[0])

        # Initialize selected_j with global landmarks
        selected_j = set(manual_defined_landmarks)

        # Add top N combinations
        selected_j.update({j for _, j, _ in scores_dict[i][:top_n_combinations]})

        # Add neighborhood pairs
        for edge in edges_indices:
            if edge[0] == i:
                selected_j.add(edge[1])
            elif edge[1] == i:
                selected_j.add(edge[0])

        # Generate heatmaps for selected pairs
        for _, j, gmm in scores_dict[i]:
            scores = gmm.score_samples(all_pixels)
            heatmaps[i, j] = np.exp(scores).reshape(image_size[1], image_size[0])
            if j in selected_j:
                landmark_pairs_dict[i].append(j)

    return heatmaps, landmark_pairs_dict


def exact_argmax2d(batch_heatmaps):
    """
    Extracts the 2D coordinates of the maximum value in a 2D heatmap in a non-differentiable way.

    Args:
    batch_heatmaps (Tensor): A batch of 2D heatmaps. Shape: [batch_size, height, width]

    Returns:
    Tensor: The 2D coordinates of the maximum value for each heatmap. Shape: [batch_size, 2]
    """
    # Flatten the heatmap and find the index of the max value
    max_indices = torch.argmax(batch_heatmaps.view(batch_heatmaps.size(0), -1), dim=1)

    # Convert the indices into 2D coordinates
    max_coords = torch.stack((max_indices // batch_heatmaps.size(2), max_indices % batch_heatmaps.size(2)), dim=1)

    return max_coords
