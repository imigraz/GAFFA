from medical_data_augment_tool.utils.io.image import read
from medical_data_augment_tool.utils.sitk_np import sitk_to_np
import pandas as pd
import numpy as np
import numbers, math
from scipy.stats import t
import factorgraph as fg
from skimage import data, feature
from matplotlib import pyplot as plt
import SimpleITK as sitk
from medical_data_augment_tool.utils.landmark.landmark_statistics import LandmarkStatistics
from medical_data_augment_tool.utils.landmark.common import Landmark
from medical_data_augment_tool.utils.landmark.heatmap_test import HeatmapTest
from medical_data_augment_tool.utils.landmark.heatmap_image_generator import HeatmapImageGenerator
import networkx as nx
from matplotlib import pyplot as plt, pyplot
import os
import torch


def normalize(costs):
    return np.array([float(i) / sum(costs) for i in costs])


def get_ratio_cost(org_dist, label_ratio,
                   landmarks_l1, landmarks_l2, stat):
    list_ratio = []
    for l1 in landmarks_l1:
        for l2 in landmarks_l2:
            dist = stat.get_distance(l1, l2, None, 1.0)
            list_ratio.append(org_dist / (dist + 1e-4))
    data_ratio = min(list_ratio, key=lambda x: abs(x - label_ratio))
    return np.abs(label_ratio - data_ratio)


def train_graph(config_dic: {}, data_cur_cv, cur_cv_nr):
    train_loader = data_cur_cv.train_dataloader()
    file1 = open('configs/MRF_connectivity/xray_hand_edges.txt', 'r')
    Lines = file1.readlines()
    edges_indices = set()
    column_names = []
    for line in Lines:
        column_names.append(line.rstrip())
        a, b = str.split(line, " ")
        edges_indices.add((int(a) - 1, int(b) - 1))
    edges_indices = sorted(edges_indices)
    num_train_samples = config_dic["num_train_samples"]
    distance_measurements = np.empty(shape=(num_train_samples, len(edges_indices)))
    angle_measurements = np.empty(shape=(num_train_samples, len(edges_indices)))
    stat = LandmarkStatistics()
    normalization_factor = 50
    graph = True
    for idx, sample in enumerate(train_loader):
        landmarks_batch = np.array(sample[1])
        for landmarks_batch_id in range(0, len(landmarks_batch)):
            global_id = idx * config_dic["batch_size"] + landmarks_batch_id
            landmarks = landmarks_batch[landmarks_batch_id]
            x = landmarks[:, 1]
            y = landmarks[:, 2]
            n = range(1, 38)

            if graph:
                G = nx.Graph()
                for i in range(len(x)):
                    G.add_node(str(n[i]), pos=(x[i], y[i]))

                file1 = open('configs/MRF_connectivity/xray_hand_edges.txt', 'r')
                Lines = file1.readlines()
                for line in Lines:
                    a, b = str.split(line, " ")
                    G.add_edge(a, b.strip())
                pyplot.gca().invert_yaxis()
                nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=0)
                plt.show()
                graph = False

            cur_landmarks = []
            for cur_target_point in landmarks:
                valid = cur_target_point[0] >= 1.0
                coords = cur_target_point[1:]
                landmark = Landmark(coords, valid, 1.0)
                cur_landmarks.append(landmark)

            norm_distance = stat.get_distance(cur_landmarks[1], cur_landmarks[5], None, 1.0)
            if np.isnan(norm_distance):
                break
            nf = normalization_factor / norm_distance
            for edge_idx, edge in enumerate(edges_indices):
                l1 = cur_landmarks[edge[0]]
                l2 = cur_landmarks[edge[1]]
                distance_measurements[global_id][edge_idx] = stat.get_distance(l1, l2, None,
                                                                               normalization_factor=nf)
                v = l2.coords - l1.coords
                angle_measurements[global_id][edge_idx] = np.arctan2(v[1], v[0]) * 180 / np.pi
    plt.rcParams.update({'font.size': 5})
    df_distances = pd.DataFrame(distance_measurements, columns=column_names)
    df_angles = pd.DataFrame(angle_measurements, columns=column_names)
    # df_distances['2 6'].plot(kind='density')
    # plt.show()

    t_distribs_distances = np.empty(shape=(len(df_distances.columns), 3))
    for idx, column in enumerate(df_distances):
        columnData = df_distances[column]
        n_df, mu_t, sigma_t = t.fit(columnData)
        t_distribs_distances[idx] = (n_df, mu_t, sigma_t)
    angle_means = np.empty(shape=(len(df_distances.columns)))
    for idx, column in enumerate(df_angles):
        columnData = df_angles[column]
        angle_means[idx] = columnData.mean()
    path = os.path.join("graph_distr", str(cur_cv_nr), str(num_train_samples))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, "t_distribs_norm.csv")
    np.savetxt(path, t_distribs_distances, delimiter=",")


def apply_lbp_ratio_pdf(prediction_heatmaps, distance_distr, edges_indices, heatmap_generator):
    heatmap_test = HeatmapTest(channel_axis=0, invert_transformation=False)
    stat = LandmarkStatistics()
    ratio_mean_std = distance_distr[:, 2] / distance_distr[:, 1]
    num_used_ratios = 3
    min_indices_ratio_mean_std = np.argpartition(ratio_mean_std, num_used_ratios)[0:num_used_ratios]
    landmark_point1 = heatmap_test.get_landmark(prediction_heatmaps[1])
    landmark_point5 = heatmap_test.get_landmark(prediction_heatmaps[5])

    normalization_factor = 50
    norm_distance = stat.get_distance(landmark_point1, landmark_point5, None, 1.0)
    nf = normalization_factor / norm_distance

    landmarks_all = []
    confs_all = []
    for idx, single_heatmap in enumerate(prediction_heatmaps):
        threshold = np.percentile(single_heatmap, 95)
        peak_pts = feature.blob_log(single_heatmap, min_sigma=2, max_sigma=4, threshold=threshold)[0:25]
        landmarks = []
        confs = []
        for point in peak_pts:
            conf = single_heatmap[int(point[0])][int(point[1])]
            confs.append(conf)
            valid = 1.0
            coords = np.array([point[1], point[0]])
            landmark = Landmark(coords, valid, 1.0, conf)
            landmarks.append(landmark)
        confs = np.array(confs)
        landmarks = np.array(landmarks)
        sorted_confs_indices = np.argsort(confs)[::-1]
        norm_confs = normalize(confs[sorted_confs_indices])
        landmarks = landmarks[sorted_confs_indices]
        cutoff_index = [x for x, val in enumerate(norm_confs) if val < 0.015]
        if len(cutoff_index) > 0:
            cutoff_index = cutoff_index[0]
        else:
            cutoff_index = len(norm_confs - 1)
        if len(landmarks[0:cutoff_index]) < 1:
            # Find the maximum value in the heatmap
            max_val = torch.max(single_heatmap)
            # Get the coordinates of all occurrences of the max value
            max_coords = torch.nonzero(single_heatmap == max_val)
            # For simplicity, return the first occurrence if multiple exist
            landmarks_all.append([max_coords])
            confs_all.append([max_val])
        landmarks_all.append(landmarks[0:cutoff_index])
        confs_all.append(confs[0:cutoff_index])

    # Make an empty graph
    g = fg.Graph()

    # Add some discrete random variables (RVs) and add unary factors
    for i in range(0, len(landmarks_all)):
        node_name = str(i + 1)
        if len(landmarks_all[i]) < 1:
            i = 0
        g.rv(node_name, len(landmarks_all[i]))
        g.factor([node_name], potential=np.array(confs_all[i]))

    # Add binary factors
    rand_indices = min_indices_ratio_mean_std
    for idx, cur_edge_indices in enumerate(edges_indices):
        binary_costs = []
        for l1 in landmarks_all[cur_edge_indices[0]]:
            ratio_costs = []
            pdf_costs = []
            for l2 in landmarks_all[cur_edge_indices[1]]:
                dist = stat.get_distance(l1, l2, None, nf)
                cur_distr = distance_distr[idx]
                ratio_exps_cost = []
                for rand_idx in rand_indices:
                    if rand_idx == idx:
                        continue
                    edge = edges_indices[rand_idx]
                    label_ratio = cur_distr[1] / (distance_distr[rand_idx][1] + 1e-4)
                    ratio_cost = get_ratio_cost(dist, label_ratio, landmarks_all[edge[0]], landmarks_all[edge[1]], stat)
                    ratio_exps_cost.append(np.exp(-ratio_cost))
                pdf = t.pdf(dist, df=cur_distr[0], loc=cur_distr[1], scale=cur_distr[2])
                pdf_costs.append(pdf)
                ratio_probs = np.mean(ratio_exps_cost)
                ratio_costs.append(ratio_probs)
            norm_ratio_costs = normalize(ratio_costs)
            norm_pdf = normalize(pdf_costs)
            binary_costs.append(norm_ratio_costs + norm_pdf)
        binary_costs_array = np.array(binary_costs)
        g.factor([str(cur_edge_indices[0] + 1), str(cur_edge_indices[1] + 1)], potential=binary_costs_array)

    # Run (loopy) belief propagation (LBP)
    g.lbp(normalize=True)

    postprocessed_landmarks = []
    for idx, marginal in enumerate(g.rv_marginals()):
        max_marg_idx = np.argmax(marginal[1])
        landmark = landmarks_all[idx][max_marg_idx]
        postprocessed_landmarks.append(landmark)
    return heatmap_generator.generate_heatmaps(postprocessed_landmarks, 0)
