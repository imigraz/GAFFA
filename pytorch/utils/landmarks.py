from medical_data_augment_tool.utils.landmark.common import Landmark


def create_landmarks(cur_target_points):
    """
    Creates a list of landmarks from target points.

    Args:
    cur_target_points (list): A list of target points where each point is a list or tensor.
                              The first element of each point is used to determine validity,
                              and the rest are coordinates.

    Returns:
    list: A list of Landmark objects.
    """
    cur_target_landmarks = []
    for cur_target_point in cur_target_points:
        valid = cur_target_point[0] >= 1.0  # Check if the first element is >= 1.0
        coords = cur_target_point[1:]  # Extract the coordinates
        landmark = Landmark(coords, valid, 1.0)  # Create Landmark object
        cur_target_landmarks.append(landmark)  # Append to the list

    return cur_target_landmarks


def compute_metrics(landmark_statistics, total_num_landmarks):
    ipe = landmark_statistics.get_ipe_statistics()
    pe = landmark_statistics.get_pe_statistics()
    outliers = landmark_statistics.get_num_outliers([2.0, 4.0, 6.0, 10.0])

    acc_2mm = (total_num_landmarks - outliers[0]) / total_num_landmarks
    acc_4mm = (total_num_landmarks - outliers[1]) / total_num_landmarks
    acc_6mm = (total_num_landmarks - outliers[2]) / total_num_landmarks
    acc_10mm = (total_num_landmarks - outliers[3]) / total_num_landmarks

    metrics = {
        "ipe_mean": ipe[0], "ipe_std": ipe[1], "ipe_median": ipe[2],
        "pe_mean": pe[0], "pe_std": pe[1], "pe_median": pe[2],
        "outliers_2mm": float(outliers[0]), "outliers_4mm": float(outliers[1]),
        "outliers_6mm": float(outliers[2]), "outliers_10mm": float(outliers[3]),
        "Accuracy_2mm": acc_2mm, "Accuracy_4mm": acc_4mm,
        "Accuracy_6mm": acc_6mm, "Accuracy_10mm": acc_10mm
    }

    return metrics


def compute_GAFFA_metrics(landmark_statistics, total_num_landmarks):
    ipe = landmark_statistics.get_ipe_statistics()
    pe = landmark_statistics.get_pe_statistics()
    outliers = landmark_statistics.get_num_outliers([2.0, 4.0, 6.0, 10.0])

    acc_2mm = (total_num_landmarks - outliers[0]) / total_num_landmarks
    acc_4mm = (total_num_landmarks - outliers[1]) / total_num_landmarks
    acc_6mm = (total_num_landmarks - outliers[2]) / total_num_landmarks
    acc_10mm = (total_num_landmarks - outliers[3]) / total_num_landmarks

    metrics = {
        "GAFFA_ipe_mean": ipe[0], "GAFFA_ipe_std": ipe[1], "GAFFA_ipe_median": ipe[2],
        "GAFFA_pe_mean": pe[0], "GAFFA_pe_std": pe[1], "GAFFA_pe_median": pe[2],
        "GAFFA_outliers_2mm": float(outliers[0]), "GAFFA_outliers_4mm": float(outliers[1]),
        "GAFFA_outliers_6mm": float(outliers[2]), "GAFFA_outliers_10mm": float(outliers[3]),
        "GAFFA_Accuracy_2mm": acc_2mm, "GAFFA_Accuracy_4mm": acc_4mm,
        "GAFFA_Accuracy_6mm": acc_6mm, "GAFFA_Accuracy_10mm": acc_10mm
    }

    return metrics
