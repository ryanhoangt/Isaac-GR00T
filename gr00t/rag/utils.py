import random
import numpy as np
import torch
from dtaidistance import dtw_ndim, clustering


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_elapsed_time(start, end):
    seconds = int(end - start)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def log_message(message, log_dir=None):
    print(message)
    if log_dir is not None:
        with open(f"{log_dir}/messages.log", "a") as f:
            print(message, file=f)


def random_distribute_n_to_k(n, k):
    num_cuts = k - 1
    cut_points = random.choices(range(n + 1), k=num_cuts)

    points = [0, n] + cut_points
    points.sort()

    distribution = []
    for i in range(1, len(points)):
        x_i = points[i] - points[i-1]
        distribution.append(x_i)
    return distribution


def _get_traj_actions(dataset, traj_ids):
    traj_actions = []
    for traj_id in traj_ids:
        traj = dataset.get_trajectory_data(traj_id)
        traj_action = np.stack(traj["action"].tolist(), dtype=np.double)
        traj_actions.append(traj_action)
    return traj_actions


def get_sampling_weights(dataset, task):
    traj_ids = dataset.task_groups[task]
    traj_actions = _get_traj_actions(dataset, traj_ids)

    distance_matrix = dtw_ndim.distance_matrix_fast(traj_actions)
    np.fill_diagonal(distance_matrix, 10000)  # NOTE: (optional)

    rank_matrix = distance_matrix.argsort()[::-1].argsort()
    weight_matrix = 2 ** rank_matrix  # NOTE: overflow risk...
    weight_matrix = weight_matrix / weight_matrix.sum(-1, keepdims=True)

    sampling_weights = dict(zip(traj_ids, weight_matrix))
    return sampling_weights


def _filter_unique_first_element(tuple_list):
    result_list = []
    seen_first_elements = set()

    for item in tuple_list:
        first_element = item[0]
        if first_element not in seen_first_elements:
            result_list.append(item)
            seen_first_elements.add(first_element)
    return result_list


def _random_sample_preserving_order(population, k):
    n = len(population)
    random_indices = random.sample(range(n), k)
    sorted_indices = sorted(random_indices)

    samples = [population[i] for i in sorted_indices]
    return samples


def _visualize_aligned_frame_ids(dataset, traj_id1, traj_id2, frame_ids1, frame_ids2):
    frames1 = [dataset.get_step_data(traj_id1, frame_id1) for frame_id1 in frame_ids1]
    frames2 = [dataset.get_step_data(traj_id2, frame_id2) for frame_id2 in frame_ids2]

    images1 = [frame1["video.image"][0] for frame1 in frames1]
    images2 = [frame2["video.image"][0] for frame2 in frames2]
    images = np.concatenate([
        np.concatenate(images1, axis=1),
        np.concatenate(images2, axis=1),
    ], axis=0)

    from PIL import Image
    Image.fromarray(images).save("scripts/rag/aligned_frames.png")


def get_aligned_frame_ids(dataset, traj_id1, traj_id2, num_frames):
    assert dataset.trajectory_ids[traj_id1] == traj_id1, "Trajectory ID != Trajectory Index"       
    assert dataset.trajectory_ids[traj_id2] == traj_id2, "Trajectory ID != Trajectory Index"

    traj1 = dataset.get_trajectory_data(traj_id1)
    traj2 = dataset.get_trajectory_data(traj_id2)
    traj_action1 = np.stack(traj1["action"].tolist(), dtype=np.double)
    traj_action2 = np.stack(traj2["action"].tolist(), dtype=np.double)

    frame_pairs = dtw_ndim.warping_path(traj_action1, traj_action2)
    frame_pairs = _filter_unique_first_element(frame_pairs)
    frame_pairs = _random_sample_preserving_order(frame_pairs, k=num_frames)

    frame_ids1, frame_ids2 = zip(*frame_pairs)
    # _visualize_aligned_frame_ids(dataset, traj_id1, traj_id2, frame_ids1, frame_ids2)
    return frame_ids1, frame_ids2


def _get_key_traj_ids(dataset, task, num_trajs_per_task, method="random"):
    traj_ids = dataset.task_groups[task]
    traj_actions = _get_traj_actions(dataset, traj_ids)

    if method == "kmedoids":
        # w/o normalization (distance / len(path))
        kmedoids = clustering.KMedoids(dtw_ndim.distance_matrix_fast, {}, k=num_trajs_per_task)
        key_traj_indices = kmedoids.fit(traj_actions).keys()
    else:
        key_traj_indices = random.sample(range(len(traj_ids)), num_trajs_per_task)

    key_traj_ids = [traj_ids[key_traj_index] for key_traj_index in key_traj_indices]
    key_traj_actions = [traj_actions[key_traj_index] for key_traj_index in key_traj_indices]
    return key_traj_ids, key_traj_actions


def get_key_frames(dataset, task, num_trajs_per_task, frame_stride, transforms, method="random"):
    key_traj_ids, _ = _get_key_traj_ids(dataset, task, num_trajs_per_task, method)

    key_frames = []
    for key_traj_id in key_traj_ids:
        assert dataset.trajectory_ids[key_traj_id] == key_traj_id, (
            "Trajectory ID != Trajectory Index")
        key_traj_len = dataset.trajectory_lengths[key_traj_id]
        for frame_id in range(0, key_traj_len, frame_stride):
            key_frame = transforms(dataset.get_step_data(key_traj_id, frame_id))
            key_frames.append(key_frame)
    return key_frames


def get_key_trajs(dataset, task, num_trajs_per_task, transforms, method="random"):
    key_traj_ids, key_traj_actions = _get_key_traj_ids(dataset, task, num_trajs_per_task, method)

    key_traj_frames = []
    for key_traj_id in key_traj_ids:
        assert dataset.trajectory_ids[key_traj_id] == key_traj_id, (
            "Trajectory ID != Trajectory Index")
        key_traj_frame = transforms(dataset.get_step_data(key_traj_id, 0))
        key_traj_frames.append(key_traj_frame)
    return key_traj_frames, key_traj_actions
