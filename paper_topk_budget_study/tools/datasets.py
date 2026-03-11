from typing import Dict, Tuple

import mne
import numpy as np
import torch
from moabb.datasets import BNCI2014001, BNCI2014004


EEG_CHANNELS = {
    "BNCI2014001": [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    ],
    "BNCI2014004": ["C3", "Cz", "C4"],
}


def get_edge_weight_from_electrode(edge_pos: Dict[str, np.ndarray]) -> Tuple[list, np.ndarray]:
    edge_pos_value = [v for _, v in edge_pos.items()]
    num_nodes = len(edge_pos_value)
    edge_weight = np.zeros([num_nodes, num_nodes], dtype=np.float32)
    edge_index = [[], []]

    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index[0].append(i)
            edge_index[1].append(j)
            if i == j:
                edge_weight[i, j] = 0.0
            else:
                edge_weight[i, j] = np.sum(
                    [(edge_pos_value[i][k] - edge_pos_value[j][k]) ** 2 for k in range(2)]
                )

    return edge_index, edge_weight


def exponential_moving_standardize(
    data: np.ndarray,
    factor_new: float = 1e-3,
    init_block_size: int = 1000,
    eps: float = 1e-4,
) -> np.ndarray:
    # data: [channels, time]
    standardized = np.array(data, dtype=np.float32, copy=True)
    if standardized.shape[-1] <= init_block_size:
        mean = standardized.mean(axis=-1, keepdims=True)
        std = standardized.std(axis=-1, keepdims=True)
        return (standardized - mean) / np.maximum(std, eps)

    init_mean = standardized[:, :init_block_size].mean(axis=-1)
    init_var = standardized[:, :init_block_size].var(axis=-1)

    mean_t = init_mean.copy()
    var_t = init_var.copy()

    for t in range(standardized.shape[-1]):
        x_t = standardized[:, t]
        mean_t = (1.0 - factor_new) * mean_t + factor_new * x_t
        centered = x_t - mean_t
        var_t = (1.0 - factor_new) * var_t + factor_new * (centered ** 2)
        standardized[:, t] = centered / np.maximum(np.sqrt(var_t), eps)

    standardized[:, :init_block_size] = (
        standardized[:, :init_block_size] - init_mean[:, None]
    ) / np.maximum(np.sqrt(init_var)[:, None], eps)
    return standardized


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    classes = sorted(np.unique(labels).tolist())
    mapping = {label: idx for idx, label in enumerate(classes)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def _preprocess_run(raw: mne.io.BaseRaw, channels, l_freq: float, h_freq: float) -> mne.io.BaseRaw:
    raw = raw.copy().pick(channels)
    raw.load_data()
    raw._data = raw._data.astype(np.float64) * 1e6
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw._data = exponential_moving_standardize(raw._data).astype(np.float64)
    return raw


def _extract_epochs(raw: mne.io.BaseRaw, dataset: str, duration: float) -> Tuple[np.ndarray, np.ndarray]:
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    wanted = ["left_hand", "right_hand"] if dataset == "BNCI2014004" else ["left_hand", "right_hand", "feet", "tongue"]
    use_event_id = {name: event_id[name] for name in wanted}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=use_event_id,
        tmin=-0.5,
        tmax=duration,
        baseline=None,
        preload=True,
        verbose=False,
    )
    x = epochs.get_data(copy=True).astype(np.float32)
    y = epochs.events[:, -1].astype(np.int64)
    return x, y


def _load_subject_arrays(dataset: str, subject_id: int, duration: float):
    if dataset == "BNCI2014001":
        ds = BNCI2014001()
        train_session_key = "0train"
        test_session_key = "1test"
        ch_names = EEG_CHANNELS["BNCI2014001"]
    elif dataset == "BNCI2014004":
        ds = BNCI2014004()
        train_session_key = None
        test_session_key = None
        ch_names = EEG_CHANNELS["BNCI2014004"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    subject_data = ds.get_data(subjects=[subject_id])[subject_id]

    train_x_list, train_y_list = [], []
    test_x_list, test_y_list = [], []

    if dataset == "BNCI2014001":
        for run in subject_data[train_session_key].values():
            raw = _preprocess_run(run, ch_names, 0.0, 40.0)
            x, y = _extract_epochs(raw, dataset, duration)
            train_x_list.append(x)
            train_y_list.append(y)
        for run in subject_data[test_session_key].values():
            raw = _preprocess_run(run, ch_names, 0.0, 40.0)
            x, y = _extract_epochs(raw, dataset, duration)
            test_x_list.append(x)
            test_y_list.append(y)
    else:
        for session_name, runs in subject_data.items():
            session_idx = int("".join(ch for ch in session_name if ch.isdigit()) or "-1")
            target_x = train_x_list if session_idx <= 2 else test_x_list
            target_y = train_y_list if session_idx <= 2 else test_y_list
            for run in runs.values():
                raw = _preprocess_run(run, ch_names, 0.0, 40.0)
                x, y = _extract_epochs(raw, dataset, duration)
                target_x.append(x)
                target_y.append(y)

    train_X = np.concatenate(train_x_list, axis=0)
    train_y = _encode_labels(np.concatenate(train_y_list, axis=0))
    test_X = np.concatenate(test_x_list, axis=0)
    test_y = _encode_labels(np.concatenate(test_y_list, axis=0))

    montage = mne.channels.make_standard_montage("biosemi64")
    all_pos = montage.get_positions()["ch_pos"]
    electrodes_pos = {ch: np.asarray(all_pos[ch][:2]) * 100 for ch in ch_names if ch in all_pos}
    _, eu_adj = get_edge_weight_from_electrode(edge_pos=electrodes_pos)
    return train_X, train_y, test_X, test_y, eu_adj


def load_single_subject(dataset, subject_id, duration, to_tensor=True):
    train_X, train_y, test_X, test_y, eu_adj = _load_subject_arrays(dataset, subject_id, duration)

    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y, eu_adj
