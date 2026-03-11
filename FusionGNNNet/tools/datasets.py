"""
datasets.py — MOABB-based EEG data loading (same as NexusNet baseline).
"""
from typing import Dict, Iterable, Tuple

import mne
import numpy as np
import torch
from moabb.datasets import BNCI2014001, BNCI2014004
from moabb.paradigms import LeftRightImagery, MotorImagery


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


def _build_dataset_and_paradigm(dataset: str, duration: float):
    if dataset == "BNCI2014001":
        ds = BNCI2014001()
        paradigm = MotorImagery(
            n_classes=4,
            fmin=0.0,
            fmax=40.0,
            tmin=-0.5,
            tmax=duration,
            baseline=None,
            resample=250,
        )
    elif dataset == "BNCI2014004":
        ds = BNCI2014004()
        paradigm = LeftRightImagery(
            fmin=0.0,
            fmax=40.0,
            tmin=-0.5,
            tmax=duration,
            baseline=None,
            resample=250,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return ds, paradigm


def _session_mask(dataset: str, sessions: Iterable[str], phase: str) -> np.ndarray:
    sessions = np.asarray(list(sessions))
    lower_sessions = np.array([str(s).lower() for s in sessions])

    if dataset == "BNCI2014001":
        if phase == "train":
            return np.array(["train" in s or s.startswith("0") for s in lower_sessions])
        return np.array(["test" in s or s.startswith("1") for s in lower_sessions])

    if dataset == "BNCI2014004":
        numeric_suffix = []
        for s in lower_sessions:
            digits = "".join(ch for ch in s if ch.isdigit())
            numeric_suffix.append(int(digits) if digits else -1)
        numeric_suffix = np.asarray(numeric_suffix)
        if phase == "train":
            return numeric_suffix <= 2
        return numeric_suffix >= 3

    raise ValueError(f"Unsupported dataset: {dataset}")


def _encode_labels(labels: np.ndarray) -> np.ndarray:
    classes = sorted(np.unique(labels).tolist())
    mapping = {label: idx for idx, label in enumerate(classes)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def _load_subject_arrays(dataset: str, subject_id: int, duration: float):
    ds, paradigm = _build_dataset_and_paradigm(dataset, duration)
    X, y, metadata = paradigm.get_data(dataset=ds, subjects=[subject_id])

    session_mask_train = _session_mask(dataset, metadata["session"].astype(str).to_numpy(), "train")
    session_mask_test = _session_mask(dataset, metadata["session"].astype(str).to_numpy(), "test")

    train_X = np.asarray(X[session_mask_train], dtype=np.float32)
    train_y = _encode_labels(np.asarray(y[session_mask_train]))
    test_X = np.asarray(X[session_mask_test], dtype=np.float32)
    test_y = _encode_labels(np.asarray(y[session_mask_test]))

    ch_names = EEG_CHANNELS[dataset]

    montage = mne.channels.make_standard_montage("biosemi64")
    all_pos = montage.get_positions()["ch_pos"]
    electrodes_pos = {ch: np.asarray(all_pos[ch][:2]) * 100 for ch in ch_names if ch in all_pos}
    _, eu_adj = get_edge_weight_from_electrode(edge_pos=electrodes_pos)
    return train_X, train_y, test_X, test_y, eu_adj, ch_names


def load_single_subject(dataset, subject_id, duration, to_tensor=True):
    train_X, train_y, test_X, test_y, eu_adj, _ = _load_subject_arrays(dataset, subject_id, duration)

    if to_tensor:
        train_X = torch.tensor(train_X, dtype=torch.float32)
        test_X = torch.tensor(test_X, dtype=torch.float32)
        train_y = torch.tensor(train_y, dtype=torch.int64)
        test_y = torch.tensor(test_y, dtype=torch.int64)
    return train_X, train_y, test_X, test_y, eu_adj
