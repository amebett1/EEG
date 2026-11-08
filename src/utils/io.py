import mne
import numpy as np

from config import (BINARY_CONDITIONS, EPOCH_DIR, EPOCH_FILE, EVENT_ID,
                    SUBJECTS, TASK_MISUNDERSTOOD)


def epoch_path(sub: str):
    return EPOCH_DIR / EPOCH_FILE.format(sub=sub)


def load_epochs(sub: str, conditions=BINARY_CONDITIONS, correct_only=True) -> mne.Epochs:
    ep = mne.read_epochs(epoch_path(sub), preload=True, verbose="error")
    if correct_only:
        ep = ep[ep.metadata["correct"].values]
    return ep[list(conditions)]


def load_dataset(subjects=None, conditions=BINARY_CONDITIONS, correct_only=True,
                 exclude_misunderstood=True):
    subjects = list(subjects or SUBJECTS)
    if exclude_misunderstood:
        subjects = [s for s in subjects if s not in TASK_MISUNDERSTOOD]
    X, y, groups = [], [], []
    for i, sub in enumerate(subjects):
        ep = load_epochs(sub, conditions, correct_only)
        X.append(ep.get_data())
        y.append((ep.events[:, 2] == EVENT_ID["target"]).astype(int))
        groups.append(np.full(len(ep), i))
    return np.concatenate(X), np.concatenate(y), np.concatenate(groups)
