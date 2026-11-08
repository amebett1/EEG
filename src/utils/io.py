import mne
import numpy as np

from config import (BINARY_CONDITIONS, EPOCH_DIR, EPOCH_FILE, EVENT_ID,
                    SUBJECTS, TASK_MISUNDERSTOOD)


def epoch_path(sub: str):
    return EPOCH_DIR / EPOCH_FILE.format(sub=sub)


def load_epochs(sub: str, conditions=BINARY_CONDITIONS, correct_only=True) -> mne.Epochs:
    """Đọc epoch của một người. Dữ liệu đơn vị volt (nhân 1e6 để ra µV).
    correct_only=True chỉ giữ trial trả lời đúng (cột metadata `correct`)."""
    ep = mne.read_epochs(epoch_path(sub), preload=True, verbose="error")
    if correct_only:
        ep = ep[ep.metadata["correct"].values]
    return ep[list(conditions)]


def load_dataset(subjects=None, conditions=BINARY_CONDITIONS, correct_only=True,
                 exclude_misunderstood=True):
    """Trả về X (n_epoch, n_kênh, n_mẫu), y (1 = target), groups (chỉ số người).
    Dùng groups với GroupKFold / LeaveOneGroupOut để chia train/test theo người."""
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
