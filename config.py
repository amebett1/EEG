"""Quy ước chung cho cả nhóm. Khớp với pipeline trong notebooks/01_preprocessing.ipynb."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DERIV_DIR = ROOT / "derivatives" / "v1"      # "v1" = phiên bản pipeline, không phải vai trò V1
EPOCH_DIR = DERIV_DIR / "epochs"             # tải từ GitHub Release data-v1, không commit
QC_DIR = DERIV_DIR / "qc"
QC_SUMMARY = DERIV_DIR / "qc_summary.csv"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"
FEATURE_DIR = DERIV_DIR / "features"         # V4 xuất đặc trưng ra đây (không commit)

# --- Định dạng epoch ---
EPOCH_FILE = "{sub}_task-oddball_v1-epo.fif"  # sub = "sub-01" ... "sub-42"
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 43)]
TMIN, TMAX = -0.2, 1.0                        # giây
BASELINE = (-0.2, 0.0)
SFREQ = 250                                   # Hz (1000 Hz decim 4), mỗi epoch 301 mẫu
N_CHANNELS = 128                              # cùng thứ tự kênh ở cả 42 người

# --- Nhãn: ba điều kiện, không phải hai ---
EVENT_ID = {"standard": 5, "target": 6, "distractor": 7}
BINARY_CONDITIONS = ("standard", "target")    # nhiệm vụ phân loại chính (distractor bị loại)

# Người cần lưu ý (theo paper dataset và QC của pipeline)
TASK_MISUNDERSTOOD = {"sub-02", "sub-14", "sub-17"}   # hiểu sai nhiệm vụ, nhiều false alarm
HIGH_IMPEDANCE = {"sub-23", "sub-24", "sub-32"}
PAPER_LISTED_IRREGULAR = {"sub-13", "sub-23", "sub-24", "sub-37", "sub-38"}

RANDOM_SEED = 42
