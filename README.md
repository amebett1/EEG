# EEG P300 Mini Project

## Cấu trúc repository

```text
EEG/
├── README.md                         # Tổng quan và cấu trúc dự án
├── requirements.txt                  # Các thư viện Python của dự án
├── config.py                         # Đường dẫn, cấu hình epoch, nhãn và danh sách người tham gia
├── .gitignore                        # Quy tắc loại dữ liệu lớn và file tạm khỏi Git
├── data/
│   └── raw/                          # Dữ liệu EEG gốc; không commit
├── derivatives/
│   └── v1/                           # Đầu ra của phiên bản pipeline v1
│       ├── epochs/ (*)               # Epoch đã xử lý: sub-XX_task-oddball_v1-epo.fif
│       ├── ica/ (*)                  # Mô hình ICA theo người: sub-XX-ica.fif
│       ├── features/ (*)             # Nơi lưu đặc trưng theo quy ước trong config.py
│       ├── qc/                       # Kết quả kiểm tra chất lượng theo người: sub-XX.json
│       ├── qc_summary.csv            # Bảng tổng hợp kiểm tra chất lượng
│       └── figures/                  # ERP từng người và grand_average.png từ tiền xử lý
├── docs/
│   ├── dataset_cheatsheet.md         # Ghi chú và nguồn dataset
│   └── contributions.md              # Phân công vai trò (V1-V6) và nhật ký đóng góp
├── notebooks/
│   ├── 00_explore_dataset.ipynb       # Khảo sát ban đầu: liệt kê dữ liệu trong RAW_DIR
│   ├── 01_preprocessing.ipynb         # Tải dữ liệu, tiền xử lý, xuất epoch/ICA và QC
│   ├── 02_erp_visualization.ipynb     # V3: Trực quan hóa ERP, Scalp Topomap & kiểm tra P300
│   ├── 03_machine_learning.ipynb      # V5: Huấn luyện và đánh giá benchmark ML baseline
│   └── 04_deep_learning.ipynb         # V6: EEGNet trên epoch thô, CV theo người, saliency map
├── src/
│   ├── preprocessing/
│   │   └── README.md                 # Chỉ dẫn tới pipeline trong notebook
│   ├── utils/
│   │   └── io.py                     # Đọc epoch, tạo X/y/groups để chia tập theo người
│   ├── erp/
│   │   └── plot_erp.py               # V3: ERP Fz/Cz/Pz, difference waves, topomap, kiểm tra P300
│   ├── features/
│   │   ├── extract_features.py       # V4: 402 đặc trưng/trial (đỉnh ERP, biên độ cửa sổ, band power, Hjorth, entropy, wavelet)
│   │   └── rank_features.py          # V4: xếp hạng đặc trưng theo AUC -> results/tables/v4_feature_auc.csv
│   ├── ml/
│   │   └── train_baseline.py         # V5: Pipeline huấn luyện và đánh giá LDA, SVM, RF, LR
│   └── dl/
│       └── train_eegnet.py           # V6: EEGNet-8,2 (PyTorch), dữ liệu mô phỏng, CV, saliency
├── results/
│   ├── figures/                      # Nơi lưu hình kết quả phân tích ERP & topomap
│   ├── tables/                       # Nơi lưu bảng kết quả
│   └── models/                       # Nơi lưu mô hình đã huấn luyện; không commit
└── report/
    ├── README.md                     # Báo cáo phương pháp luận Polich và tiêu chuẩn P300
    └── figures/                      # Nơi lưu hình dùng trong báo cáo
## Hướng dẫn thực thi

1. **Phân tích ERP và bản đồ địa hình da đầu (V3)**:
   ```bash
   python src/erp/plot_erp.py --population --grand-average --table
   ```

2. **Xếp hạng 402 đặc trưng theo mức phân biệt AUC (V4)**:
   ```bash
   python src/features/rank_features.py
   ```

3. **Huấn luyện và đánh giá benchmark Machine Learning baseline trên 39 đối tượng (V5)**:
   ```bash
   python src/ml/train_baseline.py --save
   ```

4. **Huấn luyện mô hình Deep Learning EEGNet và phân tích Saliency map (V6)**:
   ```bash
   python src/dl/train_eegnet.py --subjects sub-01 --epochs 15 --batch-size 32 --save
   ```


