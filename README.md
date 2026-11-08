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
│   └── contributions.md              # File dành cho ghi nhận đóng góp; hiện còn trống
├── notebooks/
│   ├── 00_explore_dataset.ipynb       # Khảo sát ban đầu: liệt kê dữ liệu trong RAW_DIR
│   └── 01_preprocessing.ipynb         # Tải dữ liệu, tiền xử lý, xuất epoch/ICA và QC
├── src/
│   ├── preprocessing/
│   │   └── README.md                 # Chỉ dẫn tới pipeline trong notebook
│   ├── utils/
│   │   └── io.py                     # Đọc epoch, tạo X/y/groups để chia tập theo người
│   ├── erp/
│   │   └── plot_erp.py               # Khung TODO: ERP Fz/Pz, topomap, kiểm tra P300
│   ├── features/
│   │   └── extract_features.py       # Khung TODO: biên độ đỉnh, latency, band power
│   ├── ml/
│   │   └── train_baseline.py         # Khung TODO: SVM, LDA, Random Forest
│   └── dl/
│       └── train_eegnet.py           # Khung TODO: EEGNet và giải thích mô hình
├── results/
│   ├── figures/                      # Nơi lưu hình kết quả phân tích
│   ├── tables/                       # Nơi lưu bảng kết quả
│   └── models/                       # Nơi lưu mô hình đã huấn luyện; không commit
└── report/
    ├── README.md                     # File dành cho nội dung báo cáo; hiện còn trống
    └── figures/                      # Nơi lưu hình dùng trong báo cáo
```


