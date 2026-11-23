# EEG P300 Mini Project

Báo cáo cuối để nộp: [PDF tiếng Anh](docs/EEG_Final_Report_EN.pdf) · [nguồn LaTeX](docs/EEG_Final_Report_EN.tex) · [Markdown](report/REPORT_EN.md).

Đối chứng bổ sung và kiểm chứng kết quả từ thư mục gốc, trong môi trường ảo hiện có:

```bash
python -m src.preprocessing.audit_epochs
python src/ml/train_baseline.py --splits 5
python -m src.ml.report_controls
python -u -m src.dl.report_controls --epochs 50 --batch-size 64
python -m src.dl.finalize_results
python -m src.verify_report
python -m unittest discover -s tests
```

Kết quả cuối: EEGNet target–standard ROC-AUC 0.970 ± 0.013, balanced accuracy 90.3%; LDA 128 kênh ROC-AUC 0.949 ± 0.022. Đã hoàn tất EEGNet distractor–standard không bấm nút (ROC-AUC 0.926 ± 0.015) và saliency test đủ năm fold. Kết quả canonical dùng lần chạy hiện tại; cấu hình và từng fold ở `results/tables/eegnet_controls/`, kiểm chứng ở `results/tables/report_verification.json`. (Các lệnh EEGNet và 3 test PyTorch cần chạy trong môi trường có PyTorch + CUDA.)

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
│   ├── contributions.md              # Tỷ lệ đóng góp, phân công vai trò (V1-V6) và nhật ký công việc
│   └── presentation_slides.md        # Khung thuyết trình 10 phút (8 slide, người nói, kịch bản tiếng Anh)
├── notebooks/
│   ├── 00_explore_dataset.ipynb       # Khảo sát dữ liệu: 42 người, sfreq, montage Colin27, tỉ lệ kích thích, RT
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
    ├── REPORT_EN.md                  # Báo cáo tiếng Anh dạng Markdown
    └── figures/                      # Nơi lưu hình dùng trong báo cáo
```

## Hướng dẫn thực thi

1. **Phân tích ERP và bản đồ địa hình da đầu (V3)**:
   ```bash
   python src/erp/plot_erp.py --group
   # Hoặc chạy theo từng đối tượng: python src/erp/plot_erp.py --sub sub-01
   ```

2. **Xếp hạng 402 đặc trưng theo mức phân biệt AUC (V4)**:
   ```bash
   python src/features/rank_features.py
   ```

3. **Huấn luyện và đánh giá benchmark Machine Learning baseline trên 39 đối tượng (V5)**:
   ```bash
   python src/ml/train_baseline.py --save-models
   ```

4. **Huấn luyện mô hình Deep Learning EEGNet và phân tích Saliency map (V6)**:
   ```bash
   python -u -m src.dl.report_controls --epochs 50 --batch-size 64
   python -m src.dl.finalize_results
   python -m src.verify_report
   ```
   Chạy hai bài toán target–standard và distractor–standard, mỗi bài toán năm fold trên 39 người, cùng nhóm test như V5. Lệnh yêu cầu GPU CUDA và lưu checkpoint từng fold để tiếp tục khi gián đoạn. Bước finalize kiểm chứng hai bài toán rồi thay bảng, dự đoán và hình EEGNet chính thức; verify kiểm tra lại toàn bộ số liệu báo cáo.


