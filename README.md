# EEG P300 Mini Project

Phân loại đơn trial standard và target trong nhiệm vụ oddball thị giác (Nencki-Symfonia, GigaDB 100990), môn AIT3012 Neuroscience (GS. Guy Nagels, VUB).

Báo cáo nộp: [PDF tiếng Anh](docs/EEG_Final_Report_EN.pdf) · [nguồn LaTeX](docs/EEG_Final_Report_EN.tex) · [bản Markdown](report/REPORT_EN.md). Phân công và tỷ lệ đóng góp: [docs/contributions.md](docs/contributions.md). Khung thuyết trình: [docs/presentation_slides.md](docs/presentation_slides.md).

## Kết quả chính

39 người làm đúng nhiệm vụ, 21.878 trial đúng (13,5% target), `GroupKFold` 5 fold theo người; trung bình ± SD qua 5 fold.

| Mô hình | Balanced accuracy | ROC-AUC | AP |
|---|---|---|---|
| Logistic Regression (402 đặc trưng) | 0,875 ± 0,037 | 0,946 ± 0,021 | 0,791 ± 0,073 |
| LDA shrinkage (402 đặc trưng) | 0,815 ± 0,060 | 0,947 ± 0,022 | 0,791 ± 0,075 |
| LDA shrinkage (128 kênh × 12 bin) | 0,830 ± 0,054 | 0,949 ± 0,022 | 0,821 ± 0,059 |
| EEGNet-8,2 (epoch 128 × 301) | 0,903 ± 0,021 | 0,970 ± 0,013 | 0,865 ± 0,045 |
| EEGNet, distractor – standard (không bấm nút) | 0,855 ± 0,020 | 0,926 ± 0,015 | 0,745 ± 0,048 |

ERP: Target − Standard tại Pz, 300–500 ms, 5,05 ± 2,28 µV, t(38) = 13,82, dz = 2,21. Saliency EEGNet chỉ tính trên người test: đỉnh 504 ms, 40,7% trong 300–600 ms, tương quan không gian r = 0,253 với hiệu ứng ERP.

## Cấu trúc repository

```text
EEG/
├── README.md
├── requirements.txt                  # Thư viện Python
├── config.py                         # Đường dẫn, định dạng epoch, nhãn, danh sách người cần lưu ý
├── data/raw/ (*)                     # Dữ liệu EEG gốc
├── derivatives/v1/                   # Đầu ra pipeline tiền xử lý phiên bản v1
│   ├── epochs/ (*)                   # sub-XX_task-oddball_v1-epo.fif (Release data-v1)
│   ├── ica/ (*)                      # sub-XX-ica.fif (Release data-v1)
│   ├── features/ (*)                 # all_subjects_features.csv (21.878 trial × 402 đặc trưng)
│   ├── qc/, qc_summary.csv           # QC từng người và bảng tổng hợp
│   └── figures/                      # ERP từng người và grand_average.png
├── docs/
│   ├── EEG_Final_Report_EN.pdf/.tex  # Báo cáo tiếng Anh chính thức
│   ├── report_figures/               # Hình dùng khi biên dịch LaTeX
│   ├── contributions.md              # Vai trò V1–V6, tỷ lệ đóng góp, nhật ký công việc
│   ├── presentation_slides.md        # Khung thuyết trình 10 phút
│   └── dataset_cheatsheet.md         # Nguồn dataset
├── notebooks/
│   ├── 00_explore_dataset.ipynb      # Khảo sát dữ liệu: 42 người, montage Colin27, tỉ lệ kích thích, RT
│   ├── 01_preprocessing.ipynb        # V2: tải dữ liệu, kênh hỏng, ICA/ICLabel, lọc, epoch, QC
│   ├── 02_erp_visualization.ipynb    # V3: ERP, sóng hiệu số, topomap, kiểm tra P300
│   ├── 03_machine_learning.ipynb     # V5: benchmark ML baseline
│   └── 04_deep_learning.ipynb        # V6: EEGNet, thử nghiệm một lần chia và saliency
├── src/
│   ├── utils/io.py                   # Đọc epoch, tạo X/y/groups theo người
│   ├── preprocessing/
│   │   ├── README.md                 # Chỉ dẫn tới notebook tiền xử lý
│   │   └── audit_epochs.py           # Audit 42 file epoch và bảng đặc trưng
│   ├── erp/plot_erp.py               # V3: ERP Fz/Cz/Pz, sóng hiệu số, topomap, bảng P300 nhóm
│   ├── features/
│   │   ├── extract_features.py       # V4: 402 đặc trưng/trial (đỉnh ERP, cửa sổ, Hjorth, Welch, wavelet db4)
│   │   └── rank_features.py          # V4: xếp hạng đặc trưng theo AUC (chỉ tham khảo)
│   ├── ml/
│   │   ├── train_baseline.py         # V5: Dummy, LDA, Logistic Regression, RBF SVM, Random Forest
│   │   └── report_controls.py        # Đối chứng LDA 12 và 128 kênh
│   ├── dl/
│   │   ├── train_eegnet.py           # V6: mô hình EEGNet-8,2, huấn luyện, saliency, dữ liệu mô phỏng
│   │   ├── report_controls.py        # EEGNet target và distractor, saliency trên người test (GPU)
│   │   └── finalize_results.py       # Gộp lần chạy cuối thành bảng và hình chính thức
│   └── verify_report.py              # Tính lại toàn bộ số liệu báo cáo
├── tests/                            # 11 unit test (ERP, nhãn đối chứng, fold, saliency)
├── results/
│   ├── tables/                       # Bảng kết quả, dự đoán out-of-fold, eegnet_controls/
│   ├── figures/                      # Toàn bộ hình kết quả
│   └── models/                       # 10 checkpoint EEGNet của lần chạy cuối
└── report/
    ├── REPORT_EN.md                  # Báo cáo tiếng Anh dạng Markdown
    └── figures/                      # Hình dùng trong báo cáo
```

(*) Không commit. Epoch và mô hình ICA tải từ GitHub Release [`data-v1`](https://github.com/amebett1/EEG/releases/tag/data-v1), hướng dẫn giải nén ở mục 5 của `notebooks/01_preprocessing.ipynb`.

## Hướng dẫn thực thi

Chạy từ thư mục gốc, sau khi đã có `derivatives/v1/epochs/`.

1. Đặc trưng và xếp hạng (V4):
   ```bash
   python src/features/extract_features.py --all
   python src/features/rank_features.py
   ```
2. ERP nhóm và bảng kiểm tra P300 (V3):
   ```bash
   python src/erp/plot_erp.py --group
   ```
3. Audit dữ liệu bàn giao (V2):
   ```bash
   python -m src.preprocessing.audit_epochs
   ```
4. Baseline ML và đối chứng số kênh (V5):
   ```bash
   python src/ml/train_baseline.py --splits 5
   python -m src.ml.report_controls
   ```
5. EEGNet và saliency (V6, cần GPU CUDA):
   ```bash
   python -u -m src.dl.report_controls --epochs 50 --batch-size 64
   python -m src.dl.finalize_results
   ```
   Chạy hai bài toán target – standard và distractor – standard, mỗi bài năm fold với cùng nhóm người test như V5; checkpoint từng fold được lưu để chạy tiếp khi gián đoạn.
6. Kiểm chứng số liệu báo cáo và unit test:
   ```bash
   python -m src.verify_report
   python -m unittest discover -s tests
   ```

`python src/dl/train_eegnet.py --synthetic` chỉ dùng để thử pipeline trên dữ liệu mô phỏng. Không chạy `train_eegnet.py --save` trên dữ liệu thật vì lệnh này ghi đè bảng chính thức bằng saliency của mô hình tham chiếu cũ.
