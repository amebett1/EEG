# Phân công & Ghi nhận Đóng góp (Contributions)

Dự án nghiên cứu và phân loại sóng P300 trên tín hiệu EEG (Oddball Paradigm - 42 đối tượng, GigaDB 100990).

---

## 1. Cơ cấu Phân công Vai trò (Team Roles)

| Vai trò | Trách nhiệm chính | File mã nguồn / Tài liệu phụ trách | Trạng thái |
|:---:|:---|:---|:---:|
| **V1** | **Tiền xử lý dữ liệu (Preprocessing)**<br>- Lọc dải tần 0.5 - 40 Hz<br>- Phát hiện và nội suy kênh hỏng (IQR, tương quan lân cận)<br>- Picard ICA kết hợp ICLabel loại bỏ artifact (mắt, cơ, tim)<br>- Cắt epoch (-200..+1000 ms), decimation 4 (1000 Hz -> 250 Hz)<br>- Xuất file QC từng người và `qc_summary.csv` | `notebooks/01_preprocessing.ipynb`<br>`src/preprocessing/`<br>`derivatives/v1/qc/` | **Hoàn thành** |
| **V3** | **Trực quan hóa & Phân tích ERP (Visualization / ERP)**<br>- Đọc và tổng hợp lý thuyết Polich (2007) về P3a/P3b<br>- Code vẽ dạng sóng ERP đường giữa (**Fz, Cz, Pz**) với 3 điều kiện<br>- Code vẽ sóng hiệu số (**Difference Waves** Target-Std, Distractor-Std)<br>- Code vẽ bản đồ địa hình điện thế da đầu (**Scalp Topomap** 128 kênh)<br>- Kiểm tra định lượng các tiêu chuẩn P300 (Parietal gradient, Oddball effect)<br>- Trực quan hóa và báo cáo kết quả trên đối tượng thử nghiệm | `src/erp/plot_erp.py`<br>`notebooks/02_erp_visualization.ipynb`<br>`results/figures/` | **Hoàn thành (Tuần 1)** |
| **V4** | **Trích xuất đặc trưng (Feature Extraction)**<br>- Biên độ đỉnh (Peak amplitude) và độ trễ (Peak latency)<br>- Biên độ trung bình các cửa sổ thời gian quan trọng<br>- Năng lượng dải sóng (Power Spectral Density: Delta, Theta, Alpha, Beta)<br>- Đặc trưng phi tuyến (Entropy, Wavelet transforms) | `src/features/extract_features.py`<br>`derivatives/v1/features/` | Kế hoạch Tuần 2 |
| **V5** | **Mô hình học máy truyền thống (Machine Learning Baseline)**<br>- Huấn luyện SVM, LDA, Random Forest<br>- Chia tập kiểm định theo người (Leave-One-Subject-Out / GroupKFold)<br>- Xử lý mất cân bằng lớp (Class weights, resampling) | `src/ml/train_baseline.py`<br>`results/models/`<br>`results/tables/` | Kế hoạch Tuần 3 |
| **V6** | **Mô hình học sâu & Giải thích (Deep Learning & Interpretability)**<br>- Huấn luyện kiến trúc chuyên dụng EEGNet (Braindecode / PyTorch)<br>- Saliency maps, Layer-wise Relevance Propagation giải thích mô hình | `src/dl/train_eegnet.py` | Kế hoạch Tuần 4 |

---

## 2. Nhật ký công việc Tuần 1 (3–9/11): Vai trò V3

- **Nhiệm vụ theo kế hoạch**: *Đọc Polich, viết sẵn code ERP và topomap, vẽ thử trên 1 người.*
- **Chi tiết thực hiện**:
  1. **Nghiên cứu tài liệu Polich (2007)**:
     - Phân định rõ cơ chế sinh lý và phân bố không gian của **P3b** (cực đại vùng đỉnh Pz, độ trễ 300–600 ms, gắn với context updating) và **P3a** (cực đại vùng trán/trung tâm Fz/Cz, sớm hơn, gắn với orienting reflex).
     - Áp dụng phương pháp sóng hiệu số (*Difference Waves*) để cô lập P300 khỏi các thành phần giác quan cơ bản.
  2. **Xây dựng module `src/erp/plot_erp.py`**:
     - `plot_midline_erp`: Vẽ Fz, Cz, Pz với dải sai số chuẩn (SEM), tô sáng cửa sổ [300, 600] ms và đánh dấu tọa độ đỉnh (biên độ, độ trễ).
     - `plot_difference_waves`: Trực quan hóa sóng hiệu số so sánh trực diện P3a và P3b.
     - `plot_erp_topomap`: Vẽ bản đồ địa hình da đầu 128 kênh (montage Colin27) qua các mốc thời gian từ 150 ms đến 550 ms.
     - `plot_p300_summary_topomap`: Topomap trung bình toàn cửa sổ P300 chỉ ra tiêu điểm dương tính vùng chẩm-đỉnh.
     - `verify_p300`: Kiểm chứng tự động 4 quy tắc Polich và xuất báo cáo chẩn đoán logic.
  3. **Tạo Notebook tương tác `notebooks/02_erp_visualization.ipynb`**:
     - Cho phép chạy trực quan từng phần, lưu hình ảnh độ phân giải cao vào `results/figures/`.
  4. **Thử nghiệm trên dữ liệu**:
     - Hoàn thiện thử nghiệm và kiểm chứng định lượng trên đối tượng đầu tiên (`sub-01`).

