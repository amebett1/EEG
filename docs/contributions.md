# Phân công & Ghi nhận Đóng góp (Contributions)

Dự án nghiên cứu và phân loại sóng P300 trên tín hiệu EEG (Oddball Paradigm - 42 đối tượng, GigaDB 100990).

---

## 1. Cơ cấu Phân công Vai trò (Team Roles)

| Vai trò | Trách nhiệm chính | File mã nguồn / Tài liệu phụ trách | Trạng thái |
|:---:|:---|:---|:---:|
| **V1** | **Tiền xử lý dữ liệu (Preprocessing)**<br>- Lọc dải tần 0.5 - 40 Hz<br>- Phát hiện và nội suy kênh hỏng (IQR, tương quan lân cận)<br>- Picard ICA kết hợp ICLabel loại bỏ artifact (mắt, cơ, tim)<br>- Cắt epoch (-200..+1000 ms), decimation 4 (1000 Hz -> 250 Hz)<br>- Xuất file QC từng người và `qc_summary.csv` | `notebooks/01_preprocessing.ipynb`<br>`src/preprocessing/`<br>`derivatives/v1/qc/` | **Hoàn thành** |
| **V3** | **Trực quan hóa & Phân tích ERP (Visualization / ERP)**<br>- **Người phụ trách**: **Trương Huy Hoàng**<br>- **Nhiệm vụ chi tiết**: Grand average ERP (Fz, Pz), scalp map, kiểm tra P300 ở khoảng 300–500 ms<br>- **Đầu ra bàn giao**: Hình ERP và topomap chất lượng cao cho báo cáo, slide<br>- Code vẽ dạng sóng ERP đường giữa (**Fz, Pz**, Fz/Cz/Pz) kèm sai số chuẩn SEM<br>- Code vẽ sóng hiệu số (**Difference Waves** Target-Std, Distractor-Std)<br>- Code vẽ bản đồ địa hình điện thế da đầu (**Scalp Topomap** 128 kênh)<br>- Kiểm tra định lượng tiêu chuẩn Polich trong cửa sổ 300–500 ms (Parietal gradient, Oddball effect) | `src/erp/plot_erp.py`<br>`notebooks/02_erp_visualization.ipynb`<br>`results/figures/`<br>`report/figures/` | **Hoàn thành (Tuần 1)** |
| **V4** | **Trích xuất đặc trưng (Feature Extraction)**<br>- **Người phụ trách**: **Nguyễn Công Kiên**<br>- Biên độ đỉnh (Peak amplitude) và độ trễ (Peak latency) của P300 và N100/P200/N200 theo Laton et al. (2014)<br>- Biên độ trung bình các cửa sổ thời gian quan trọng<br>- Năng lượng dải sóng (Delta, Theta, Alpha, Beta) và tham số Hjorth<br>- Đặc trưng phi tuyến và thời gian - tần số (Spectral entropy, Wavelet db4)<br>- **Đầu ra bàn giao**: ma trận đặc trưng 21.878 trial × 402 đặc trưng (39 người) cho V5 | `src/features/extract_features.py`<br>`src/features/rank_features.py`<br>`derivatives/v1/features/`<br>`results/tables/v4_feature_auc.csv` | **Hoàn thành (Tuần 2)** |
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
     - `plot_midline_erp`: Vẽ Fz, Cz, Pz với dải sai số chuẩn (SEM), tô sáng cửa sổ [300, 500] ms và đánh dấu tọa độ đỉnh (biên độ, độ trễ).
     - `plot_fz_pz_erp`: Vẽ tập trung 2 kênh chuẩn **Fz** và **Pz** với font chữ lớn chuẩn slide và báo cáo.
     - `plot_grand_average_erp`: Vẽ sóng **Grand Average ERP (Fz, Pz)**.
     - `plot_difference_waves`: Trực quan hóa sóng hiệu số so sánh trực diện P3a và P3b.
     - `plot_erp_topomap`: Vẽ bản đồ địa hình da đầu 128 kênh (montage Colin27) qua các mốc thời gian từ 150 ms đến 550 ms.
     - `plot_p300_summary_topomap`: Scalp map trung bình toàn cửa sổ P300 [300, 500] ms chỉ ra tiêu điểm dương tính vùng đỉnh.
     - `verify_p300`: Kiểm chứng tự động 4 quy tắc Polich trong khoảng 300–500 ms và xuất báo cáo logic.
  3. **Tạo Notebook tương tác `notebooks/02_erp_visualization.ipynb`**:
     - Cho phép chạy trực quan từng phần, lưu toàn bộ hình ảnh độ phân giải cao vào `results/figures/` và `report/figures/`.
  4. **Thử nghiệm & Bàn giao**:
     - Hoàn thiện thử nghiệm và kiểm chứng định lượng trên đối tượng đầu tiên (`sub-01`).
     - Chuẩn bị đầy đủ bộ hình ERP và topomap sẵn sàng chèn vào báo cáo và slide thuyết trình.

---

## 3. Nhật ký công việc Tuần 1–2 (3–16/11): Vai trò V4

- **Người phụ trách**: **Nguyễn Công Kiên**
- **Nhiệm vụ theo kế hoạch**: *Tuần 1: đọc paper Laton, viết code đặc trưng trên dữ liệu thử. Tuần 2: trích đặc trưng toàn bộ (đỉnh, latency, band power; entropy/wavelet nếu kịp).*
- **Chi tiết thực hiện**:
  1. **Tiếp nhận bản đầu** `src/features/extract_features.py` do Trương Huy Hoàng viết (8–9/11): biên độ theo cửa sổ, đỉnh P3, band power, Hjorth, spectral entropy.
  2. **Kiểm tra cửa sổ P3 trên 10 người**: với 300–500 ms, 5/10 người có đỉnh P300 ở Pz sau 500 ms (560–690 ms) nên đặc trưng bị cắt mất đỉnh. Đổi sang **300–600 ms** theo paper dataset và Laton et al. (2014) (P300 thị giác: 350–600 ms). AUC của `Pz_p3_window_mean` tăng từ 0,77 lên 0,81, tốt hơn ở 9/10 người.
  3. **Tìm đỉnh theo phương pháp Laton et al. (2014)**: đỉnh là cực trị cục bộ, dùng tùy chọn "force" khi cửa sổ chỉ có sườn dốc. Tỉ lệ trial target có đỉnh P3 nằm ở mép cửa sổ còn 2,5%.
  4. **Thêm đỉnh N100/P200/N200** ở Fz, Cz, Pz theo Bảng 2 của Laton et al. (2014).
  5. **Thêm wavelet db4** (thư viện PyWavelets): năng lượng các dải 0–8, 8–16, 16–31 Hz trong cửa sổ P3.
  6. **Chạy toàn bộ 39 người**: 21.878 trial × 402 đặc trưng, không có giá trị thiếu. Xếp hạng đặc trưng bằng `src/features/rank_features.py` → `results/tables/v4_feature_auc.csv`.
  7. **Viết phần 5 "Trích xuất đặc trưng"** trong `report/README.md`.
- **Bàn giao cho V5**: `derivatives/v1/features/all_subjects_features.csv` (khoảng 150 MB, không commit). Tạo lại bằng `python src/features/extract_features.py --all` (khoảng 7 phút). Hàm `load_features()` trả về `X, y, groups, feature_names`.

