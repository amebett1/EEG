# Phân công & Ghi nhận Đóng góp (Contributions)

Dự án nghiên cứu và phân loại sóng P300 trên tín hiệu EEG (Oddball Paradigm - 42 đối tượng, GigaDB 100990).

---

## 1. Cơ cấu Phân công Vai trò (Team Roles)

| Vai trò | Trách nhiệm chính | File mã nguồn / Tài liệu phụ trách | Trạng thái |
|:---:|:---|:---|:---:|
| **V1** | **Tiền xử lý dữ liệu (Preprocessing)**<br>- Lọc dải tần 0.5 - 40 Hz<br>- Phát hiện và nội suy kênh hỏng (IQR, tương quan lân cận)<br>- Picard ICA kết hợp ICLabel loại bỏ artifact (mắt, cơ, tim)<br>- Cắt epoch (-200..+1000 ms), decimation 4 (1000 Hz -> 250 Hz)<br>- Xuất file QC từng người và `qc_summary.csv` | `notebooks/01_preprocessing.ipynb`<br>`src/preprocessing/`<br>`derivatives/v1/qc/` | **Hoàn thành** |
| **V3** | **Trực quan hóa & Phân tích ERP (Visualization / ERP)**<br>- **Người phụ trách**: **Trương Huy Hoàng**<br>- **Nhiệm vụ chi tiết**: Grand average ERP (Fz, Pz), scalp map, kiểm tra P300 ở khoảng 300–500 ms<br>- **Đầu ra bàn giao**: Hình ERP và topomap chất lượng cao cho báo cáo, slide<br>- Code vẽ dạng sóng ERP đường giữa (**Fz, Pz**, Fz/Cz/Pz) kèm sai số chuẩn SEM<br>- Code vẽ sóng hiệu số (**Difference Waves** Target-Std, Distractor-Std)<br>- Code vẽ bản đồ địa hình điện thế da đầu (**Scalp Topomap** 128 kênh)<br>- Kiểm tra định lượng tiêu chuẩn Polich trong cửa sổ 300–500 ms (Parietal gradient, Oddball effect) | `src/erp/plot_erp.py`<br>`notebooks/02_erp_visualization.ipynb`<br>`results/figures/`<br>`report/figures/`<br>`results/tables/v3_p300_group_results.csv` | **Hoàn thành (Tuần 1, 2, 3)** |
| **V4** | **Trích xuất đặc trưng (Feature Extraction)**<br>- **Người phụ trách**: **Nguyễn Công Kiên**<br>- Biên độ đỉnh (Peak amplitude) và độ trễ (Peak latency) của P300 và N100/P200/N200 theo Laton et al. (2014)<br>- Biên độ trung bình các cửa sổ thời gian quan trọng<br>- Năng lượng dải sóng (Delta, Theta, Alpha, Beta) và tham số Hjorth<br>- Đặc trưng phi tuyến và thời gian - tần số (Spectral entropy, Wavelet db4)<br>- **Đầu ra bàn giao**: ma trận đặc trưng 21.878 trial × 402 đặc trưng (39 người) cho V5 | `src/features/extract_features.py`<br>`src/features/rank_features.py`<br>`derivatives/v1/features/`<br>`results/tables/v4_feature_auc.csv` | **Hoàn thành (Tuần 2)** |
| **V5** | **Mô hình học máy truyền thống (Machine Learning Baseline)**<br>- Huấn luyện Dummy, LDA, Logistic Regression, RBF SVM, Random Forest<br>- Chia tập kiểm định chống rò rỉ dữ liệu (GroupKFold theo người / StratifiedKFold)<br>- Xử lý mất cân bằng lớp (Class weights, balanced accuracy, PR-AUC, ROC-AUC) | `src/ml/train_baseline.py`<br>`notebooks/03_machine_learning.ipynb`<br>`results/tables/`<br>`results/figures/` | **Hoàn thành (Tuần 3)** |
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

## 3. Nhật ký công việc Tuần 2 (10–16/11): Vai trò V3

- **Người phụ trách**: **Trương Huy Hoàng**
- **Nhiệm vụ theo kế hoạch**: *ERP Fz/Pz và scalp map toàn bộ nhóm, kiểm tra P300, viết phần Results (ERP).*
- **Chi tiết thực hiện**:
  1. **Grand Average ERP (Fz, Pz) toàn bộ nhóm**:
     - Phân tích dạng sóng trên toàn bộ 39 đối tượng chuẩn tại Fz và Pz trong cửa sổ $[300, 500]\text{ ms}$.
     - Xác nhận hiệu ứng Oddball cực kỳ mạnh mẽ: chênh lệch Target − Standard tại Pz đạt $+5,64 \pm 2,12\,\mu\text{V}$, kiểm định Student $t(38) = 16,635$, $p = 4,91 \times 10^{-19}$, Cohen's $d = 2,66$.
     - Khẳng định tính chọn lọc không gian: Pz dương tính cực đại ($+7,14\,\mu\text{V}$), Fz âm tính phân cực.
  2. **Scalp Topomap 128 kênh toàn bộ nhóm**:
     - Trực quan hóa diễn tiến không gian từ chẩm (cảm giác sơ cấp) đến đỉnh (P3b nhận thức) qua các mốc thời gian 150–550 ms.
     - Xác lập bản đồ trung bình cửa sổ P300 [300, 500] ms với tiêu điểm dương tính đỉnh (Parietal Hotspot).
  3. **Kiểm định định lượng Polich (2007) trên 42 người**:
     - $100\%$ đối tượng bình thường (39/39) thỏa mãn tiêu chuẩn Oddball và Gradient đỉnh.
     - Phát hiện tương quan tuyến tính nghịch có ý nghĩa giữa thời gian phản ứng Target RT và biên độ Pz ($r = -0,434, p = 0,0058$): phản xạ nhanh đi kèm với biên độ P3b lớn.
     - Làm rõ cơ chế suy giảm ở nhóm hiểu sai nhiệm vụ (sub-02/14/17): False Alarm $> 90\%$, biên độ Target bị suy giảm còn $4,99\,\mu\text{V}$.
  4. **Xuất bảng kết quả và soạn thảo báo cáo**:
     - Xuất bảng kết quả kiểm định 42 đối tượng ra file `results/tables/v3_p300_group_results.csv`.
     - Soạn thảo toàn bộ phần 8 "Kết quả Nghiên cứu & Phân tích ERP (Results – ERP)" trong `report/README.md`.

---

## 4. Nhật ký công việc Tuần 3 (17–23/11): Vai trò V3

- **Người phụ trách**: **Trương Huy Hoàng**
- **Nhiệm vụ theo kế hoạch**: *Chọn hình cho báo cáo và slide.*
- **Chi tiết thực hiện**:
  1. Tuyển chọn 4 bộ hình ảnh cốt lõi chuẩn slide 16:9, độ phân giải cao 300 DPI:
     - Slide 1: Grand Average ERP (Fz, Pz) tô sáng cửa sổ 300–500 ms và sóng hiệu số.
     - Slide 2: Động học không gian Scalp Topomap 128 kênh và tiêu điểm đỉnh.
     - Slide 3: Phân tách cơ chế P3a vs P3b bằng sóng hiệu số Difference Waves.
     - Slide 4: Thống kê quần thể 42 đối tượng, so sánh nhóm hiểu sai task và tương quan RT vs P3b.
  2. Biên soạn đầy đủ Figure Captions chuẩn bài báo khoa học và các thông điệp thuyết trình trọng tâm (Key Takeaways) cho từng slide tại mục 9 của `report/README.md`.
  3. Lưu trữ và đồng bộ toàn bộ file ảnh sẵn dùng tại `report/figures/` và `results/figures/`.

---

## 5. Nhật ký công việc Tuần 1–2 (3–16/11): Vai trò V4

- **Người phụ trách**: **Nguyễn Công Kiên**
- **Nhiệm vụ theo kế hoạch**: *Tuần 1: đọc paper Laton, viết code đặc trưng trên dữ liệu thử. Tuần 2: trích đặc trưng toàn bộ (đỉnh, latency, band power; entropy/wavelet nếu kịp).*
- **Chi tiết thực hiện**:
  1. **Tiếp nhận bản đầu** `src/features/extract_features.py` do Trương Huy Hoàng viết (8–9/11): biên độ theo cửa sổ, đỉnh P3, band power, Hjorth, spectral entropy.
  2. **Kiểm tra cửa sổ P3 trên 10 người**: với 300–500 ms, 5/10 người có đỉnh P300 ở Pz sau 500 ms (560–690 ms) nên đặc trưng bị cắt mất đỉnh. Đổi sang **300–600 ms** theo paper dataset và Laton et al. (2014) (P300 thị giác: 350–600 ms). AUC của `Pz_p3_window_mean` tăng từ 0,77 lên 0,81, tốt hơn ở 9/10 người.
  3. **Tìm đỉnh theo phương pháp Laton et al. (2014)**: đỉnh là cực trị cục bộ, dùng tùy chọn "force" khi cửa sổ chỉ có sườn dốc. Ở Pz, tỉ lệ trial target có đỉnh P3 nằm ở mép cửa sổ còn 2,5%.
  4. **Thêm đỉnh N100/P200/N200** ở Fz, Cz, Pz theo Bảng 2 của Laton et al. (2014).
  5. **Thêm wavelet db4** (thư viện PyWavelets): năng lượng các dải 0–8, 8–16, 16–31 Hz trong cửa sổ P3.
  6. **Chạy toàn bộ 39 người**: 21.878 trial × 402 đặc trưng, không có giá trị thiếu. Xếp hạng đặc trưng bằng `src/features/rank_features.py` → `results/tables/v4_feature_auc.csv`.
  7. **Viết phần 5 "Trích xuất đặc trưng"** trong `report/README.md`.
- **Bàn giao cho V5**: `derivatives/v1/features/all_subjects_features.csv` (khoảng 150 MB, không commit). Tạo lại bằng `python src/features/extract_features.py --all` (khoảng 7 phút). Hàm `load_features()` trả về `X, y, groups, feature_names`.

---

## 6. Nhật ký công việc Tuần 3: Vai trò V5

- **Nhiệm vụ theo kế hoạch**: *Xây dựng pipeline học máy truyền thống (Baseline) phân loại P300 đơn trial (single-trial), kiểm định Cross-Validation không rò rỉ dữ liệu, đánh giá và so sánh đa mô hình.*
- **Chi tiết thực hiện**:
  1. **Xây dựng module dòng lệnh `src/ml/train_baseline.py` và Notebook tương tác `notebooks/03_machine_learning.ipynb`**:
     - Đồng bộ cả 2 hình thức: script thực thi tự động qua CLI có `argparse` và notebook trực quan hóa từng bước.
  2. **Thiết lập Pipeline chống rò rỉ dữ liệu (No Data Leakage)**:
     - Tích hợp `StandardScaler` bên trong `Pipeline` của `scikit-learn`, đảm bảo chỉ fit trên tập huấn luyện của từng fold.
     - Hỗ trợ cơ chế phân chia linh hoạt: tự động chia `GroupKFold(n_splits=5)` theo `subject_id` khi có nhiều người và fallback an toàn sang `StratifiedKFold(n_splits=5)` khi chạy thử nghiệm trên 1 người (`sub-01`).
  3. **Xử lý mất cân bằng lớp nghiêm trọng (Prevalence 11,3% - 13,5%)**:
     - Cấu hình `class_weight="balanced"` cho Logistic Regression, SVM và `balanced_subsample` cho Random Forest.
     - Đánh giá bằng bộ chỉ số đa chiều: Balanced Accuracy, ROC-AUC, PR-AUC, F1-macro, Sensitivity và Specificity (loại bỏ việc chỉ dựa vào raw accuracy dễ gây ngộ nhận).
  4. **Triển khai và đánh giá 5 mô hình benchmark**:
     - `DummyClassifier` (Stratified chance level).
     - `LinearDiscriminantAnalysis` (Bộ giải `lsqr` kèm tự động điều chuẩn co rút shrinkage - Gold standard cho BCI).
     - `LogisticRegression` (Điều chuẩn L2).
     - `SVC` (RBF kernel, xác suất Platl).
     - `RandomForestClassifier` (100 cây quyết định, tối ưu hóa đa luồng `n_jobs=-1`).
  5. **Xuất bản kết quả thực nghiệm và biểu đồ khoa học**:
     - Bảng số liệu chi tiết: `results/tables/baseline_metrics.csv`.
     - Bộ 3 biểu đồ chuẩn học thuật tiếng Anh Sentence Case (300 DPI): `baseline_model_comparison.png`, `baseline_roc_curves.png`, `baseline_pr_curves.png` được lưu tại cả `results/figures/` và `report/figures/`.
  6. **Viết phần 6 "Mô hình Học máy Truyền thống Baseline"** trong `report/README.md`.
- **Bàn giao cho V6**: Bảng kết quả benchmark làm mốc so sánh (Baseline) cho mô hình học sâu chuyên dụng (EEGNet, DeepConvNet).

---

## 5. Nhật ký công việc Tuần 1 (3–9/11): Vai trò V2

- **Người phụ trách**: **Thái Nguyễn Hoàng Bách**
- **Nhiệm vụ theo kế hoạch**: *Chạy tiền xử lý (lọc, ICA, epoch −200 đến +1000 ms) cho toàn bộ người tham gia, xuất dữ liệu sạch trước cuối tuần.*
- **Chi tiết thực hiện**:
  1. Viết pipeline `notebooks/01_preprocessing.ipynb` và chạy cho cả 42 người:
     - tải dữ liệu;
     - cắt đoạn làm nhiệm vụ;
     - dò và nội suy kênh hỏng;
     - Picard ICA + ICLabel;
     - lọc 0,5–40 Hz;
     - epoch −200..+1000 ms ở 250 Hz;
     - gắn metadata hành vi (`condition`, `rt_ms`, `correct`).
  2. Kiểm tra lại toàn bộ đầu ra:
     - Cả 42 file cùng định dạng: 128 kênh × 301 mẫu, cùng thứ tự kênh.
     - Số liệu hành vi khớp Bảng 5 của paper dataset (thời gian phản ứng 542,1 ± 65,5 ms).
     - Tìm ra 3 người (sub-13, 14, 36) có thành phần chớp mắt bị ICLabel chấm dưới ngưỡng, loại bằng tay rồi chạy lại.
  3. Xuất `qc/`, `qc_summary.csv` và `figures/`. Đưa 42 file epoch (4 file zip) lên GitHub Release `data-v1`.
  4. Thử chạy phần V3/V4/V5 trên sub-12 để chắc dữ liệu dùng được ngay.
- **Bàn giao cho V3–V6**: 42 file epoch sạch (`derivatives/v1/epochs/`, tải từ Release `data-v1`) kèm metadata hành vi, `qc_summary.csv` và hướng dẫn đọc dữ liệu ở mục 5 của notebook.

---

## 6. Nhật ký công việc Tuần 2 (10–16/11): Vai trò V2

- **Người phụ trách**: **Thái Nguyễn Hoàng Bách**
- **Nhiệm vụ theo kế hoạch**: *Hỗ trợ sửa lỗi dữ liệu, rồi viết phần Methods (tiền xử lý).*
- **Chi tiết thực hiện**:
  1. **Viết mục 7 "Dữ liệu và Tiền xử lý"** trong `report/README.md`, gồm:
     - quy trình 7 bước;
     - bảng QC;
     - kiểm chứng hành vi và P300;
     - danh sách người cần lưu ý;
     - khác biệt so với paper;
     - hạn chế.
  2. **Kiểm tra dữ liệu của V4**: `all_subjects_features.csv` (Release `feature`) khớp từng dòng với epoch gốc của 39 người. Kiểm tra số trial, nhãn và RT; không có NaN hoặc giá trị vô hạn.
  3. **Phát hiện lỗi bộ nhớ trong `src/utils/io.py` và báo lại nhóm**:
     - `load_dataset` nạp dữ liệu ở float64, nên 39 người tốn khoảng 6,7 GB RAM. Khi V6 nhân thêm `1e6`, RAM đỉnh lên khoảng 13–17 GB.
     - Đề xuất ép về float32 khi đọc: giá trị không đổi vì file đã lưu ở float32, còn RAM giảm còn khoảng 3,4 GB. Đã chạy thử trên 3 người, sai khác bằng 0.
  4. Sửa đường dẫn trong notebook tiền xử lý để chạy được từ thư mục `notebooks/`.


