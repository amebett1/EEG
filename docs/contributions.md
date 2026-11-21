# Phân công & Ghi nhận Đóng góp (Contributions)

Dự án nghiên cứu và phân loại sóng P300 trên tín hiệu EEG (Oddball Paradigm - 42 đối tượng, GigaDB 100990).
Môn AIT3012 Neuroscience (GS. Guy Nagels, VUB).

---

## Tỷ lệ đóng góp cá nhân (Individual Contribution Shares)

Theo yêu cầu Slide 143 của GS. Guy Nagels, mỗi thành viên được ghi nhận tỷ lệ đóng góp, tổng bằng 100%. Nhóm thống nhất chia đều cho 6 thành viên.

| Mã | Thành viên | Vai trò | Tỷ lệ đóng góp |
|:---:|:---|:---|:---:|
| V1 | Nguyễn Văn Vũ | Quản lý dự án & dữ liệu (Project Manager & Data Management) | 16,7% |
| V2 | Thái Nguyễn Hoàng Bách | Tiền xử lý dữ liệu (Preprocessing) | 16,7% |
| V3 | Trương Huy Hoàng | Trực quan hóa & Phân tích ERP (Visualization / ERP) | 16,7% |
| V4 | Nguyễn Công Kiên | Trích xuất đặc trưng (Feature Extraction) | 16,7% |
| V5 | Nguyễn Việt Đức | Học máy truyền thống (Machine Learning Baseline) | 16,7% |
| V6 | Đồng Minh Đức | Học sâu & Giải thích (Deep Learning & Interpretability) | 16,7% |
| | **Tổng** | | **100%** |

*Ghi chú: mỗi thành viên đóng góp đúng 1/6 (≈ 16,67%); số trong bảng được làm tròn đến một chữ số thập phân nên hiển thị 16,7%.*

---

## 1. Cơ cấu Phân công Vai trò (Team Roles)

Theo sheet "Vai trò" trong `tai_lieu/Ke_hoach_EEG.xlsx`.

| Vai trò | Thành viên | Trách nhiệm chính | File mã nguồn / Tài liệu phụ trách | Trạng thái |
|:---:|:---|:---|:---|:---:|
| **V1** | **Nguyễn Văn Vũ** | **Quản lý dự án + dữ liệu (Project Manager & Data Management)**<br>- Đọc paper dataset (Dzianok et al., 2022), tải dữ liệu GigaDB 100990<br>- Chốt cấu trúc repo và định dạng epoch dùng chung (`config.py`)<br>- Điều phối, theo dõi deadline, chuẩn bị buổi trung gian với GS. Nagels<br>- Ghép báo cáo, kiểm tra tài liệu tham khảo, chốt bảng % đóng góp<br>- **Đầu ra bàn giao**: kế hoạch 3 tuần, tóm tắt dataset, bảng % đóng góp, báo cáo ghép hoàn chỉnh | `tai_lieu/Ke_hoach_EEG.xlsx`<br>`docs/dataset_cheatsheet.md`<br>`docs/contributions.md`<br>`config.py`<br>`report/README.md`, `report/REPORT_EN.md` | **Tuần 1–2 hoàn thành; Tuần 3 đang làm** |
| **V2** | **Thái Nguyễn Hoàng Bách** | **Tiền xử lý dữ liệu (Preprocessing)**<br>- Lọc dải tần 0,5–40 Hz<br>- Phát hiện và nội suy kênh hỏng (IQR, tương quan lân cận)<br>- Picard ICA kết hợp ICLabel loại bỏ artifact (mắt, cơ, tim)<br>- Cắt epoch (−200..+1000 ms), decimation 4 (1000 Hz → 250 Hz), gán nhãn standard/target/distractor<br>- Xuất file QC từng người và `qc_summary.csv`<br>- **Đầu ra bàn giao**: script tiền xử lý và 42 file epoch sạch cho cả nhóm | `notebooks/01_preprocessing.ipynb`<br>`src/preprocessing/`<br>`derivatives/v1/qc/` | **Tuần 1–2 hoàn thành; Tuần 3 đang làm** |
| **V3** | **Trương Huy Hoàng** | **Trực quan hóa & Phân tích ERP (Visualization / ERP)**<br>- Grand average ERP (Fz, Pz), scalp map, kiểm tra P300 ở khoảng 300–500 ms<br>- Code vẽ dạng sóng ERP đường giữa (Fz, Cz, Pz) kèm sai số chuẩn SEM<br>- Code vẽ sóng hiệu số (Target − Standard, Distractor − Standard)<br>- Code vẽ bản đồ địa hình da đầu 128 kênh (Scalp Topomap)<br>- Kiểm tra định lượng tiêu chuẩn Polich trong cửa sổ 300–500 ms<br>- **Đầu ra bàn giao**: hình ERP và topomap chất lượng cao cho báo cáo, slide | `src/erp/plot_erp.py`<br>`notebooks/02_erp_visualization.ipynb`<br>`results/figures/`, `report/figures/`<br>`results/tables/v3_p300_group_results.csv` | **Hoàn thành (Tuần 1, 2, 3)** |
| **V4** | **Nguyễn Công Kiên** | **Trích xuất đặc trưng (Feature Extraction)**<br>- Biên độ đỉnh và độ trễ của P300 và N100/P200/N200 theo Laton et al. (2014)<br>- Biên độ trung bình các cửa sổ thời gian quan trọng<br>- Năng lượng dải sóng (Delta, Theta, Alpha, Beta) và tham số Hjorth<br>- Spectral entropy, wavelet db4<br>- **Đầu ra bàn giao**: ma trận đặc trưng 21.878 trial × 402 đặc trưng (39 người) cho V5 | `src/features/extract_features.py`<br>`src/features/rank_features.py`<br>`derivatives/v1/features/`<br>`results/tables/v4_feature_auc.csv` | **Hoàn thành (Tuần 1, 2, 3)** |
| **V5** | **Nguyễn Việt Đức** | **Học máy truyền thống (Machine Learning Baseline)**<br>- Huấn luyện Dummy, LDA, Logistic Regression, RBF SVM, Random Forest<br>- Chia tập chống rò rỉ dữ liệu (GroupKFold theo người)<br>- Xử lý mất cân bằng lớp (class weights; balanced accuracy, ROC-AUC, PR-AUC)<br>- Benchmark trên 39 đối tượng (21.878 trials, 402 đặc trưng)<br>- **Đầu ra bàn giao**: kết quả baseline, trả lời câu hỏi nghiên cứu 1 | `src/ml/train_baseline.py`<br>`notebooks/03_machine_learning.ipynb`<br>`results/tables/baseline_metrics.csv`<br>`results/figures/baseline_*.png` | **Hoàn thành (Tuần 1, 2, 3)** |
| **V6** | **Đồng Minh Đức** | **Học sâu & Giải thích (Deep Learning & Interpretability)**<br>- Huấn luyện EEGNet-8,2 (PyTorch thuần) trên epoch thô (128 kênh × 301 mẫu)<br>- Đánh giá GroupKFold theo người trên cùng 39 đối tượng như V5<br>- Saliency map theo kênh và thời gian, đối chiếu với P300 (Polich, 2007)<br>- So sánh ML vs DL<br>- **Đầu ra bàn giao**: kết quả DL, trả lời câu hỏi nghiên cứu 2 và 3 | `src/dl/train_eegnet.py`<br>`notebooks/04_deep_learning.ipynb`<br>`results/tables/eegnet_metrics_real.csv`<br>`results/tables/ml_vs_dl_comparison.csv`<br>`results/figures/eegnet_*.png`, `ml_vs_dl_comparison.png` | **Hoàn thành (Tuần 1, 2, 3)** |

---

## 2. Nhật ký công việc: Vai trò V1

- **Người phụ trách**: **Nguyễn Văn Vũ**

### Tuần 1 (3–9/11)
- **Nhiệm vụ theo kế hoạch**: *Đọc paper dataset, tải data, chốt repo và định dạng epoch, lập bảng đóng góp.*
- **Chi tiết thực hiện**:
  1. Đọc paper dataset Nencki-Symfonia (Dzianok et al., 2022); ghi nguồn GigaDB 100990 vào `docs/dataset_cheatsheet.md` và chuyển các thông tin cần cho nhóm (mã sự kiện S5/S6/S7 = standard/target/distractor, danh sách người cần lưu ý theo paper) vào `config.py`.
  2. Tải dữ liệu thô và thống nhất cấu trúc repo dùng chung cho 6 vai trò.
  3. Chốt định dạng epoch dùng chung trong `config.py`: −200..+1000 ms, 250 Hz (301 mẫu), 128 kênh cùng thứ tự, `EVENT_ID`, danh sách người hiểu sai nhiệm vụ (sub-02, sub-14, sub-17).
  4. Lập kế hoạch 3 tuần và bảng phân công/đóng góp trong `tai_lieu/Ke_hoach_EEG.xlsx`.

### Tuần 2 (10–16/11)
- **Nhiệm vụ theo kế hoạch**: *Họp ngắn giữa tuần, chuẩn bị buổi trung gian với GS. Nagels, xác nhận ngày với thầy Sáng.*
- **Chi tiết thực hiện**:
  1. Tổ chức họp giữa tuần, theo dõi tiến độ của V2–V6 trong sheet "Kế hoạch 3 tuần".
  2. Chuẩn bị nội dung buổi trung gian với GS. Nagels: ERP nhóm (V3) và kết quả baseline sơ bộ (V5).
  3. Xác nhận lịch thuyết trình với thầy Sáng.

### Tuần 3 (17–23/11)
- **Nhiệm vụ theo kế hoạch**: *Ghép báo cáo bản nháp, kiểm tra tài liệu tham khảo, chốt % đóng góp.*
- **Chi tiết thực hiện**:
  1. Chốt bảng tỷ lệ đóng góp (đầu file này) và chuẩn hóa lại bảng vai trò V1–V6.
  2. Ghép báo cáo tiếng Việt `report/README.md` và bản tiếng Anh chính thức `report/REPORT_EN.md`.
  3. Kiểm tra tài liệu tham khảo của các mục.
  4. Lập khung thuyết trình 10 phút `docs/presentation_slides.md` và phân công người nói.
- **Hạn chót**: nộp báo cáo tiếng Anh ít nhất 1 tuần trước buổi thuyết trình (~1/12).

---

## 3. Nhật ký công việc: Vai trò V2

- **Người phụ trách**: **Thái Nguyễn Hoàng Bách**

### Tuần 1 (3–9/11)
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

### Tuần 2 (10–16/11)
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

### Tuần 3 (17–23/11)
- **Nhiệm vụ theo kế hoạch**: *Đọc soát phần Methods, hỗ trợ V1 ghép bài.*
- **Chi tiết thực hiện**:
  1. **Đọc soát mục 3 "Dataset and preprocessing" của `report/REPORT_EN.md`** so với mục 7 bản tiếng Việt và với QC: bản dịch đúng, mọi con số khớp. Đề xuất V1 sửa câu trong Abstract và câu về ICA trong mục 11.1 (dựa trên kết quả ở ý 2).
  2. **Kiểm tra quyết định ICA từ dữ liệu thô** (phần script audit chưa làm):
     - 42 mô hình ICA đều có 40 thành phần, số thành phần bị loại khớp QC.
     - Với sub-13, 14, 36: dựng lại dữ liệu đầu vào ICA từ dữ liệu thô và chạy lại ICLabel, ra đúng xác suất chớp mắt 0,78 / 0,70 / 0,59 như đã báo cáo. Trọng số lớn nhất ở vùng trán cực.
     - Đưa 42 mô hình ICA lên Release `data-v1` (`ica_v1_42subjects.zip`) để nhóm và người chấm kiểm tra lại. Cập nhật mục 7.6 và 7.7 trong `report/README.md`.
  3. **Vẽ sơ đồ pipeline 7 bước cho slide 2**: `report/figures/v2_preprocessing_pipeline.png`, đồng thời chèn vào mục 7.2 của báo cáo.

---

### Rà soát bàn giao V2/V3

- V2: `src/preprocessing/audit_epochs.py` đọc lần lượt 42 file (27.034 epoch), đối chiếu metadata, kênh, lọc, baseline metadata, average reference, ngưỡng loại epoch và QC; đối chiếu trial/nhãn/RT và `Pz_p3b_mean` với CSV của 39 người. Tất cả kiểm tra đã triển khai đạt; chi tiết `results/tables/v2_epoch_audit.json`.
- Không chạy lại tiền xử lý từ dữ liệu thô hoặc xác minh lại các quyết định ICA vì dữ liệu và mô hình ICA tương ứng chưa có trong bản làm việc.
- V3: sửa kiểm tra latency vòng tròn; thống nhất bảng/ERP trên trial đúng và 300–500 ms; giữ QC 300–600 ms trên toàn bộ trial riêng biệt. Không chọn lại người cho ML/DL theo kiểm tra ERP.
- Xuất lại bảng 42 người, thống kê tái lập và 5 hình nhóm với trọng số bằng nhau cho 39 người; đồng bộ báo cáo Việt/Anh và nội dung V2/V3 trong khung slide. Có kiểm thử hồi quy bằng `python -m unittest discover -s tests -v`.

## 4. Nhật ký công việc: Vai trò V3

- **Người phụ trách**: **Trương Huy Hoàng**

### Tuần 1 (3–9/11)
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
     - `verify_p300`: ba tiêu chí lõi P3b và xu hướng P3a mô tả, do dự án định nghĩa; tìm đỉnh 200–800 ms độc lập với cửa sổ chấp nhận 300–500 ms.
  3. **Tạo Notebook tương tác `notebooks/02_erp_visualization.ipynb`**:
     - Cho phép chạy trực quan từng phần, lưu toàn bộ hình ảnh độ phân giải cao vào `results/figures/` và `report/figures/`.
  4. **Thử nghiệm & Bàn giao**:
     - Hoàn thiện thử nghiệm và kiểm chứng định lượng trên đối tượng đầu tiên (`sub-01`).
     - Chuẩn bị đầy đủ bộ hình ERP và topomap sẵn sàng chèn vào báo cáo và slide thuyết trình.

### Tuần 2 (10–16/11)
- **Nhiệm vụ theo kế hoạch**: *ERP Fz/Pz và scalp map toàn bộ nhóm, kiểm tra P300, viết phần Results (ERP).*
- **Chi tiết thực hiện**:
  1. **Grand Average ERP (Fz, Pz) toàn bộ nhóm**:
     - Phân tích dạng sóng trên toàn bộ 39 đối tượng chuẩn tại Fz và Pz trong cửa sổ $[300, 500]\text{ ms}$.
     - Số liệu tính lại từ trial đúng, 300–500 ms: Target − Standard tại Pz **5,05 ± 2,28 µV**, **t(38) = 13,82**, **p = 2,14 × 10⁻¹⁶**, **dz = 2,21**. Giá trị này thay cho số liệu trộn cửa sổ trước đây.
     - Target Pz trung bình 300–500 ms **6,97 ± 2,74 µV**, Standard **1,92 ± 1,49 µV**; 39/39 người có Target Pz > Fz.
  2. **Scalp Topomap 128 kênh toàn bộ nhóm**:
     - Trực quan hóa diễn tiến không gian từ chẩm (cảm giác sơ cấp) đến đỉnh (P3b nhận thức) qua các mốc thời gian 150–550 ms.
     - Xác lập bản đồ trung bình cửa sổ P300 [300, 500] ms với tiêu điểm dương tính đỉnh (Parietal Hotspot).
  3. **Kiểm định định lượng Polich (2007) trên 42 người**:
     - Kiểm tra tính lại: gradient 39/39, oddball 37/39, latency 22/39, xu hướng trán Distractor 26/39. Đồng thời ba tiêu chí lõi: 22/39; phân tích bổ sung 300–600 ms: 32/39.
     - Tương quan RT theo QC với Target Pz 300–500 ms trên trial đúng: **r = −0,493, p = 0,00144**, không suy diễn nhân quả.
     - Nhóm sub-02/14/17 bị loại theo hành vi (false alarm Distractor trung bình 90,95%); không khẳng định không có ERP. Target Pz trên trial đúng **5,29 ± 2,58 µV**; chỉ còn 1/1/19 Distractor đúng nên các trung bình này không ổn định.
  4. **Xuất bảng kết quả và soạn thảo báo cáo**:
     - Xuất bảng kết quả kiểm định 42 đối tượng ra file `results/tables/v3_p300_group_results.csv`.
     - Soạn thảo toàn bộ phần 8 "Kết quả Nghiên cứu & Phân tích ERP (Results – ERP)" trong `report/README.md`.

### Tuần 3 (17–23/11)
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

## 5. Nhật ký công việc: Vai trò V4

- **Người phụ trách**: **Nguyễn Công Kiên**

### Tuần 1–2 (3–16/11)
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

## 6. Nhật ký công việc: Vai trò V5

- **Người phụ trách**: **Nguyễn Việt Đức**

### Tuần 1–3 (3–23/11)
- **Nhiệm vụ theo kế hoạch**: *Xây dựng pipeline học máy truyền thống (Baseline) phân loại P300 đơn trial (single-trial), kiểm định Cross-Validation không rò rỉ dữ liệu, đánh giá và so sánh đa mô hình.*
- **Chi tiết thực hiện**:
  1. **Xây dựng module dòng lệnh `src/ml/train_baseline.py` và Notebook tương tác `notebooks/03_machine_learning.ipynb`**:
     - Đồng bộ cả 2 hình thức: script thực thi tự động qua CLI có `argparse` và notebook trực quan hóa từng bước.
  2. **Thiết lập Pipeline chống rò rỉ dữ liệu (No Data Leakage)**:
     - Tích hợp `StandardScaler` bên trong `Pipeline` của `scikit-learn`, đảm bảo chỉ fit trên tập huấn luyện của từng fold.
     - Hỗ trợ cơ chế phân chia linh hoạt: tự động chia `GroupKFold(n_splits=5)` theo `subject_id` khi có nhiều người và fallback sang `StratifiedKFold(n_splits=5)` khi chạy thử nghiệm trên 1 người (`sub-01`).
  3. **Xử lý mất cân bằng lớp nghiêm trọng (Prevalence 11,3% - 13,5%)**:
     - Cấu hình `class_weight="balanced"` cho Logistic Regression, SVM và `balanced_subsample` cho Random Forest.
     - Đánh giá bằng bộ chỉ số đa chiều: Balanced Accuracy, ROC-AUC, PR-AUC, F1-macro, Sensitivity và Specificity (không dựa vào raw accuracy dễ gây ngộ nhận).
  4. **Triển khai và đánh giá 5 mô hình benchmark**:
     - `DummyClassifier` (Stratified chance level).
     - `LinearDiscriminantAnalysis` (bộ giải `lsqr` kèm shrinkage tự động — chuẩn vàng cho BCI).
     - `LogisticRegression` (điều chuẩn L2).
     - `SVC` (RBF kernel, xác suất Platt).
     - `RandomForestClassifier` (100 cây quyết định, `n_jobs=-1`).
  5. **Xuất bản kết quả thực nghiệm và biểu đồ khoa học**:
     - Bảng số liệu chi tiết: `results/tables/baseline_metrics.csv`.
     - Bộ 3 biểu đồ tiếng Anh Sentence case (300 DPI): `baseline_model_comparison.png`, `baseline_roc_curves.png`, `baseline_pr_curves.png` tại `results/figures/` và `report/figures/`.
  6. **Viết phần 6 "Mô hình Học máy Truyền thống Baseline"** trong `report/README.md`, trả lời câu hỏi nghiên cứu 1.
- **Bàn giao cho V6**: Bảng kết quả benchmark làm mốc so sánh (Baseline) cho mô hình học sâu.

---

## 7. Nhật ký công việc: Vai trò V6

- **Người phụ trách**: **Đồng Minh Đức**

### Tuần 1 (3–9/11)
- **Nhiệm vụ theo kế hoạch**: *Dựng EEGNet, chạy thử trên dữ liệu thử.*
- **Chi tiết thực hiện**: Dựng EEGNet-8,2 và bộ sinh dữ liệu mô phỏng cùng định dạng epoch thật (`make_synthetic_dataset`) để kiểm tra pipeline trước khi có dữ liệu sạch (`results/tables/eegnet_metrics_synthetic.csv`).

### Tuần 2–3 (10–23/11)
- **Nhiệm vụ theo kế hoạch**:
  - Tuần 2: *Huấn luyện DL đầy đủ, bắt đầu saliency.*
  - Tuần 3: *Hoàn thiện saliency, so sánh với baseline, viết phần DL/Explainability, trả lời câu hỏi 2 và 3.*
- **Chi tiết thực hiện**:
  1. **Xây dựng kiến trúc EEGNet-8,2 (PyTorch thuần)**:
     - Triển khai mạng tích chập chuyên biệt cho tín hiệu điện não theo Lawhern et al. (2018) trong `src/dl/train_eegnet.py` và `notebooks/04_deep_learning.ipynb`.
     - Block 1: Tích chập thời gian (8 bộ lọc, kernel 125 mẫu) kết hợp Depthwise Spatial Conv (16 bộ lọc không gian trên 128 kênh) và Average Pooling.
     - Block 2: Separable Conv kết hợp Pointwise Conv và Average Pooling.
     - Ràng buộc chuẩn trọng số Max-Norm (1,0 cho spatial conv, 0,25 cho dense classifier) và Dropout (p = 0,5) nhằm hạn chế quá khớp.
  2. **Thử nghiệm sơ bộ trên 1 người (sub-01, 540 trial, StratifiedKFold)**: kiểm tra pipeline trên epoch thật trước khi chạy toàn bộ. Kết quả này **không** dùng để so sánh với baseline vì khác tập dữ liệu và khác cách chia fold.
  3. **Đánh giá cross-subject trên toàn bộ 39 đối tượng (GPU)** — cùng tập dữ liệu và cùng sơ đồ `GroupKFold(n_splits=5)` theo người như V5:
     - Sửa `train_eegnet` để đánh giá tập validation theo batch (tránh tràn VRAM GPU 8 GB khi chạy đủ 39 người).
     - Đã xác minh số trial của từng người khớp ma trận đặc trưng V4, nên 5 fold test trùng hoàn toàn với baseline của V5.
     - Kết quả (trung bình ± SD qua 5 fold): balanced accuracy **0,907 ± 0,021**, ROC-AUC **0,970 ± 0,010**, PR-AUC **0,873 ± 0,042**, sensitivity **0,884 ± 0,060**, specificity **0,930 ± 0,030** — vượt mọi baseline.
     - So sánh cặp theo từng người (Wilcoxon, 39 người): ROC-AUC của EEGNet cao hơn LDA ở 35/39 người và cao hơn Logistic Regression ở 34/39 người (p < 10⁻⁸) → `results/tables/ml_vs_dl_per_subject.csv`.
     - Lưu bảng `eegnet_metrics_real.csv`, `eegnet_folds_real.csv`, xác suất out-of-fold `eegnet_oof_real.npz` và mô hình `results/models/eegnet_real.pt`; khoảng 8 phút/fold trên RTX 4060.
  4. **Phân tích khả năng giải thích bằng bản đồ độ nhạy (Saliency Map)**: gradient của logit Target theo tensor đầu vào, trung bình trên các trial Target.
     - Thời gian: đỉnh ở 504 ms; 40,2% saliency trong 300–600 ms (cửa sổ chỉ chiếm 24,9% epoch).
     - Không gian: điểm nóng ở vùng trung tâm trái (C1, FCC3h, CCP3h, C3), không phải Pz (xếp hạng 37/128); tương quan với hiệu ứng P300 theo kênh r = 0,32. ERP trung bình không lệch trái (C3 vs C4, p = 0,50) → cần kiểm soát thêm giả thuyết tín hiệu vận động do bấm nút.
     - Xuất `eegnet_saliency_real.png` và topomap `eegnet_saliency_topomap_real.png` (tại `results/figures/` và `report/figures/`).
  5. **So sánh ML vs DL và trả lời câu hỏi 2, 3**: `results/tables/ml_vs_dl_comparison.csv`, `results/figures/ml_vs_dl_comparison.png`; soạn mục 10 và 11 trong `report/README.md`.
