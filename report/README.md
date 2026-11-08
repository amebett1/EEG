# Báo cáo Nghiên cứu: Phân tích & Phân loại Sóng P300 trên Tín hiệu EEG

Dự án Oddball Paradigm (GigaDB Dataset 100990) – Nhóm Nghiên cứu EEG.

---

## 1. Tổng quan & Mục tiêu Nghiên cứu

Nghiên cứu sóng não P300 (Event-Related Potential - ERP) là một trong những chủ đề kinh điển và quan trọng nhất trong lĩnh vực Khoa học Thần kinh Nhận thức (Cognitive Neuroscience) và Giao tiếp Não - Máy tính (Brain-Computer Interface - BCI). 

Dự án này tập trung vào:
1. **Phân tích hình thái & điện sinh lý**: Đánh giá sự phân bố không gian và động học thời gian của các phân nhánh sóng P300 (**P3a** và **P3b**) dựa trên nền tảng lý thuyết của **John Polich (2007)**.
2. **Quy trình tiền xử lý chuẩn mực**: Triển khai pipeline chuẩn trên 42 đối tượng với 128 kênh điện cực (Colin27 10-05 montage), kết hợp lọc thông dải, nội suy kênh hỏng, và khử nhiễu tự động bằng Picard ICA + ICLabel.
3. **Mô hình hóa dự đoán**: Trích xuất đặc trưng miền thời gian - tần số và xây dựng các mô hình phân loại học máy (SVM, LDA, RF) cùng mô hình học sâu chuyên biệt (EEGNet).

---

## 2. Cơ sở Lý thuyết: Mô hình Tích hợp P300 của Polich (2007)

Theo nghiên cứu bước ngoặt *"Updating P300: An integrative theory of P3a and P3b"* (Polich, 2007):

```
                        KÍCH THÍCH (Stimulus)
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
KÍCH THÍCH XAO NHÃNG (Distractor)                 KÍCH THÍCH ĐÍCH (Target)
- Không có nhiệm vụ, gây bất ngờ                  - Có nhiệm vụ, cần chú ý & phản hồi
- Kích hoạt chú ý không chủ ý                     - Cập nhật biểu tượng bộ nhớ làm việc
         │                                                 │
         ▼                                                 ▼
        P3a                                               P3b
- Cực đại vùng Trán/Trung tâm (Fz, Cz > Pz)       - Cực đại vùng Đỉnh/Chẩm-đỉnh (Pz > Cz > Fz)
- Độ trễ sớm (~300 - 450 ms)                      - Độ trễ dài hơn (~300 - 600 ms)
- Suy giảm nhanh theo thời gian (Habituation)     - Biên độ tỷ lệ nghịch với xác suất kích thích
```

### Phương pháp Sóng Hiệu số (Difference Waves)
Để bộc lộ chính xác thành phần nội sinh P300 mà không bị chồng lấn bởi các sóng cảm giác sớm (P1 ~100 ms, N1 ~150 ms, P2 ~200 ms), ta thực hiện phép trừ đại số giữa các điều kiện:
- $\Delta_{\text{Target}} = \text{Target} - \text{Standard}$: Cô lập hoàn toàn thành phần **P3b**.
- $\Delta_{\text{Distractor}} = \text{Distractor} - \text{Standard}$: Cô lập thành phần **P3a**.

---

## 3. Hệ thống Tiêu chuẩn Kiểm chứng P300 (Verification Rules)

Áp dụng cho từng đối tượng trong cửa sổ phân tích $[300, 600]\text{ ms}$:
1. **Parietal Gradient (Gradient vùng đỉnh)**:
   $$\text{Amplitude}(Pz) > \text{Amplitude}(Fz)$$
   Đối với Target, điện thế tại Pz phải vượt trội so với Fz, khẳng định đặc tính P3b.
2. **Oddball Effect (Hiệu ứng chênh lệch kích thích)**:
   $$\text{Amplitude}(Pz, \text{Target}) - \text{Amplitude}(Pz, \text{Standard}) \ge 1.5\,\mu\text{V}$$
   Khẳng định có sự phân biệt nhận thức giữa kích thích đích và kích thích nền.
3. **Physiological Latency Window (Cửa sổ độ trễ sinh lý)**:
   $$300\text{ ms} \le \text{Peak Latency}(Pz) \le 600\text{ ms}$$
   Đỉnh cực đại của P3b phải xuất hiện trong khoảng thời gian kích hoạt nhận thức chuẩn.
4. **Fronto-Central Tendency của P3a (Xu hướng vùng trán của Distractor)**:
   Phân bố năng lượng của Distractor dịch chuyển về phía trước (Fz/Cz) so với Target.

---

## 4. Danh mục Trực quan hóa (Vai trò V3)

Mỗi đối tượng được xuất 4 biểu đồ phân tích chuyên sâu tại `results/figures/`:
1. `{sub}_midline_erp.png`: Dạng sóng ERP so sánh 3 điều kiện và dải sai số chuẩn (SEM) tại Fz, Cz, Pz.
2. `{sub}_difference_waves.png`: So sánh trực diện hai sóng hiệu số P3a và P3b.
3. `{sub}_scalp_topomap.png`: Tiến trình không gian 128 kênh từ 150 ms đến 550 ms.
4. `{sub}_p300_summary_topomap.png`: Bản đồ trung bình cửa sổ P300 nhận diện tiêu điểm dương tính đỉnh (Parietal Hotspot).

