Pipeline tiền xử lý nằm trong `notebooks/01_preprocessing.ipynb` (tải data, bad channel, ICA/ICLabel,
lọc 0.5-40 Hz, epoch -200..+1000 ms, QC). Không cần script riêng.
Nếu muốn chạy lại cho một người: mở notebook và gọi `preprocess_subject("sub-XX")`.

Kiểm tra bộ epoch và feature có sẵn từ thư mục gốc dự án:

```bash
python -m src.preprocessing.audit_epochs
```

Script đọc lần lượt 42 người, kiểm tra định dạng, thứ tự kênh, montage, metadata,
nhãn, lọc, average reference, ngưỡng loại epoch, số trial và biên độ QC 300–600 ms.
Nó cũng đối chiếu trial/nhãn/RT và `Pz_p3b_mean` của toàn bộ 39 người với CSV
đặc trưng. Kết quả ghi ở `results/tables/v2_epoch_audit.json`; lỗi trả exit code 1.

Baseline được hiệu chỉnh ở 1000 Hz trước khi decimation xuống 250 Hz. Trung bình
baseline của các mẫu 250 Hz vì vậy không nhất thiết bằng 0; script ghi phần dư
để kiểm tra, không tự hiệu chỉnh lại epoch. Dữ liệu thô và mô hình ICA cần được
tải riêng nếu muốn kiểm chứng lại quá trình loại artifact từ đầu.
