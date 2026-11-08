Pipeline tiền xử lý nằm trong `notebooks/01_preprocessing.ipynb` (tải data, bad channel, ICA/ICLabel,
lọc 0.5-40 Hz, epoch -200..+1000 ms, QC). Không cần script riêng.
Nếu muốn chạy lại cho một người: mở notebook và gọi `preprocess_subject("sub-XX")`.
