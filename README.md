# Nhận dạng chữ số viết tay MNIST bằng các bộ phân lớp tự cài đặt

Đồ án môn **Nhận dạng mẫu (Pattern Recognition)**.

## 1. Giới thiệu

MNIST gồm 70.000 ảnh xám 28×28 của chữ số viết tay từ 0 đến 9 (60.000 ảnh huấn luyện, 10.000 ảnh kiểm thử). Đồ án tự cài đặt các bộ phân lớp đã học, huấn luyện trên MNIST, rồi so sánh và phân tích kết quả.

**Quy ước "from scratch":** phần tính toán lõi của mỗi thuật toán được viết bằng NumPy. `scikit-learn` chỉ dùng để đối chiếu kết quả. Riêng MLP được phép dùng PyTorch/TensorFlow cho phần khung.

## 2. Các bộ phân lớp

| # | Bộ phân lớp | Biến thể chọn | Cách huấn luyện |
|---|-------------|---------------|-----------------|
| 1 | Naïve Bayes | Gaussian / Multinomial (chọn 1) | Ước lượng tham số theo MLE |
| 2 | Bộ phân lớp tuyến tính | Logistic Regression (softmax) / SVM tuyến tính (chọn 1) | Gradient Descent |
| 3 | Mạng nơ-ron nhiều lớp (MLP) | _điền cấu trúc_ | Lan truyền ngược (backpropagation) |
| 4 | Ensemble | Random Forest / AdaBoost / Gradient Boosting (chọn 1) | _điền_ |

> Điền lựa chọn cuối cùng của nhóm vào bảng trên.

## 3. Đánh giá

- Chỉ số: accuracy, precision, recall, F1 (macro), confusion matrix.
- Tính trên cả **tập huấn luyện, validation và kiểm thử**.
- Thí nghiệm siêu tham số (learning rate, số lớp/nơ-ron, số cây, số vòng lặp, ...) có biểu đồ.
- So sánh 4 phương pháp về: độ chính xác, thời gian huấn luyện, thời gian dự đoán, khả năng giải thích.

## 4. Cấu trúc thư mục

```
.
├── README.md
├── WORKFLOW.md
├── requirements.txt
├── .gitignore
├── data/            # dữ liệu MNIST (không đưa lên git)
├── src/
│   ├── data/        # tải, chuẩn hoá, làm phẳng, chia tập
│   ├── models/      # mỗi bộ phân lớp một file
│   ├── metrics/     # accuracy, precision, recall, F1, confusion matrix
│   └── utils/       # đo thời gian, vẽ biểu đồ, đặt seed
├── notebooks/       # thí nghiệm, trực quan hoá (mỗi người một notebook)
├── results/         # bảng số liệu, biểu đồ, log
└── report/          # báo cáo giữa kỳ, cuối kỳ
```

## 5. Cài đặt

Yêu cầu: Python 3.10+.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Thư viện dự kiến: `numpy`, `matplotlib`, `pandas`, `scikit-learn` (chỉ để đối chiếu), `torch` hoặc `tensorflow` (chỉ cho MLP), `jupyter`.

## 6. Dữ liệu

- Tải MNIST về thư mục `data/` (cách tải ghi lại ở đây khi nhóm chốt).
- Chia tập: _ảnh_ huấn luyện / _ảnh_ validation (tách từ 60.000 ảnh huấn luyện) / 10.000 ảnh kiểm thử.
- Tiền xử lý: chia pixel cho 255, làm phẳng 28×28 thành vector 784 chiều.
- Seed cố định: _điền_.

## 7. Cách chạy

_Cập nhật lệnh chạy huấn luyện và đánh giá khi các script đã sẵn sàng._

## 8. Kết quả

_Điền sau khi chạy thí nghiệm._

| Mô hình | Acc (train) | Acc (val) | Acc (test) | F1 macro (test) | Thời gian huấn luyện | Thời gian dự đoán |
|---------|-------------|-----------|------------|-----------------|----------------------|-------------------|
| Naïve Bayes | | | | | | |
| Linear (GD) | | | | | | |
| MLP | | | | | | |
| Ensemble | | | | | | |

## 9. Thành viên

| Họ tên | MSSV | Phụ trách |
|--------|------|-----------|
| | | |
| | | |

## 10. Quy trình làm việc nhóm

Xem [WORKFLOW.md](WORKFLOW.md).