# Phân lớp chữ số MNIST bằng Naïve Bayes và MLP tự cài đặt

## 1. Dữ liệu và đánh giá
- MNIST: 70.000 ảnh 28×28, làm phẳng thành vector 784 chiều. Chia 50.000 train / 10.000 validation / 10.000 test (hoán vị ngẫu nhiên, seed cố định). Chuẩn hóa theo mean/std **tính trên tập train** rồi áp cho val và test để tránh data leakage.
- Chọn siêu tham số bằng K-fold (K = 4) **chỉ trên tập train**. Validation chỉ dùng để báo cáo và so sánh mô hình; test được chạy **một lần** cho cấu hình cuối.
- Chỉ số: accuracy, precision, recall, F1 (macro), confusion matrix, cùng thời gian train và predict.

## 2. Naïve Bayes (Gaussian)

- Tính trong không gian **log** để tránh underflow khi nhân 784 thừa số.
- Pixel gần như luôn bằng 0 có phương sai 0: cộng ε = `var_smoothing` × (phương sai lớn nhất của các đặc trưng) vào mọi phương sai. Đây là siêu tham số của mô hình; giá trị chọn: [..].
- Giả định độc lập có điều kiện và phân phối chuẩn bị vi phạm
- Tham số học được (`mean`, `var`, prior)

## 3. MLP
**Kiến trúc và huấn luyện.** 784 → [hidden_sizes] (ReLU) → 10 logits → softmax; hàm mất mát cross-entropy. Khởi tạo He, bias bằng 0; mini-batch SGD có momentum; dropout dạng inverted ở các lớp ẩn. Siêu tham số chọn: hidden_sizes = [1024, 256], p_drop = [0], learning_rate = [0.08], momentum = [0.08], batch_size = [64].

**Backpropagation tự viết.** truyền ngược qua dropout (nhân mặt nạ) và ReLU (nhân chỉ báo > 0). PyTorch chỉ làm khung: tensor, `nn.Linear` (lưu tham số), `torch.optim.SGD` (bước cập nhật), `F.cross_entropy` (chỉ để theo dõi loss). **Autograd không được dùng để huấn luyện.**

**Early stopping mặc định.** Tách 8% dữ liệu train nội bộ làm tập dừng, theo dõi cross-entropy (không dropout), patience = 5, min_delta = 1e-4, khôi phục trọng số tốt nhất. `epochs` chỉ là giới hạn trên; số epoch thực tế: [..], epoch tốt nhất: [..].

**Lưu ý.**
- Dropout chỉ bật khi huấn luyện; đo loss và dự đoán đều tắt dropout, nên loss train (bật dropout) không so trực tiếp được với loss tập dừng.
- Early stopping dùng tập dừng nội bộ thay vì validation của đồ án, để validation không tham gia vào việc chọn mô hình.
- Mỗi lần `fit` tạo mạng và optimizer mới, nên không có rò rỉ trọng số hay vận tốc giữa các fold.

## 4. Cross-validation và chọn siêu tham số
- K-fold tự cài đặt: hoán vị chỉ số có seed, chia K phần gần bằng nhau; mỗi fold lần lượt làm validation, K−1 phần còn lại làm train; trả về trung bình và độ lệch chuẩn của các chỉ số.
- Mỗi fold dùng **mô hình mới** (qua hàm tạo mô hình) để tránh rò rỉ trạng thái; **cùng seed chia fold** cho mọi cấu hình để so sánh công bằng.
- `find_grid` duyệt mọi tổ hợp trong lưới, chọn cấu hình có accuracy trung bình cao nhất, sau đó mô hình cuối cần được **huấn luyện lại trên toàn bộ tập train**.

## 5. Đánh giá
- Mọi chỉ số được suy ra từ confusion matrix C (hàng = nhãn thật, cột = dự đoán): TP_k = C_kk, FP_k = tổng cột k − TP_k, FN_k = tổng hàng k − TP_k; chỉ số macro là trung bình không trọng số trên 10 lớp; mẫu số bằng 0 được gán 0 [xác nhận đúng với code].
- Lớp `Evaluator` tách riêng huấn luyện và đánh giá, đo thời gian bằng `time.perf_counter`, tính chỉ số trên train/val/test và lưu kết quả ra JSON.

## 6. Kết quả và thảo luận
(Bảng benchmark, confusion matrix và đường cong học ở các ô phía trên.)
- **Độ chính xác:** MLP đạt [..]% trên test so với [..]% của Naïve Bayes (F1 macro lần lượt [..] và [..]).
- **Overfitting:** chênh lệch accuracy train − val là [..] (NB) và [..] (MLP); [nhận xét về tác dụng của dropout và early stopping].
- **Thời gian:** NB huấn luyện trong [..]s (chỉ cần thống kê một lượt qua dữ liệu), MLP trong [..]s ([..] epoch); dự đoán 10.000 mẫu mất [..]s và [..]s.
- **Khả năng giải thích:** NB có tham số trực quan (trung bình và phương sai theo từng pixel của từng lớp, có thể vẽ thành ảnh); MLP là mô hình hộp đen.
- **Lỗi thường gặp:** các cặp nhầm nhiều nhất là [..] (xem confusion matrix) vì [..].
- **Siêu tham số:** [1–2 câu rút ra từ các biểu đồ ảnh hưởng của var_smoothing, learning_rate, p_drop].

## 7. Hạn chế
- Tìm siêu tham số chạy trên tập dữ liệu con; early stopping dùng 8% dữ liệu train; kết quả cuối từ một seed duy nhất.
