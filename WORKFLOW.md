# Quy trình làm việc (Workflow)

## 1. Nguyên tắc chung

1. **Hiểu trước, code sau.** Với mỗi thuật toán: nắm bài toán → công thức → viết các bước bằng lời/pseudocode → mới cài đặt.
2. **Kiểm tra từng bước.** Mỗi mô-đun phải có kiểm tra nhỏ trước khi ghép vào pipeline (ví dụ: xác suất cộng lại bằng 1, loss giảm dần qua các vòng lặp).
3. **Đối chiếu với scikit-learn.** Kết quả của bản tự cài đặt phải gần với bản scikit-learn tương ứng. Nếu lệch nhiều, coi như có lỗi cần tìm.
4. **Tập test để dành đến cuối.** Chọn siêu tham số chỉ dựa trên validation. Chỉ báo cáo test một lần cho cấu hình cuối cùng.
5. **Tái lập được.** Cố định seed, ghi lại siêu tham số và kết quả mỗi lần chạy vào `results/`.

## 2. Các giai đoạn

### Giai đoạn 0: Chuẩn bị (tuần 1)
- [ ] Tạo repo GitHub, mời bạn cùng nhóm, thống nhất quy ước nhánh.
- [ ] Đọc kỹ đề, xác nhận với giảng viên: MLP được dùng PyTorch/TensorFlow đến mức nào, và phần nào của đồ án nộp vào giữa kỳ.
- [ ] Chốt biến thể: Naïve Bayes (Gaussian hay Multinomial), linear (Logistic hay SVM), ensemble (RF, AdaBoost hay GB).
- [ ] Chia việc cho từng người.

### Giai đoạn 1: Nền tảng dùng chung (tuần 1)
- [ ] Tải MNIST, chuẩn hoá, làm phẳng, chia train/val/test, cố định seed.
- [ ] Cài các chỉ số: accuracy, precision, recall, F1 macro, confusion matrix (tính từ nhãn thật và nhãn dự đoán).
- [ ] Hàm đo thời gian huấn luyện và dự đoán, hàm vẽ biểu đồ.
- [ ] Thống nhất giao diện chung cho mọi mô hình (cùng kiểu phương thức huấn luyện và dự đoán) để thí nghiệm và so sánh viết một lần dùng cho tất cả.

> Đây là phần cả nhóm phụ thuộc vào, nên làm xong và gộp vào `main` trước khi tách ra làm riêng.

### Giai đoạn 2: Cài đặt từng bộ phân lớp (tuần 2 trở đi)
Mỗi bộ phân lớp đi qua cùng một vòng:

1. Lý thuyết và công thức, viết ghi chú ngắn.
2. Viết các bước thuật toán bằng pseudocode.
3. Cài đặt trên tập con nhỏ (ví dụ vài nghìn ảnh) để chạy nhanh và dễ gỡ lỗi.
4. Đối chiếu với scikit-learn.
5. Chạy trên toàn bộ dữ liệu, ghi lại chỉ số và thời gian.

Thứ tự đề xuất (từ dễ đến khó):

| Thứ tự | Mô hình | Điểm cần lưu ý |
|--------|---------|----------------|
| 1 | Naïve Bayes | Dùng log xác suất để tránh underflow. Pixel có phương sai bằng 0 cần smoothing. |
| 2 | Linear (softmax/Logistic) | Công thức gradient, chọn learning rate, theo dõi đường cong loss. |
| 3 | MLP | Forward, backward, khởi tạo trọng số, kiểm tra gradient bằng số (gradient checking). |
| 4 | Ensemble | Cây quyết định là phần khó nhất; cần chọn tiêu chí chia (Gini/entropy) và giới hạn độ sâu để chạy được trong thời gian hợp lý. |

### Giai đoạn 3: Thí nghiệm siêu tham số
- [ ] Chọn ít nhất một thí nghiệm có biểu đồ (ví dụ: learning rate và số vòng lặp cho linear; số nơ-ron/số lớp cho MLP; số cây cho ensemble).
- [ ] Mỗi lần chỉ thay đổi một siêu tham số, các yếu tố khác giữ nguyên.
- [ ] Vẽ biểu đồ (accuracy trên train và val theo siêu tham số) và rút ra nhận xét: có overfit/underfit không, vì sao.

### Giai đoạn 4: Đánh giá cuối và so sánh
- [ ] Chốt cấu hình tốt nhất của từng mô hình dựa trên validation.
- [ ] Chạy một lần trên test, điền bảng kết quả trong README.
- [ ] Phân tích confusion matrix: các cặp chữ số hay bị nhầm (ví dụ 4–9, 3–5, 7–1) và giải thích.
- [ ] So sánh 4 phương pháp: độ chính xác, thời gian huấn luyện, thời gian dự đoán, khả năng giải thích; nêu ưu và nhược điểm.

### Giai đoạn 5: Báo cáo
Dàn ý gợi ý:
1. Giới thiệu bài toán và dữ liệu
2. Tiền xử lý
3. Lý thuyết và cài đặt từng mô hình
4. Thiết lập thí nghiệm và chỉ số đánh giá
5. Kết quả và biểu đồ
6. So sánh, thảo luận
7. Kết luận, hạn chế
8. Phân công đóng góp của từng thành viên

## 3. Mốc thời gian đề xuất

Hạn nộp giữa kỳ và cuối kỳ do nhóm điền theo đề. Gợi ý chia mốc:

| Mốc | Nội dung |
|-----|----------|
| Giữa kỳ - sớm | Xong giai đoạn 0, 1 và Naïve Bayes |
| Giữa kỳ - giữa | Xong linear, có đường cong loss và thí nghiệm siêu tham số đầu tiên |
| Giữa kỳ - hạn | Có báo cáo giữa kỳ, chốt kết quả trên validation và test cho các mô hình đã làm |
| Cuối kỳ - đầu | MLP chạy đúng |
| Cuối kỳ - giữa | Ensemble chạy đúng, hoàn thành thí nghiệm |
| Cuối kỳ - hạn | So sánh 4 phương pháp, báo cáo cuối, dọn repo |

Luôn chừa **ít nhất 2-3 ngày** trước mỗi hạn để rà soát và sửa lỗi phút cuối.

## 4. Quy trình Git cho nhóm

### Nhánh
- `main`: luôn chạy được, chỉ gộp qua Pull Request.
- Nhánh làm việc đặt tên theo `loại/tên-việc`, ví dụ `feat/naive-bayes`, `feat/metrics`, `exp/linear-lr`, `docs/report`.

### Vòng làm việc hằng ngày
1. `git checkout main` rồi `git pull` để lấy bản mới nhất.
2. `git checkout -b feat/ten-viec` để tạo nhánh.
3. Làm việc, commit nhỏ và thường xuyên.
4. `git push -u origin feat/ten-viec` để đẩy nhánh lên.
5. Tạo Pull Request trên GitHub, nhờ người còn lại xem.
6. Sau khi gộp, xoá nhánh và `git pull` về `main`.

### Quy ước commit
Message ngắn gọn, bắt đầu bằng động từ, ví dụ:
- `Add accuracy and macro F1 metrics`
- `Fix log-probability underflow in Naive Bayes`
- `Add learning-rate experiment for linear model`

### Tránh xung đột
- Chia việc theo file/mô-đun, mỗi người một mô hình.
- Không cùng sửa một file nếu không cần; nếu phải sửa file chung (ví dụ module metrics), báo cho nhau trước.
- Jupyter Notebook: mỗi người một notebook riêng, logic chính đặt trong `src/`.
- Không commit dữ liệu lớn, `.venv`, file kết quả nặng.

### Pull Request
Trước khi xin gộp, tự kiểm tra:
- [ ] Chạy được từ đầu đến cuối, không lỗi
- [ ] Đã đối chiếu với scikit-learn (nếu là mô hình mới)
- [ ] Có ghi chú ngắn về cách dùng
- [ ] Không có file thừa

Người xem PR đọc để hiểu code, không chỉ để duyệt cho qua. Đây cũng là cách tốt để cả hai cùng nắm được toàn bộ đồ án.

## 5. Quy trình chạy thí nghiệm

1. Đặt seed.
2. Huấn luyện trên tập train, chọn siêu tham số bằng val.
3. Ghi lại: siêu tham số, chỉ số trên train/val, thời gian huấn luyện, thời gian dự đoán.
4. Lưu kết quả vào `results/` (một dòng cho mỗi lần chạy).
5. Chỉ chạy test sau khi đã chốt cấu hình.

## 6. Khi gặp lỗi: cách gỡ lỗi

- Mô hình cho accuracy xấp xỉ 10% → gần như đoán ngẫu nhiên, kiểm tra nhãn, chuẩn hoá, và chiều của dữ liệu.
- Loss không giảm hoặc thành `nan` → learning rate quá lớn, hoặc lỗi trong gradient/log(0).
- Accuracy train rất cao mà val thấp → overfit.
- Accuracy cả train và val đều thấp → underfit hoặc lỗi cài đặt.
- Khác scikit-learn nhiều → thử trên tập rất nhỏ rồi so từng bước trung gian.
