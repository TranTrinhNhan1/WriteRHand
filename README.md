# WriteRHand - nhận dạng chữ số viết tay MNIST bằng các bộ phân lớp tự cài đặt

Đồ án 1 (giữa kỳ) môn **Nhận dạng**, Khoa Toán – Tin học, ĐH KHTN, ĐHQG-HCM. Bốn bộ phân lớp bắt buộc cùng một CNN nhỏ làm phần mở rộng, phần tính toán lõi viết tay
(NumPy; PyTorch chỉ làm kho tensor cho MLP và CNN, **không dùng autograd, `nn.Module`, `optim`, `F.conv2d`, `F.max_pool2d`**; scikit-learn chỉ để đối chiếu):

| # | Bộ phân lớp | Cách huấn luyện | Mã nguồn |
|---|-------------|-----------------|----------|
| 1 | Gaussian Naïve Bayes | MLE + log-xác suất | `src/models/naive_bayes.py` |
| 2 | Hồi quy logistic đa lớp (softmax) | mini-batch Gradient Descent, L2 | `src/models/logistic_regression.py` |
| 3 | MLP (ReLU, dropout, momentum) | lan truyền ngược viết tay, early stopping | `src/models/mlp.py` |
| 4 | AdaBoost-SAMME (cây Gini có trọng số) | boosting tuần tự | `src/models/adaboost.py` |
| 5 | CNN nhỏ (conv-ReLU-max-pool ×2, dense) | im2col/col2im, backprop và SGD momentum viết tay | `src/models/cnn.py` |

**English summary.** Handwritten digit classification on MNIST with four classifiers (plus a small CNN) implemented from scratch,
a leak-free reproducible protocol (stratified 50k/10k train/validation split, official 10k test), validation-only
hyper-parameter search, scikit-learn parity tests, W&B tracking (offline fallback), ONNX export to Hugging Face,
extensions (Fashion-MNIST, EMNIST Digits, five augmentation transforms with a controlled ablation, CNN vs MLP), an in-browser demo
(React + ONNX Runtime Web) with a Model Execution Explorer driven by the deployed model through a versioned trace contract, a LaTeX report and Beamer slides.

**Demo trực tuyến:** <https://trantrinhnhan1.github.io/WriteRHand/> · **Mô hình:** <https://huggingface.co/TranTrinhNhan1111/WriteRHand-MNIST>

## Kết quả

Đo trên CPU (xem `results/final/*.json` để biết phần cứng và phiên bản thư viện). Siêu tham số chọn **chỉ trên validation**;
test chỉ để báo cáo. Bảng này được sinh tự động từ `results/` bằng `uv run python -m src report`.

<!-- BENCHMARK:START -->
| Model | Acc train | Acc val | Acc test | P macro | R macro | F1 macro | Fit (s) | Predict 10k (s) | 1-sample (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Gaussian NB | 80.13 | 79.67 | 81.32 | 82.98 | 80.95 | 81.16 | 0.4 | 0.226 | 0.076 |
| Softmax regression | 93.62 | 92.15 | 92.52 | 92.45 | 92.41 | 92.42 | 12.7 | 0.024 | 0.010 |
| MLP | 99.87 | 98.18 | 98.24 | 98.23 | 98.23 | 98.23 | 11.0 | 0.051 | 0.042 |
| AdaBoost-SAMME | 100.00 | 97.59 | 97.60 | 97.58 | 97.58 | 97.58 | 817.9 | 0.751 | 18.705 |
| CNN | 99.40 | 98.65 | 98.80 | 98.79 | 98.79 | 98.79 | 143.0 | 1.163 | 0.315 |
<!-- BENCHMARK:END -->

Chi tiết: báo cáo `report/main.pdf`, slide `slides/WriteRHand-slides.pdf`, notebook `notebooks/`, nhật ký thí nghiệm `results/`.

## Cài đặt

Yêu cầu: [uv](https://docs.astral.sh/uv/) và Python ≥ 3.11 (phát triển/CI trên 3.12). PyTorch được cài bản CPU.

```bash
uv sync --extra export --extra track --group dev   # môi trường tái lập từ uv.lock
uv run pytest                                      # kiểm thử (chạy offline, ~30 s)
uv run ruff check src tests && uv run ruff format --check src tests
```

## Tái lập toàn bộ thí nghiệm (CLI)

```bash
uv run python -m src prepare                  # tải MNIST, chia phân tầng 50k/10k, ghi thống kê chuẩn hóa
uv run python -m src tune --model all         # quét siêu tham số (chỉ train/val)  -> results/tuning/
uv run python -m src train --model all        # huấn luyện cấu hình đã chọn, đánh giá train/val/test -> results/final/
uv run python -m src baselines --model all    # đối chứng scikit-learn cùng giao thức -> results/baselines/
uv run python -m src augment --model mlp --seeds 3   # đối chứng tăng cường dữ liệu (mlp | cnn) -> results/augmentation_<model>.json
uv run python -m src compare                  # CNN so với MLP (McNemar, độ bền) -> results/cnn_vs_mlp.json
uv run python -m src replay --model mlp       # ghi lại điểm kiểm tra huấn luyện thật cho trình khám phá -> artifacts/replay/
uv run python -m src report                   # bảng + hình + đoạn LaTeX và cập nhật README
uv run python -m src export --site            # ONNX/nhị phân + hợp đồng vết + kiểm chứng số học -> artifacts/bundle, web/public/data
uv run python -m src upload --write-lock      # đẩy lên Hugging Face và ghim revision vào web/model-lock.json
```

Bộ dữ liệu khác (mỗi bộ có phép chia, siêu tham số, kết quả và mô hình riêng, ghi vào `results/datasets/<tên>/`):

```bash
for ds in fashion_mnist emnist_digits; do
  uv run python -m src prepare --dataset $ds
  uv run python -m src tune --model all --light --dataset $ds      # quét rút gọn
  uv run python -m src train --model all --dataset $ds
  uv run python -m src baselines --model all --dataset $ds
  for m in softmax mlp cnn adaboost; do uv run python -m src replay --model $m --dataset $ds; done
  uv run python -m src report --dataset $ds
done
```

`--smoke` chạy bản siêu nhỏ (vài giây) để kiểm tra pipeline; `--results-dir`/`--artifacts-dir` đổi nơi ghi.
Notebook tái lập kết quả chính: `notebooks/reproduce_results.ipynb` (không phụ thuộc trạng thái ẩn).

### Theo dõi thí nghiệm (Weights & Biases)
`--wandb auto` (mặc định) dùng chế độ online nếu có `WANDB_API_KEY` (hoặc đã `wandb login`), ngược lại **offline** (ghi vào `./wandb`,
đồng bộ sau bằng `wandb sync wandb/offline-run-*`); `--wandb disabled` tắt hẳn. Mỗi run ghi tên mô hình, siêu tham số, seed, git commit,
mã băm phép chia dữ liệu, loss/accuracy theo epoch, ma trận nhầm lẫn, thời gian và artifact mô hình. Nhật ký JSON trong `results/` luôn được ghi, không phụ thuộc W&B.

### Mô hình và website
Trọng số **không** được commit. Bundle (ONNX + tham số nhị phân + hợp đồng vết + model card) nằm trên Hugging Face và được **ghim theo mã commit** trong `web/model-lock.json`:
<https://huggingface.co/TranTrinhNhan1111/WriteRHand-MNIST>. Website tĩnh (React + TypeScript + Vite + ONNX Runtime Web) nằm trong `web/`
(xem `web/README.md` và `docs/TRACE_CONTRACT.md`): chọn bộ dữ liệu và một trong năm mô hình, có giao diện sáng/tối, và một **trình khám phá thực thi mô hình** vẽ đúng phép tính của mô hình đã triển khai (sơ đồ dựng từ metadata của hợp đồng vết, giá trị từ ONNX Runtime Web, đối chiếu với tính lại từ tham số) cho MLP, CNN, Naïve Bayes, softmax và AdaBoost, cùng chế độ phát lại huấn luyện chỉ từ các điểm kiểm tra thật. Trực quan hoá tính toán không phải là giải thích quyết định; điều này được ghi rõ trong giao diện và báo cáo. CI chạy kiểm thử Python, kiểm thử Vitest/parity trên bundle đã ghim (thiếu bundle là lỗi, không bỏ qua), kiểm thử trình duyệt Playwright, rồi build và triển khai lên GitHub Pages (`.github/workflows/`).

## Giao thức thực nghiệm (tóm tắt)
- 60 000 ảnh huấn luyện chính thức → **50 000 train / 10 000 validation** (phân tầng, seed 42); 10 000 ảnh test chính thức.
- Chuẩn hóa `(pixel/255 − mean)/(std + 1e-8)` với `mean`, `std` chỉ tính trên train; cùng công thức dùng trong trình duyệt.
- Chọn siêu tham số bằng quét tuần tự từng tham số (coordinate descent do nhóm tự viết, không dùng Optuna), chọn theo validation; mọi cấu hình đã chạy được ghi lại. Hàm tìm kiếm không nhận mảng test.
- Mô hình cuối huấn luyện trên 50 000 ảnh train; báo cáo train/val/test với accuracy, macro precision/recall/F1, confusion matrix.

## Báo cáo và slide
```bash
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex   # cần TeX Live có T5 + lmodern
cd slides/beamer && latexmk -pdf main.tex && cp main.pdf ../WriteRHand-slides.pdf                                             # slide Beamer (Computer Modern)
```
Các con số trong báo cáo và slide được đọc từ `results/` (xem `src/experiments/texfrag.py`, `src/experiments/extras_report.py`), không nhập tay.

Hai PDF được commit và được website phát hành tại `/WriteRHand/docs/report.pdf` và `/WriteRHand/docs/slides.pdf` (trang **Tài liệu**). Để một PDF cũ không bị phát hành âm thầm,
sau khi dựng lại PDF hãy ghi dấu nguồn: `python3 scripts/docs_stamp.py write`; CI và workflow Pages chạy `python3 scripts/docs_stamp.py check` và thất bại nếu nguồn LaTeX, đoạn sinh tự động
hoặc hình đã đổi mà PDF chưa dựng lại.

## Nộp bài và quyền truy cập
Đề bài yêu cầu repository Git **công khai** hoặc mã nén `.zip`. Repository `TranTrinhNhan1/WriteRHand` hiện ở chế độ **riêng tư** (chỉ chủ sở hữu và người được mời xem được mã nguồn);
website, mô hình trên Hugging Face và hai PDF vẫn truy cập công khai. Nếu giữ repository riêng tư, nộp bản `.zip` sạch (không `data/`, `wandb/`, `.venv/`, `node_modules/`, `.git/`):
`git archive --format=zip -o WriteRHand.zip HEAD`. Việc đổi chế độ hiển thị do nhóm quyết định.

## Cấu trúc thư mục
```
src/                 models/ (5 mô hình), preprocessing/ (3 bộ dữ liệu), evaluation/, experiments/ (tune, train, augment, compare, replay, report), export/ (ONNX, hợp đồng vết, Hugging Face), cli.py
tests/               kiểm thử đơn vị, đối chiếu scikit-learn/PyTorch, bundle và hợp đồng vết (chạy offline)
docs/                TRACE_CONTRACT.md: hợp đồng giữa Python và trình khám phá; pdf-stamps.json: dấu nguồn của hai PDF
scripts/             docs_stamp.py: kiểm tra PDF còn khớp nguồn
notebooks/           EDA, reproduce_results (chính), fine-tuning, final_eval_with_test_set, eval_for_ensemble (khám phá)
results/             tuning/ final/ baselines/ tables/ figures/ (sinh ra bởi chương trình)
report/              báo cáo LaTeX (main.tex → main.pdf, các đoạn sinh tự động ở generated/)
slides/              slide Beamer (beamer/main.tex) + PDF
web/                 website demo (src/engine, src/components/explorer, e2e/)      .github/workflows/   CI + GitHub Pages
```

## Thành viên
Trần Trinh Nhân (24280088) và Lâm Nhật Tiến (24280027), đóng góp 50%/50% (nhóm tự thống nhất); chi tiết ở mục 9 của báo cáo. Nhóm có dùng trợ lý AI để **hỗ trợ lập trình và điều chỉnh siêu tham số**, theo công bố trong báo cáo. Quy trình nhóm: [WORKFLOW.md](WORKFLOW.md).

## Tài liệu / nguồn
LeCun et al. (1998) cho MNIST; Zhu et al. (2009) cho SAMME; xem danh mục đầy đủ trong báo cáo.
