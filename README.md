# WriteRHand - nhận dạng chữ số viết tay MNIST bằng các bộ phân lớp tự cài đặt

Đồ án môn **Nhận dạng mẫu (Pattern Recognition)**. Bốn bộ phân lớp, phần tính toán lõi viết bằng NumPy
(PyTorch chỉ làm tensor cho MLP, **không dùng autograd**; scikit-learn chỉ để đối chiếu):

| # | Bộ phân lớp | Cách huấn luyện | Mã nguồn |
|---|-------------|-----------------|----------|
| 1 | Gaussian Naïve Bayes | MLE + log-xác suất | `src/models/naive_bayes.py` |
| 2 | Hồi quy logistic đa lớp (softmax) | mini-batch Gradient Descent, L2 | `src/models/logistic_regression.py` |
| 3 | MLP (ReLU, dropout, momentum) | lan truyền ngược viết tay, early stopping | `src/models/mlp.py` |
| 4 | AdaBoost-SAMME (cây Gini có trọng số) | boosting tuần tự | `src/models/adaboost.py` |

**English summary.** Handwritten digit classification on MNIST with four classifiers implemented from scratch,
a leak-free reproducible protocol (stratified 50k/10k train/validation split, official 10k test), validation-only
hyper-parameter search, scikit-learn parity tests, W&B tracking (offline fallback), ONNX export to Hugging Face,
an in-browser demo (React + ONNX Runtime Web) and a LaTeX report with slides.

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
uv run python -m src augment                  # (tuỳ chọn) nghiên cứu tăng cường dữ liệu -> results/augmentation.json
uv run python -m src report                   # bảng + hình (results/tables, results/figures) và cập nhật README
uv run python -m src export                   # ONNX/JSON + kiểm chứng số học -> artifacts/bundle
uv run python -m src upload --dry-run         # xem sẽ đẩy gì lên Hugging Face (bỏ --dry-run để đẩy)
```

`--smoke` chạy bản siêu nhỏ (vài giây) để kiểm tra pipeline; `--results-dir`/`--artifacts-dir` đổi nơi ghi.
Notebook tái lập kết quả chính: `notebooks/reproduce_results.ipynb` (không phụ thuộc trạng thái ẩn).

### Theo dõi thí nghiệm (Weights & Biases)
`--wandb auto` (mặc định) dùng chế độ online nếu có `WANDB_API_KEY` (hoặc đã `wandb login`), ngược lại **offline** (ghi vào `./wandb`,
đồng bộ sau bằng `wandb sync wandb/offline-run-*`); `--wandb disabled` tắt hẳn. Mỗi run ghi tên mô hình, siêu tham số, seed, git commit,
mã băm phép chia dữ liệu, loss/accuracy theo epoch, ma trận nhầm lẫn, thời gian và artifact mô hình. Nhật ký JSON trong `results/` luôn được ghi, không phụ thuộc W&B.

### Mô hình và website
Trọng số **không** được commit. Bundle (ONNX + JSON + model card) nằm trên Hugging Face:
<https://huggingface.co/TranTrinhNhan1111/WriteRHand-MNIST>. Website tĩnh (React + TypeScript + Vite + ONNX Runtime Web) nằm trong `web/`
(xem `web/README.md`); CI build và triển khai lên GitHub Pages bằng `.github/workflows/pages.yml`.

## Giao thức thực nghiệm (tóm tắt)
- 60 000 ảnh huấn luyện chính thức → **50 000 train / 10 000 validation** (phân tầng, seed 42); 10 000 ảnh test chính thức.
- Chuẩn hóa `(pixel/255 − mean)/(std + 1e-8)` với `mean`, `std` chỉ tính trên train; cùng công thức dùng trong trình duyệt.
- Chọn siêu tham số bằng quét tuần tự từng tham số, chọn theo validation; mọi cấu hình đã chạy được ghi lại. Hàm tìm kiếm không nhận mảng test.
- Mô hình cuối huấn luyện trên 50 000 ảnh train; báo cáo train/val/test với accuracy, macro precision/recall/F1, confusion matrix.

## Báo cáo và slide
```bash
cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex   # cần TeX Live có T5 + lmodern
uv run python slides/build_slides.py && node slides/export_pdf.mjs                        # slide (cần playwright-core + Chromium cho PDF)
```
Các con số trong báo cáo và slide được đọc từ `results/` (xem `src/experiments/texfrag.py`), không nhập tay.

## Cấu trúc thư mục
```
src/                 models/ (4 bộ phân lớp), preprocessing/, evaluation/, experiments/ (tune, train, report, augment), export/ (ONNX, Hugging Face), cli.py
tests/               kiểm thử đơn vị + đối chiếu scikit-learn (72 kiểm thử, chạy offline)
notebooks/           EDA, reproduce_results (chính), fine-tuning, final_eval_with_test_set, eval_for_ensemble (khám phá)
results/             tuning/ final/ baselines/ tables/ figures/ (sinh ra bởi chương trình)
report/              báo cáo LaTeX (main.tex → main.pdf, các đoạn sinh tự động ở generated/)
slides/              slide HTML (index.html) + PDF; dựng lại: uv run python slides/build_slides.py
web/                 website demo      .github/workflows/   CI + GitHub Pages
```

## Thành viên
Trần Trinh Nhân (24280088) và Lâm Nhật Tiến (24280029), đóng góp 50%/50% (nhóm tự thống nhất); chi tiết ở mục 10 của báo cáo. Phần hoàn thiện đồ án được thực hiện **với sự hỗ trợ của
trợ lý lập trình AI (Claude Code)** theo đúng công bố trong báo cáo. Quy trình nhóm: [WORKFLOW.md](WORKFLOW.md).

## Tài liệu / nguồn
LeCun et al. (1998) cho MNIST; Zhu et al. (2009) cho SAMME; xem danh mục đầy đủ trong báo cáo.
