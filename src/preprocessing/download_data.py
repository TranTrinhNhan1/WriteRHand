from pathlib import Path
from torchvision.datasets import MNIST

# Xác định thư mục gốc của project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # lấy đường dẫn của file đang chạy, sau đó truy về folder cha 3 lần để ra folder project

# Xác định nơi lưu dataset
RAW_DIR = PROJECT_ROOT / "data" # code tự tạo thư mục raw bruh
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Dataset path:", RAW_DIR)

# Tải và khởi tạo tập train: 60.000 ảnh
train_dataset = MNIST(
    root=str(RAW_DIR),
    train=True,
    download=True
)

# Tải và khởi tạo tập test: 10.000 ảnh
test_dataset = MNIST(
    root=str(RAW_DIR),
    train=False,
    download=True # nhung lan sau neu dl day du se khong tai lai du lieu
)

print("Tải MNIST thành công!")
print("Số mẫu train:", len(train_dataset))
print("Số mẫu test:", len(test_dataset))

# Kiểm tra một mẫu
image, label = train_dataset[0]
print("Kích thước ảnh:", image.size)
print("Nhãn của ảnh:", label)
