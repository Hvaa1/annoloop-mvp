# Cài đặt trên máy mới

## Điều kiện

Profile đầu tiên dành cho **Ubuntu 22.04 / Linux x86_64, Python 3.10**. Train/predict cần NVIDIA GPU và driver phù hợp CUDA 11.8. Kiểm tra `nvidia-smi` trước khi cài; script không thay driver hay chạy train.

Cần compiler GCC/G++ 6–11, `make`, Python 3.10 và development headers (`Python.h`). Trên Ubuntu 22.04, nếu thiếu:

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev build-essential g++-11
```

Script tự bootstrap pip cả khi hệ thống thiếu ensurepip. Nếu compiler mặc định quá mới, chọn compiler tương thích:

```bash
CC=gcc-11 CXX=g++-11 bash scripts/setup.sh --gpu
```

Cần dung lượng trống cho hai venv, Torch CUDA, cache tải và compiler. Thời gian cài phụ thuộc tốc độ mạng/CPU. Bộ nhớ GPU cần kiểm tra với model/dataset thực; 4 GB chỉ là cấu hình pilot nhỏ, không bảo đảm train được mọi class hoặc point cloud.

## Setup làm gì?

- Tạo `.venv` cho Streamlit và `.venv-gpu` cho model, không hạ phiên bản venv đang dùng.
- Cài dependency theo `requirements-ui.txt` / `requirements-gpu.txt` cùng file constraints đã kiểm chứng, Torch từ index chính thức.
- Tải CUDA 11.8 nvcc/cudart/CCCL từ NVIDIA, kiểm tra SHA-256 rồi đặt ở `runtime/cuda-11.8`; không cần cài CUDA toàn hệ thống.
- Biên dịch năm extension ST3D bằng Ninja từ mã nguồn trên máy clone, không copy `.so` của máy khác. Mặc định hai compiler worker, bỏ debug symbols để giảm RAM. Máy ít RAM có thể dùng `MAX_JOBS=1 bash scripts/setup.sh --gpu`.
- Chạy preflight import CPU. Đây chưa phải kiểm tra GPU forward/backward của dự án; thao tác đó nằm trong UI sau khi chọn dữ liệu/checkpoint.

```bash
bash scripts/setup.sh --gpu
bash scripts/start_ui.sh
```

[PyTorch: phiên bản cũ](https://pytorch.org/get-started/previous-versions/) · [NVIDIA: CUDA redistributables](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_11.8.0.json).

## Chạy trên venv sẵn có

Mặc định dùng venv của checkout. Nếu quản trị viên đã chuẩn bị môi trường tương thích, có thể chọn nó mà không chạy setup:

```bash
VINAI_PYTHON=/path/to/ui-venv/bin/python \
ANNOLOOP_WORKER_PYTHON=/path/to/gpu-venv/bin/python \
ANNOLOOP_CUDA_TOOLKIT=/path/to/cuda-11.8 \
bash scripts/start_ui.sh
```

GPU venv cần đúng Torch/CUDA/spconv và các dependency trong profile; runtime vendored cần được build với chính venv đó. Không trỏ vào venv UI chỉ vì nó import được Torch. Có thể lưu overrides trong `.env` (Git bỏ qua), xem `.env.example`.

## Khi cài/chạy lỗi

| Hiện tượng | Cách kiểm tra |
|---|---|
| Thiếu Python 3.10/compiler | Cài điều kiện ở trên; kiểm tra `python3.10 --version`, `g++ --version` |
| `nvidia-smi` lỗi | Sửa driver trước; setup không sửa driver |
| GPU mới không hợp CUDA 11.8 | Cần profile model/CUDA đã kiểm chứng mới; không tự chuyển detector sang CPU |
| Compile thiếu header/compiler | Đọc log build; toolkit cần `include/cuda/std/cassert`, GCC/G++ đúng phiên bản |
| Thiếu `easydict`/`pcdet` | Chạy lại `setup.sh --gpu`; kiểm tra app đang dùng đúng `ANNOLOOP_WORKER_PYTHON` |
| `Address already in use` | Đổi `ANNOLOOP_PORT=8503`; không tắt tiến trình khác |
| Không có checkpoint trong danh sách | Repo không có weights; train source hoặc nhập checkpoint + metadata đã xác minh |
| Calibration thiếu mẫu | Chọn checkpoint/source validation tốt hơn hoặc thu hẹp class; không lấy gold target để bù |
| GPU bận | Chờ job hiện tại hoặc dừng bằng UI; A/B luôn dùng GPU lần lượt |

`Ctrl+C` dừng server UI; trainer nền vẫn có thể chạy. Dùng UI/monitor để xem PID và dừng đúng job. Không xóa `data/` khi đang chạy.

## Kiểm tra code

```bash
bash scripts/setup.sh --ui --test
bash scripts/check.sh
```

`--test` cài CPU Torch riêng trong UI venv cho test lưu/khôi phục Adam. Nó không tạo bản detector CPU. Suite không gửi train thật hoặc upload CVAT.
