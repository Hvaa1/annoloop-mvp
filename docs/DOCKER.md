# Chạy AnnoLoop bằng Docker trên GPU của bạn

App, worker GPU và dữ liệu đều chạy trên máy của bạn. Không cần cài Python, CUDA hay
biên dịch ST3D bằng tay — image đã có sẵn.

## Máy cần có

| | Yêu cầu |
|---|---|
| GPU | NVIDIA, compute capability 5.0–9.0 (Maxwell → Hopper) |
| Driver | NVIDIA driver hỗ trợ CUDA 11.8 trở lên (`nvidia-smi` chạy được) |
| Hệ điều hành | Linux x86_64. Windows: Docker Desktop + WSL2 (chưa kiểm chứng) |
| Docker | Docker Engine 24+ và **NVIDIA Container Toolkit** |
| Ổ đĩa | ~20 GB cho image + chỗ cho dataset, checkpoint, kết quả |

macOS không có CUDA nên không chạy được phần GPU.

### Cài NVIDIA Container Toolkit (Ubuntu)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi   # phải thấy GPU
```

Hướng dẫn chính thức: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Chạy

```bash
git clone https://github.com/Hvaa1/annoloop-mvp.git
cd annoloop-mvp
mkdir -p data datasets            # đặt dataset của bạn vào ./datasets (hoặc xem bên dưới)
docker compose up -d --build      # lần đầu build ~30–60 phút: tải Torch, biên dịch ST3D
```

Mở **http://localhost:8502**. Dừng: `docker compose down`. Cập nhật code: `git pull && docker compose up -d --build`.

### Dataset ở chỗ khác

```bash
ANNOLOOP_DATASETS=/duong/dan/toi/datasets docker compose up -d
```

Folder này được gắn **chỉ đọc** vào `/datasets` trong container; bộ chọn thư mục của app
chỉ duyệt trong `/datasets`. Chuẩn bị nuScenes source và `pointcloud_format.json` cho file BIN
trước khi gắn (xem README và `scripts/prepare_source.sh`).

### Dữ liệu được lưu ở đâu

`./data` trên máy bạn: dự án, phiên bản nhãn, checkpoint đã tải và đã train, log, gói bàn giao.
Xóa container không mất dữ liệu; xóa `./data` thì mất.

### Đổi cổng, chỉ định GPU

```bash
ANNOLOOP_PORT=8600 docker compose up -d                      # mở http://localhost:8600
NVIDIA_VISIBLE_DEVICES=1 docker compose up -d                # chỉ dùng GPU số 1
```

## CVAT

CVAT chạy riêng (theo hướng dẫn của CVAT). Nếu CVAT ở cùng máy, trong form bàn giao của app
nhập địa chỉ `http://host.docker.internal:8080`.

## Kiểm tra nhanh sau khi cài

```bash
docker compose exec annoloop bash -lc 'source scripts/env.sh && $ANNOLOOP_WORKER_PYTHON -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

Trong app, bước **3 · Chuẩn bị model** chạy thử GPU thật (predict, backward, ignore, calibration)
trước khi train — đây là bài kiểm tra quyết định GPU của bạn có chạy được model đã chọn không.

## Giới hạn

- GPU chạy **một job tại một thời điểm** trong mỗi máy.
- VRAM nhỏ (≤ 4 GB) chỉ chạy được model nhỏ; model vùng phủ rộng có thể hết bộ nhớ.
- Image không chứa dataset, checkpoint hay mật khẩu. Checkpoint tải trong app lưu vào `./data`.
- Mỗi máy có lịch sử dự án riêng; chưa có server dùng chung.
