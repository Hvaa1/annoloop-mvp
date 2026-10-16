# Checkpoint ST3D và nguồn liên quan

## Dùng trong app

1. Tạo dự án với source có nhãn và target point cloud, chọn class cần gán.
2. Ở **Class & model**, chọn pretrained **3DOD-LSF · SECONDNetIoU** cho A
   và **3DOD-LSF · PVRCNN** cho B. Đây là initializer native, không phải generic
   MMDetection3D/OpenPCDet.
3. Xác nhận để sang **Chuẩn bị model**. Bấm **Tải pretrained ST3D/LSF** cho
   lựa chọn cần dùng. Tải chạy nền, chỉ lấy member checkpoint đã pin trong ZIP,
   không tự tải toàn archive 4,4 GB. SHA256 sai hoặc server không hỗ trợ range
   sẽ dừng, không đăng ký weights lỗi. Link remote hiện không được kiểm tra lại
   bằng download thật trong lần triển khai này.
4. Bấm **Kết nối initializer A/B vào dự án**. Tool tự tạo metadata và cấu hình
   source trong dự án. Không cần viết JSON/YAML. Không khởi chạy GPU/train.
5. Nếu checkpoint thiếu nhãn yêu cầu, bấm **Học các class còn thiếu**. Worker
   giữ weights tương thích, khởi tạo lại head khi taxonomy khác. A rồi B chạy
   tuần tự; checkpoint mỗi 10 epoch và cuối. Nếu đã đủ class, bỏ qua học nguồn.
6. Sau học nguồn, bấm tiếp tục adaptation, xác nhận tọa độ và kiểm tra GPU.
   Calibration dùng source validation của từng checkpoint, không dùng gold target.

**Class gốc của cả hai file được ghim hiện là `Vehicle`.** Không coi chúng đã
biết `car` hoặc `pedestrian`. Với yêu cầu car + pedestrian, cần học các nhãn đó
trên source có nhãn. Đây là học head mới trên backbone pretrained; không phải
tự học được class mới chỉ từ target không nhãn. Lựa chọn tải hoặc kết nối không
tự train, không đổi snapshot nhãn hoặc chứng nhận GPU đã lưu.

## Import file đã tải bằng script

Từ thư mục VIN_Ai:

```bash
bash scripts/connect_model.sh \
  --native-source lsf-secondiou-waymo-source \
  --checkpoint /duong/dan/secondiou.pth
```

File phải khớp chính xác SHA256 và kích thước catalog; không suy đoán class từ
tên file. Thay ID bằng `lsf-pvrcnn-waymo-source` cho PV-RCNN. Chỉ metadata
payload có bằng chứng class/config đã ghi nhận mới được đăng ký. Nếu cần gắn
initializer vào cấu hình source của dự án, thêm `--project ID --role A` (hoặc B).
Thao tác này không tự đổi lựa chọn A/B; chọn cùng initializer ở màn Class & model.

Xem danh mục hoặc tải qua CLI:

```bash
PYTHONPATH=buildphase/annoloop_mvp/src .venv/bin/python -m annoloop.native_sources list
PYTHONPATH=buildphase/annoloop_mvp/src .venv/bin/python -m annoloop.native_sources download lsf-secondiou-waymo-source
```

Cache ở `data/native_model_cache/ID/`. Đường tải lại tạo file tạm riêng, giữ file
tải dở nếu lỗi; hiện chưa resume byte-offset member nén, thử lại sẽ đọc member từ
đầu. `ANNOLOOP_DATA_DIR` / `--data` chọn storage khác. Không cần demo trên máy
người dùng để đọc catalog hoặc lấy weights. Runtime ST3D vẫn cần được cài và
build riêng như adapter native đang dùng.

## Các model trong repo chính thức

[CVMI-Lab/ST3D](https://github.com/CVMI-Lab/ST3D#model-zoo) có các cấu hình/
thí nghiệm SECOND-IoU, PV-RCNN và PointRCNN. Class phải đọc theo config/checkpoint
cụ thể; bảng nhiều cột Car/Ped/Cyc không chứng minh một file weights chứa cả ba.
README có link pretrained nuScenes → KITTI; checkpoint Waymo có hướng dẫn cấp
quyền riêng. Không biến link thư mục SharePoint thành URL weights đã kiểm chứng.

[3DOD-LSF](https://github.com/Tiffamy/3DOD-LSF) công bố hai họ SECOND-IoU/PV-RCNN
dựa trên ST3D. Hai member đã pin ở đây là `ckpt/waymo_nus/source_only/*.pth`,
với hash đối chiếu từ payload đã kiểm tra trước trong workspace. Native profiles
được trích config ST3D revision `8100ff44a7a7c4e7087363eec101831b343a99b7`, giữ
attribution/license tại `src/annoloop/catalog/NATIVE_SOURCES_LICENSE.txt`.

Thêm nguồn này không mở adapter PointRCNN hay đảm bảo mọi fork ST3D chạy được.
Nguồn chính thức được liên kết trong UI; checkpoint chưa pin identity/config
chưa được cho chạy native. Hai kiến trúc native hiện vẫn là SECOND-IoU/PV-RCNN.
