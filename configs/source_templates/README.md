# Template source nuScenes mini

Chọn file theo model và các class cần học:

| Model | Chỉ car | Car + pedestrian |
|---|---|---|
| SECOND-IoU | `secondiou_nuscenes_car_xyz.yaml` | `secondiou_nuscenes_car_pedestrian_xyz.yaml` |
| PV-RCNN | `pvrcnn_nuscenes_car_xyz.yaml` | `pvrcnn_nuscenes_car_pedestrian_xyz.yaml` |

Dùng ở ô **Template YAML (nâng cao)** trên trang tạo cấu hình nguồn. Chọn đúng class dự án;
chọn thêm `motorbike` sẽ bị chặn vì template này chưa có anchor class đó.

**Folder source** là output từ `scripts/prepare_source.sh`. App tự xác định
loader root/version và tạo YAML dự án; không cần chỉnh `DATA_PATH` trong template.

Các template đã materialize, không còn base path phụ thuộc working directory.
Source-only, NuScenesDataset, một sweep, XYZ, batch 1; LR 0.0015 lấy từ profile
pilot. ROS và epoch sẽ được form source_config áp dụng. Checkpoint cadence là
10 epoch + cuối do worker quản lý, không điều khiển bằng YAML này.

Anchor và model lấy từ các profile ST3D/demo local; nguồn/hash ở provenance.json.
Pedestrian anchor ban đầu lấy từ profile nuScenes→KITTI pedestrian, không khẳng
định tối ưu cho nuScenes mini. Cần kiểm tra trên source validation. Không đọc
nhãn target Day14 để thiết kế anchor hay tạo template.

Checkpoint Waymo có Vehicle; khi class mới là car/pedestrian, worker bỏ weights
các head không cùng taxonomy, chỉ nạp phần còn tương thích để fine-tune.

Đã kiểm tra YAML, source_config, class/anchor và inventory thật của source;
chưa chạy train/GPU để chứng minh runtime hoặc bộ nhớ đủ cho multiclass. Không
thay đổi tiến trình trainer đang hoạt động khi tạo/sửa template.
