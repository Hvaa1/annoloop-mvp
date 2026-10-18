# Kết nối model đã tải

Chạy từ thư mục VIN_Ai, dùng venv VIN_Ai:

```bash
bash buildphase/annoloop_mvp/scripts/connect_model.sh --model MODEL_ID
```

Kiểm tra thêm feature/class với một dự án đã lưu:

```bash
bash buildphase/annoloop_mvp/scripts/connect_model.sh --model MODEL_ID --project PROJECT_ID
```

Hoặc kết nối lại các gói đã tải, không tải thêm:

```bash
bash buildphase/annoloop_mvp/scripts/connect_model.sh --all-installed
```

`--data /path/to/data` chọn nơi lưu trạng thái khác. Model ID lấy từ kho model;
script dùng registry có sẵn, không nhận URL/config Python tùy ý để thực thi.

Script kiểm tra hash weights và toàn bộ config đã pin, nhận diện framework,
tạo `adapter.json` nối với code `FrameworkAdapter.predict_raw` / framework worker
đã có. Kết quả là **connected_unverified**, chưa phải chứng nhận chạy GPU.
Framework hoặc modality chưa hỗ trợ ghi **unsupported**, không tạo adapter giả.

Kết quả trong `data/model_cache/MODEL_ID/`:

- `adapter.json`: config, weights, class, hash và recipe của framework.
- `connection.json`: kết quả lần gần nhất, entrypoint hiện có và phần còn thiếu.
- `connections/<attempt>/report.json`: lịch sử kết nối, không ghi đè lần trước.

Luồng tải của app/CLI (`model_hub download` và `setup`) tự gọi bước kết nối sau
khi tải thành công. Không kết nối trên page refresh. API byte downloader cấp
thấp `download()` vẫn chỉ tải file; caller dùng `connect()` nếu gọi trực tiếp.

**Giới hạn:** script không tự sinh code loss, calibration hoặc signed-ignore.
Hiện adapter registry vẫn chỉ có raw predict; các entrypoint chưa có ghi `null`
và `adaptation_ready=false`. Không cài runtime, load weights bằng Torch, chạy
GPU hoặc train. Có thể thiếu runtime dù binding tạo thành công. Chọn `--project`
không đổi model A/B, không sửa verification hoặc snapshot nhãn, không đọc gold.

## Nếu checkpoint đến từ ST3D hoặc repo liên quan

Đã có đường import/kết nối riêng cho **hai payload ST3D/3DOD-LSF đã pin**:

```bash
bash buildphase/annoloop_mvp/scripts/connect_model.sh --native-source lsf-secondiou-waymo-source --checkpoint /path/secondiou.pth
```

Xem [hướng dẫn native source](NATIVE_SOURCES.md). Chọn chúng ngay ở Class & model;
không đi qua generic adapter. Vehicle không tự thành car/pedestrian; class mới
vẫn cần source có nhãn. Chỉ các payload trong catalog native được nhập, không
nhận tùy ý checkpoint ngoài danh mục hoặc tự cấp chứng nhận GPU.

[ST3D](https://github.com/CVMI-Lab/ST3D) có config SECOND-IoU, PV-RCNN và
PointRCNN; [3DOD-LSF](https://github.com/Tiffamy/3DOD-LSF) công bố checkpoint
SECOND-IoU/PV-RCNN và dựa trên ST3D. Đây là ứng viên gần runtime native hiện có,
nhưng script registry ở trên chưa nhập tùy ý checkpoint ngoài danh mục vào native.
Cần kiểm tra revision runtime, config/head/class và feature của từng checkpoint;
không gắn `classes_verified` từ tên file hoặc đổi Vehicle thành car.

[MS3D](https://github.com/darrenjkt/MS3D) có thêm PV-RCNN++, VoxelRCNN và IA-SSD,
với các cấu hình sweep khác nhau; không coi là dùng ngay trong ST3D fork cũ.
Nguồn repo chứa train/self-training không thay thế calibration source validation
riêng của AnnoLoop. Adapter native đã nối hiện vẫn giới hạn SECOND-IoU/PV-RCNN.
