# Ghép dự đoán từ checkpoint local

Ở **2 · Class & model**, chọn **Ghép checkpoint local**. Bảng checkpoint hiển thị
cả các model đã tải nhưng chưa tương thích; không phải tải xong là chạy được.
Chọn nhiều checkpoint hoặc bấm chọn tất cả đủ điều kiện, chọn class và xác nhận
XYZ/mét cùng số cột BIN. Bấm lưu, qua bước 3 rồi chạy ở bước 4. Bước 5 xem sample
BEV overlay point cloud và nguồn từng box, tải JSON cùng gói shape cho exporter.

Luồng này là **inference ensemble**, chưa phải self-training N detector qua nhiều
vòng. Lịch sử M của A/B giữ riêng; quay về A/B ngay ở bước chọn model. Không trộn
trọng số `.pth`, không đổi checkpoint khởi tạo, không tự train hoặc upload CVAT.

## Cách kiểm tra và hợp nhất

1. Quét receipt source jobs, target trained jobs/experiments, model dự án, cache
   và `data/model_imports/*/*/provenance.json`. Không quét tùy ý toàn ổ đĩa và
   không load pickle lúc mở trang. Checkpoint ngoài app cần đăng ký config/nguồn.
2. Chỉ native SECONDNetIoU/PVRCNN XYZ một sweep, có config trỏ đúng source
   train/validation của dự án, mới có thể gửi kiểm tra. Metadata/config không
   tự chứng nhận trọng số đúng. Generic framework/DSNorm chưa được mở quyền chạy.
3. Khi bấm chạy: kiểm tra hash, dataset/source identity, runtime, GPU idle, giữ
   GPU lease chung. Chạy từng model tuần tự: strict-load + inference source
   validation → calibration per class (ít nhất 10 dự đoán/class) → predict toàn
   split adaptation. Lỗi một model dừng lượt; giữ log và artifacts, không dùng
   thiếu frame như một dự đoán rỗng. Không có optimizer step.
4. Checkpoint cùng SHA256 chỉ dùng một lần. Cùng kiến trúc chỉ tính một nhóm phiếu
   bảo thủ, tránh nhiều epoch của một detector lấn át detector khác. Đây là quy
   tắc chống đếm lặp, không phải bằng chứng các kiến trúc độc lập thống kê.
5. So box cùng class, IoU ≥ 0.5 với box seed. Chỉ nhóm có hỗ trợ class đó tham gia
   mẫu số. Mỗi nhóm có ít nhất một box confidence ≥ 0.7, quality ≥ 0.5 và ≥ 5 point
   mới có phiếu. Cần ít nhất hai nhóm và nửa số nhóm hỗ trợ class. Nếu class chỉ
   có một nhóm: confidence ≥ 0.9, quality ≥ 0.6. Giữ box thực có quality tốt nhất
   trong bằng chứng đạt ngưỡng; không nội suy/average box thiếu chứng cứ.
6. Bất đồng, ít point hoặc xung đột class chuyển ignore. Lưu confidence/quality,
   hash checkpoint và provenance mỗi box. Thống kê đầu ra không phải accuracy.

## Artifacts và giới hạn

Lựa chọn ở `projects/<id>/ensemble.json`; mỗi lần chạy tạo
`ensemble_jobs/<uuid>/` riêng, gồm spec, state/PID, worker.log, log inference,
calibration, calibrated predictions, frames, result và `cvat_shapes.json`.
Source calibration được giữ riêng khỏi target artifact. Không đọc nhãn target;
không dự đoán holdout để tạo pseudo-label. View/refresh không dispatch model.
Dừng gửi SIGTERM cho controller; controller chờ child inference thoát rồi nhả
GPU lease. Chưa có resume cache cho lượt ensemble, chạy lại tạo lượt mới.

Gói shape tại trang sample dùng class-name contract của exporter AnnoLoop, chưa
phải ZIP import trực tiếp CVAT. Để bàn giao, bấm **final predict & CVAT** và làm
theo [FINAL_DELIVERY.md](FINAL_DELIVERY.md): chọn phạm vi, chạy final, chuẩn hóa
media rồi upload task shape bằng SDK. Không tạo temporal track và chưa có
N-model target training.
Các file official Waymo → Lyft trong `model_imports/st3d_official_20261015` hiện
vẫn bị chặn: chưa strict-load/kiểm chứng head, config và source calibration;
không ép `Vehicle` thành car/pedestrian. Không lấy classification score làm IoU.

Kiểm tra triển khai dùng fixture/mocked adapter; chưa chạy GPU thật để chứng nhận
luồng N-model hoặc khẳng định nhãn tốt hơn A/B. Người dùng bấm chạy mới thực hiện
kiểm tra và inference thật; calibration kém có thể cho ít hoặc không có box nhận.
