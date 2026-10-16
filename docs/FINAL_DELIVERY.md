# Final predict và bàn giao CVAT local

1. Với A/B: chọn vòng ở **So sánh phiên bản**. Với ghép local: hoàn tất một lượt
   ensemble và chọn lượt đó ở trang **Xuất CVAT**.
2. Chọn phạm vi: adaptation + delivery (mặc định, trừ holdout), chỉ delivery,
   hoặc cả dataset. Lựa chọn cả dataset chỉ suy luận holdout cho bàn giao cuối;
   không đọc gold, không tạo nhãn train và không thay đổi nhãn đang chọn.
3. Bấm **Chạy final predict & tạo gói bàn giao**. Job mới kiểm tra hash/media,
   runtime và GPU idle. Từng checkpoint calibration trên source validation rồi
   predict phạm vi đã chọn, tuần tự dưới GPU lease chung. Không có optimizer.
4. Xem log và sample final, chọn gói đã hoàn tất. Media chuyển thành PCD XYZ,
   tên file đánh số theo đúng thứ tự frame nhãn. Thiếu một frame hoặc hash đổi
   sẽ không công bố kết quả hoàn tất. Vùng ignore không thành nhãn dương.
5. Nhập CVAT host, tài khoản và mật khẩu. Bấm **Kiểm tra đăng nhập CVAT** để
   xác thực riêng; nút này dùng được ngay cả khi chưa có gói final, chỉ login và
   đọc tài khoản hiện tại, không tạo task. Kết quả kiểm tra lưu trong memory UI,
   có thời điểm theo giờ Việt Nam; refresh không tự xác thực lại. Lỗi đăng nhập
   được báo riêng với lỗi kết nối server, không in token hoặc response body.
   Khi đã có gói final, nhập tên task. Bấm **Tạo task CVAT** mới
   gọi API: tạo task mới → upload media → kiểm tra thứ tự → ghi annotation →
   đọc lại class, geometry, frame và provenance. Không ghi đè task có nhãn khác;
   task đã upload/verified được giữ khi người dùng review. Credential chỉ trong
   bộ nhớ UI/call, không ghi vào job, database hoặc receipt.

Luồng A/B dùng checkpoint copy của vòng `selected_version`, không dùng M0 nguồn
và không lấy `labels.json` làm final prediction. Config/head class vẫn theo model
gốc; calibration và nhãn bàn giao theo class active của vòng. Fusion A/B dùng
compare đã calibration (IoU ≥ 0.3, confidence ≥ 0.7, quality ≥ 0.5), thêm ≥ 5
point thật. Ghép N model dùng policy đã lưu của lượt ensemble. Final inference có
thể khác snapshot adaptation vốn có memory/lịch sử nhiều vòng; không ghi đè nó.

Hiện xuất **shape 3D**. Không tạo temporal track khi chưa có pipeline pose/sequence
được nối vào final delivery; không biến mỗi box thành track giả. Không có adapter
final 2D. Model/calibration thiếu điều kiện vẫn dừng và giữ artifacts. Luồng mới
chưa được chạy inference GPU thật hoặc upload CVAT thật trong kiểm tra triển khai;
test dùng fixture/mocked adapter/API, không chứng nhận chất lượng nhãn.

Mỗi lượt nằm ở `projects/<id>/delivery_jobs/<uuid>/`: spec chốt version/hash,
state/PID, log từng model, calibration riêng, frame predictions, `result.json` và
`exports/shape/`. Thư mục export gồm media.zip, media.json (frame mapping),
annotations.json (class-name contract), export.json và annoloop_delivery.zip.
ZIP này là gói vận chuyển của uploader AnnoLoop, không phải định dạng import
annotation ZIP trực tiếp của CVAT. App upload qua SDK để ánh xạ label/attribute ID.
File lớn hơn64MB hiện đường dẫn local thay vì nạp vào download widget.

Receipt state neo SHA256 result; result neo manifest export; manifest neo media
và annotation. Upload bị chặn khi file thay đổi. Lượt bị dừng không có resume
riêng; bấm chạy lại tạo lượt mới và giữ artifacts cũ. Upload cùng một gói có
receipt để tránh tạo task trùng khi thành công; lỗi giữa lúc gửi media có thể cần
kiểm tra task còn dở trước khi thử lại. Không tự xóa task CVAT.

UI/controller dùng venv VIN_Ai; inference dùng model worker venv. Cần extra
`cvat` (`cvat-sdk`) trong venv UI để bật upload. Máy này đã có dependency. Mở trang
và refresh không chạy GPU, train, tải model hoặc gọi CVAT.
