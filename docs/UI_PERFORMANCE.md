# Tối ưu giao diện và đường đọc dữ liệu

Đo trên dự án Test đã có, không chạy model, không đọc target gold và không gọi
CVAT. Thời gian dưới đây là thời gian hàm backend, không phải thời gian tải toàn
trang hoặc benchmark trình duyệt. Ổ đĩa và cache hệ điều hành ảnh hưởng kết quả.

| Thao tác | Trước | Sau |
|---|---:|---:|
| Tìm sample các vòng đã lưu | 26,7–27,5 s | 2,70 s lần đầu; 1,60 s lần tiếp |
| Kiểm tra gói CVAT khi xem | 1,07–1,42 s/lượt | 1,10 s lần đầu; 0,02 s lần tiếp |
| Tìm checkpoint local | 1,25–1,51 s | 0,52–0,61 s |

## Thay đổi

- `previews.saved_attempts`: một lần đọc chỉ mục thư mục frame cho mỗi lượt
  predict, dùng lại giữa các vòng. Không stat từng frame lặp lại; mỗi lần refresh
  vẫn thấy file mới/xóa và giữ kiểm tra frame/hash/config liên kết.
- `jobs.process_alive`: chỉ đọc receipt PID/start identity. Kiểm tra worker còn
  sống không cần parse YAML và đọc log. Vẫn phát hiện trainer mồ côi ở nested
  attempts và từ chối PID đã tái sử dụng.
- `final_delivery.preview_export`: cache giới hạn 8 gói, vô hiệu hóa theo inode,
  size, mtime, ctime của các file liên quan. Trả bản sao document để UI không
  sửa dữ liệu cache. Upload luôn gọi kiểm tra SHA256 đầy đủ riêng.
- `local_ensemble.discover`: đọc mỗi YAML một lần trong mỗi lượt khám phá;
  không lưu toàn cục cấu hình có thể đã đổi.
- Plot point cloud dùng mẫu tối đa 12.000 điểm và WebGL, giữ số điểm gốc để
  phân biệt dữ liệu với phần hiển thị. Box không bị bỏ khỏi kết quả gốc.
- Điều hướng có bước Xuất CVAT riêng, tên/ID dự án luôn hiện; viewer final đọc
  cả accepted/ignore. Giữ so sánh phiên bản, chọn vòng và rollback.

Lần đầu đọc media, dataset lớn, biểu đồ nhiều box hoặc ổ đĩa chậm vẫn có thể
chờ. Đây không phải thay đổi tốc độ train/inference GPU. Không thay đổi policy
nhãn hoặc giảm ngưỡng để làm giao diện nhanh hơn.
