# Giao diện AnnoLoop

Chạy `bash scripts/start_ui.sh` từ root repo, mở http://localhost:8502.

- `onboarding.py`: folder picker và wizard source → target → nhãn/checkpoint.
- `app.py`: kế hoạch, theo dõi train, review/rollback, CVAT, hướng dẫn.
- `components.py`: cards/empty states, biểu đồ box có dữ liệu thật.
- `assets/style.css`: nền sáng, tím, responsive, font hệ thống offline.

Default workflow không có field JSON/YAML; setup service tự sinh trong project.
Checkpoint đủ class bỏ pretrain; chỉ thiếu class mới xuất hiện nút học nguồn.
Native adaptation/final CVAT chỉ mở khi đủ điều kiện thực. Bước 5 xem kết quả
accepted/ignore và so sánh phiên bản; bước 6 final inference và xuất shape CVAT.
Hướng dẫn đầy đủ: [USER_GUIDE](../docs/USER_GUIDE.md).
