# Giao diện AnnoLoop

Chạy `bash buildphase/annoloop_mvp/scripts/start_ui.sh`, mở http://localhost:8502.

- `onboarding.py`: folder picker và wizard source → target → nhãn/checkpoint.
- `app.py`: kế hoạch, theo dõi train, review/rollback, CVAT, hướng dẫn.
- `components.py`: cards/empty states, biểu đồ box có dữ liệu thật.
- `assets/style.css`: nền sáng, tím, responsive, font hệ thống offline.

Default workflow không có field JSON/YAML; setup service tự sinh trong project.
Checkpoint đủ class bỏ pretrain; chỉ thiếu class mới xuất hiện nút học nguồn.
Native adaptation/final CVAT vẫn bị khóa với blocker thật.
Hướng dẫn đầy đủ: [USER_GUIDE](../docs/USER_GUIDE.md).
