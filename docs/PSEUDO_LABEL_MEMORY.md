# Bộ nhớ pseudo-label qua các vòng

Một box có ID bộ nhớ cố định trong **cùng một frame** khi ghép được qua các
phiên bản model. Đây là lịch sử M0 → M1 → M2, riêng với track giữa các frame.
Bộ nhớ chỉ đọc dự đoán/nhãn tự động và point cloud; không đọc gold target.

## Box được giữ hoặc loại như thế nào?

| Tình huống | Kết quả mặc định |
|---|---|
| Box hiện tại được bước hợp nhất chấp nhận, đủ điểm và điểm chất lượng | Cập nhật box, ghi confidence/quality/IoU vào lịch sử |
| Đã có ít nhất 2 xác nhận ổn định từ các checkpoint khác nhau, rồi bị bỏ sót 1 lần | Giữ nhãn dương nếu còn ít nhất 5 điểm trong box |
| Chỉ có một detector còn hỗ trợ mạnh | Có thể giữ trong cửa sổ bỏ sót trên; không tạo thêm một xác nhận đầy đủ |
| Bỏ sót lần thứ 2 | Chuyển sang ignore |
| Bỏ sót lần thứ 3 | Hết hạn trong bộ nhớ, không phát ra box bộ nhớ để train |
| Class mâu thuẫn, quality/confidence giảm mạnh hoặc bằng chứng hiện tại yếu | Ignore ngay, mất chuỗi ổn định |
| Box xuất hiện lại sau hết hạn | Có thể cập nhật, nhưng phải tích lũy ổn định lại để được giữ khi bỏ sót |
| Cùng checkpoint được predict lại | Ghi lịch sử nhưng không tăng số xác nhận độc lập |

Ví dụ: M0 và M1 xác nhận cùng một box; M2 không thấy nó nhưng point cloud vẫn
đủ điểm → giữ. M3 vẫn không thấy → ignore. M4 vẫn mất → hết hạn.
Box ignore từ các dự đoán khác vẫn có thể tồn tại nếu còn bằng chứng bất đồng;
hết hạn bộ nhớ không tự biến vùng ấy thành background.

Các ngưỡng mặc định: IoU ghép **0.3**, IoU ổn định **0.5**, confidence **0.7**,
quality **0.5**, ít nhất **5 điểm**, giảm confidence/quality tối đa **0.1**.
Điểm đã calibration trên source validation; bộ nhớ không tự nâng/đổi score cũ
thành score của một dự đoán mới. Box chưa từng được hợp nhất chấp nhận vẫn ở
luồng ignore/review thông thường, chưa được giữ dương nhờ lịch sử.

## Nhãn tốt nhất và lịch sử là hai phần riêng

Mỗi vòng hoàn tất lưu quan sát vào bộ nhớ, kể cả vòng bị bộ chọn tự động từ chối.
**Train vẫn dùng nhãn của phiên bản được chọn.** Quan sát trong bộ nhớ của M2 bị
từ chối không tự trở thành nhãn train thay M1. Khi rollback về M1, lịch sử các
vòng sau vẫn được giữ để theo dõi; không sửa nhãn/checkpoint M1.

Nhãn giữ nhờ lịch sử được đếm riêng. Chúng **không làm tăng agreement** của các
dự đoán hiện tại. Nhờ vậy, cơ chế memory không tự giúp một vòng vượt ngưỡng 60%.
Box count và point support được cập nhật theo nhãn đầu ra thực.

## Dùng trên giao diện

1. Mở **Vòng adaptation → Bộ nhớ pseudo-label qua các vòng**.
2. Bật **Giữ lịch sử từng box**, chọn số xác nhận và số vòng được giữ khi bỏ sót,
   rồi **Lưu thiết lập adaptation**. Mặc định bộ nhớ bật cho lượt chạy mới.
3. Sau khi vòng hoàn tất, mở **So sánh phiên bản → Bộ nhớ pseudo-label qua các vòng**.
4. Chọn vòng, frame và box để xem lịch sử score, IoU, số điểm, bỏ sót và mâu thuẫn.

Thay đổi áp dụng từ lượt controller mới. Worker đang chạy tiếp tục theo code và
thiết lập đã nạp; công việc này không dừng hay khởi chạy lại worker đó.

## File và khôi phục

- Logic ghép/voting: [pseudo_memory.py](../src/annoloop/pseudo_memory.py).
- Đọc snapshot cũ: [memory_store.py](../src/annoloop/memory_store.py).
- Mỗi attempt mới lưu `memory_before.json` và `memory_after.json`.
- Snapshot vòng mới lưu `rounds/Mn/memory.json`; `round.json` giữ hash và thống kê.
- Resume kiểm tra hash và policy, dùng lại memory trước vòng để không đếm lịch sử
  hai lần. Chọn/rollback kiểm tra cả hash memory nếu snapshot có file này.
- Với vòng legacy, lịch sử được dựng từ nhãn/dự đoán tự động đã lưu. Initial
  pseudo-label M0 chỉ được dùng khi hash khớp receipt train của cả A/B. Mỗi point
  cloud đọc một lần khi replay; không ghi thêm/sửa memory vào các snapshot cũ.

Đây là cơ chế lấy ý tưởng ensemble/voting của [ST3D++](https://arxiv.org/html/2108.06682v1#S4.SS3.SSS3),
có thêm điều kiện số lần xác nhận, point support và score regression. Nó chưa
tái hiện đầy đủ thuật toán paper; curriculum augmentation, DSNorm và EMA chưa
được thêm trong thay đổi này. Memory không bảo đảm nhãn đúng và không tạo được
pseudo-label cho class mà detector chưa bao giờ đưa ra bằng chứng đủ tốt.
