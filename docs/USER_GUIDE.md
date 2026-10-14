# AnnoLoop — hướng dẫn dùng nhanh

## Mở app

```bash
bash scripts/start_ui.sh
```

Mở **http://localhost:8502**. Cài trước bằng `bash scripts/setup.sh --gpu`.
UI dùng `.venv`, worker GPU dùng `.venv-gpu`; mã runtime nằm trong `vendor/ST3D`.
Không chạy `app.py` trực tiếp bằng python. Đổi cổng bằng `ANNOLOOP_PORT=8503`.
Xem [cài đặt trên máy mới](INSTALL.md).

## Chỉ cần chọn folder và nhãn

1. Mở **Dữ liệu**. Bấm **📂 Chọn** để duyệt folder source và target,
   hoặc dán đường dẫn nếu đã biết. Folder nằm trên máy chạy AnnoLoop.
2. Bấm **Đọc hai thư mục**. Tool đọc nhãn source, train/validation,
   media target và class schema nếu target đã có.
3. Chọn **nhãn cần gán**. Target chưa có schema vẫn dùng được: chọn từ class
   nguồn hoặc nhập tên class cần có. Đây là yêu cầu của bạn, không phải nhãn gold.
4. Chọn checkpoint A/B đã nhập hoặc “Không có checkpoint · học mới”. App tự lấy
   metadata theo checkpoint; không yêu cầu bạn nhập JSON.
5. Bấm **Tạo dự án và tự chuẩn bị**. Tool tự lưu schema, inventory, YAML và
   job spec trong `data/projects/<id>/generated/`. Không ghi đè dataset gốc.

Tùy chọn nâng cao nằm trong expander: số vòng, thủ công/tự động,
chọn phiên bản bằng tay/theo ngưỡng và epoch/ROS khi cần học class mới.

## Khi nào cần train nguồn?

- Checkpoint hỗ trợ đủ class được chọn → **bỏ qua pretrain**, dùng trực tiếp.
- Chỉ A hoặc B thiếu class → **chỉ train detector đó**.
- Cả hai thiếu → chạy A rồi B, không chạy đồng thời.
- Chạy nguồn theo cặp A → B: **Dừng trainer này** ở A sẽ chuyển sang B sau khi
  A dừng hẳn và GPU rảnh. Log/checkpoint A được giữ; A vẫn là model dừng sớm.
- Muốn hủy cả lượt: **Dừng cả chuỗi A → B**. B đang chờ sẽ không được chạy.
  A lỗi hoặc mất tiến trình bất thường cũng dừng chuỗi. Chọn chỉ A thì không tự chạy B.
- Source chưa có nhãn class mới → hiển thị thiếu nhãn, chặn train.
- Không có checkpoint → chọn “Không có checkpoint · học mới”.

Model cần học class mới sẽ train trên các class được chọn, gồm class cũ cần
được giữ. Checkpoint nhiều class vẫn có thể dùng cho tập con; tool không cắt
head để giả rằng đó là checkpoint mới. `Vehicle` không tự được coi là car hay
pedestrian. Khác tên/định nghĩa cần mapping đã xác minh hoặc checkpoint thích hợp.

**Bỏ pretrain nguồn không có nghĩa bỏ huấn luyện adaptation.** Các vòng ST3D vẫn
có thể fine-tune trên pseudo-label target để thích nghi với dữ liệu mới.

## Dataset hiện hỗ trợ tự nhận dạng

**Source:** nuScenes có hai file `nuscenes_infos_1sweeps_train.pkl` và
`nuscenes_infos_1sweeps_val.pkl` ở folder phiên bản. Bạn có thể chọn folder
`nuscenes` hoặc `v1.0-mini`; tool tự tìm loader root đúng, không nối version hai lần.
JSON inventory được tạo tự động. Train/validation được kiểm tra trùng token;
metadata `sample.json` có sẵn cho phép kiểm tra thêm trùng scene.
Chuẩn bị nuScenes raw bằng `bash scripts/prepare_source.sh --input /path/raw --output /path/prepared --version v1.0-mini`. Source KITTI/Waymo/format tự tạo còn cần adapter.

**Target:** folder PCD/NPY/BIN cho 3D; nhận dạng ảnh PNG/JPG/JPEG cho 2D.
Schema nếu có chỉ dùng để đề xuất nhãn. Khi bạn chọn nhãn, schema dự án được tạo
riêng; không đọc annotation instances/gold target để chọn nhãn hay train.
Không tự suy ra class từ tọa độ point cloud. Detector/trainer 2D chưa triển khai.

Manifest target nếu có sẽ được giữ: thứ tự frame, split adaptation/holdout/delivery,
sequence, timestamp và pose. Không chia cùng sequence sang các split khác nhau.
Không có manifest: media mặc định thuộc adaptation; difficulty unknown.

## Trang chuẩn bị model

Hiển thị A/B: **dùng checkpoint hiện có** hoặc **cần học class nào**.
Chỉ bấm **Học các class còn thiếu** mới launch source job; việc chuẩn bị không train.
YAML đã được tool tạo từ profile tương ứng, không phải file do người dùng viết.
Template có sẵn anchor car/pedestrian. Với class khác có nhãn và geometry
trong source train, tool tự lấy kích thước trung vị để tạo anchor ban đầu;
không lấy nhãn target để thiết kế anchor. Đây là cấu hình khởi đầu, cần đánh giá
trên source validation. Thiếu geometry thì báo blocker, không tự bịa kích thước. Runtime ST3D mặc định là `vendor/ST3D`; quản trị viên
có thể đổi `ANNOLOOP_ST3D_RUNTIME` khi chuyển máy.

Sau khi job hoàn tất với receipt/checkpoint hợp lệ, app nhận checkpoint mới và
chuyển model sang trạng thái dùng trực tiếp. Không gửi job lại khi refresh.
Checkpoint mỗi **10 epoch + epoch cuối**. Log, PID và checkpoint cập nhật 5 giây.
**Dừng trainer** giữ artifacts; chưa có resume tự động. Ctrl+C đóng UI,
không tự dừng trainer nền. Có thể dùng thêm:

```bash
bash scripts/watch_train.sh
```

Trước khi gửi trainer, app kiểm tra import dependency và các extension ST3D
bằng đúng Python worker. Có thể đổi `ANNOLOOP_WORKER_PYTHON` và
`ANNOLOOP_RUNTIME_DEPS` khi chuyển máy. CUDA toolkit cho NVRTC có thể đổi bằng
`ANNOLOOP_CUDA_TOOLKIT`; toolkit phải có `include/cuda/std/cassert`. Run lỗi có nút **Thử lại với môi trường
đã sửa**: giữ toàn bộ run cũ, tạo output mới, không tự retry.
CUDA available chưa bảo đảm mọi GPU kernel/model chạy đủ bộ nhớ. Kết quả kiểm tra GPU phụ thuộc môi trường và dự án của bạn; import thành công chưa chứng minh full model đủ bộ nhớ. Trên GPU cũ, biên dịch kernel NVRTC
lần đầu có thể mất vài phút ở 0%; PID còn sống và log compile không phải lỗi.

## Sample, rollback và CVAT

Sau mỗi M1, M2, M3… mở **So sánh phiên bản** → **Dự đoán A/B trước và sau
train**. Chọn lượt/vòng rồi chọn **Frame sample dự đoán**. Bốn biểu đồ hiển thị
A và B trước/sau train trên cùng point cloud, cùng tỷ lệ trục. Có thể zoom,
lọc class và score; mở chi tiết để đọc box. Sample cố định giữa các vòng.

Trang tự cập nhật mỗi 5 giây. Nếu vòng bị ngắt, vẫn xem được frame đã predict;
frame chưa có được ghi rõ, không coi là model bỏ sót vật thể. Bảng hiển thị
số frame đã predict của từng model, trước/sau. Vòng chưa hoàn tất chưa có nhãn
hợp nhất để chọn hoặc bàn giao. Xem dự đoán đã lưu không chạy thêm inference/train.
App ưu tiên mở lượt gần nhất có predict sau train; các lượt cũ vẫn chọn được.
Trong **Vòng adaptation**, nút **Xem dự đoán A/B trước/sau** mở nhanh trang này.
Thanh tiến độ predict/calibration hiển thị số frame và phân biệt job còn sống
với kết quả của job đã dừng. Đọc PCD cần dependency `pypcd4` (extra `[pcd]`).

So sánh cùng frame sample, xem box BEV/bbox và chọn phiên bản nhãn.
Rollback giữ mọi checkpoint/snapshot. Proxy agreement/point support không phải
accuracy hoặc recall; hai model có thể cùng bỏ sót vật thể.

Native adaptation GPU adapter đã nối với UI, có bước kiểm tra trước khi mở nút
chạy. Final inference/CVAT còn cần nối nên nút tạo task CVAT vẫn khóa.
Snapshot adaptation tải xuống là JSON nội bộ, không phải gói import CVAT.
Shape và track sẽ bàn giao thành task riêng sau bước final predict toàn dataset.

## Kiểm tra model đã dừng train

Mở **Kiểm tra model** (hoặc nút “Kiểm tra checkpoint và xem dự đoán” trên trang
Huấn luyện nguồn), chọn checkpoint trên đĩa như SECONDNetIoU epoch 10.
Không cần metadata mới, YAML mới hoặc hoàn tất 30 epoch để kiểm tra checkpoint
trung gian. App không tự đưa checkpoint trung gian vào phiên bản model chính.

- **Đánh giá source validation**: inference toàn bộ split validation có nhãn;
  xuất TP/FP/FN, precision/recall theo class, ở ngưỡng score và 3D IoU đã chọn.
  Ghép một-một theo score, cùng class, mỗi frame. Không phải nuScenes mAP chính thức.
- **Predict sample target**: cần xác nhận đơn vị mét và trục X trước/Y trái/Z lên.
  BIN cần chọn đúng số cột. App chạy trên sample cố định và hiển thị BEV point
  cloud + box, class/score. Không đọc nhãn/gold target, không train lại.

Job chạy nền trên worker GPU riêng và dùng cùng GPU lock với trainer. Nếu GPU
bận thì báo chờ; không chạy đồng thời. Progress/log/kết quả cập nhật 5 giây.
Trạng thái completed chỉ xuất hiện sau inference hoàn tất; thất bại giữ log.


## Dự án hiện tại và vòng adaptation

Sidebar tự chọn dự án gần nhất, kèm ID ngắn để phân biệt các tên trùng. Hiện số
frame, class, vòng hoàn tất và phiên bản nhãn đang dùng. Mở **Thiết lập dự án**
để sửa tên, số vòng tối đa, chạy từng vòng/tự động và chọn nhãn bằng tay/theo
ngưỡng. Thay đổi này giữ dataset và các snapshot.

1. Mở **Vòng adaptation**, chọn checkpoint A (SECOND-IoU) và B (PV-RCNN) có
   đủ class dự án. Checkpoint trung gian có thể dùng để thử, kèm thông báo rõ.
   Chọn **Class chạy adaptation**: có thể tạm chọn `car`, `pedestrian` dù
   checkpoint có thêm `motorcycle`. Chỉ class được chọn cần calibration và
   tạo pseudo-label dương. Box dự đoán của class còn lại vào vùng ignore;
   giữ nguyên ID class/đầu ra model và source loss. Có thể đổi trước vòng đầu,
   sau khi có snapshot thì giữ phạm vi để so sánh các vòng nhất quán.
2. Xác nhận target theo mét, X trước/Y trái/Z lên trong hệ LiDAR; BIN cần đúng
   số cột. App chỉ nhận XYZ chuẩn hóa ở nhánh này, chưa tự đoán/đổi trục.
3. Chọn epoch mỗi model/vòng và LR; **Lưu thiết lập adaptation**. Cấu hình
   đang sửa chưa được lưu sẽ khóa các nút gửi job.
4. Bấm **Kiểm tra adapter GPU A → B**. Worker chạy inference source/target và
   forward/backward trên dữ liệu nguồn cùng sample target không nhãn. Đây là
   diagnostic **0 optimizer step**, không lưu weights mới hay nhãn để train.
5. Khi cả hai qua kiểm tra, bấm **Chạy vòng tiếp theo**. Worker calibration
   confidence/IoU theo class trên source validation cho A → B trước, predict
   target khi đủ mẫu, rồi hợp nhất nhãn,
   rồi train A → B trên cùng pseudo-label và source loss. Epoch/LR do bạn chọn.
   Khi job kiểm tra kết thúc, giao diện tự cập nhật trạng thái và nút chạy.
6. Xem log, source/target loss, optimizer steps ở **Theo dõi adaptation**.
   Checkpoint lưu mỗi 10 epoch và epoch cuối thành công. Sau vòng, mở **So sánh
   phiên bản** để xem sample cố định và chọn/rollback. Auto train vẫn chờ nếu
   chính sách chọn nhãn yêu cầu bạn duyệt.

**Kết thúc sau vòng hiện tại** giữ vòng đang chạy hoàn tất, không tạo vòng mới.
Muốn tiếp tục sau yêu cầu kết thúc: bấm **Cho phép chạy tiếp**, rồi bấm nút chạy
vòng. Thao tác cho phép chỉ bỏ cờ kết thúc, không tự chạy model. Nếu đã hết số
vòng, tăng số vòng tối đa ở **Thiết lập dự án** trước.
**Ngắt job ngay** dừng controller/worker và giữ checkpoint/log. Chờ worker cũ
thoát trước khi tiếp tục. Refresh trang không tự gửi job.

Nếu có lượt native hợp lệ, chọn **Lượt cần tiếp tục**, rồi **Tiếp tục vòng M…**:

- Model đã train xong dùng checkpoint hoàn tất, bỏ qua train model đó.
- Model đang dở dùng checkpoint target gần nhất, khôi phục weights và Adam rồi
  chạy số epoch còn lại. Ví dụ lưu epoch 10/30 thì tiếp tục từ epoch 11.
- Chưa có checkpoint target thì train lại riêng model đó từ checkpoint khởi tạo
  của vòng. Các iteration chưa lưu không thể khôi phục.
- Predict dùng lại frame đã lưu khi dataset/checkpoint/config khớp; chỉ chạy
  forward cho frame còn thiếu. Calibration dùng source validation đã lưu phù hợp.

Tool tạo thư mục lượt mới, giữ nguyên lượt cũ. Không yêu cầu pretrain/kiểm tra GPU
lại khi đầu vào và chứng nhận vẫn khớp. Tiếp tục tuân theo chế độ dự án:
**chạy tự động + chọn nhãn tự động** thì đi tiếp; chọn nhãn thủ công thì chờ duyệt,
chạy từng vòng thì dừng sau vòng đang khôi phục. Khôi phục ở
checkpoint epoch, không tiếp nối chính xác iteration hay thứ tự augmentation/RNG.
Nếu đổi dataset, class, checkpoint, source infos hoặc thiết lập train, lượt cũ
không được đưa vào danh sách tiếp tục.

**Chạy lại vòng M…** tạo lượt mới và chạy lại toàn bộ predict/train của vòng từ
checkpoint khởi tạo hoặc phiên bản đang chọn. Dùng nút này khi muốn bắt đầu lại.
Các lượt legacy chưa có cấu trúc attempts vẫn được xem và giữ nguyên artifacts.


Nếu chọn nhãn tự động mà vẫn phải duyệt, xem thông báo **M… không được tự động
chọn** và bảng **Kết quả chọn nhãn và ngưỡng** ở Vòng adaptation hoặc So sánh
phiên bản. Vòng đầu bị từ chối thì chưa có nhãn nào để train tiếp. Khi đã có
phiên bản nhãn, vòng mới bị từ chối sẽ giữ phiên bản đó. App không tự giảm ngưỡng.
Việc bấm **Dùng phiên bản này** có thể chọn thủ công cả một vòng đã bị từ chối;
giao diện phân biệt lựa chọn này với chấp nhận tự động.

**Bộ nhớ pseudo-label:** mở phần cùng tên ở Vòng adaptation để bật và chọn số
xác nhận/số vòng giữ khi bỏ sót. Mặc định ≥2 xác nhận, giữ 1 vòng nếu đủ điểm,
bỏ sót lần 2 thành ignore, lần 3 hết hạn; bằng chứng mâu thuẫn ignore ngay.
So sánh phiên bản có bảng lịch sử từng memory ID theo frame. Vòng cũ được đọc
để dựng lịch sử mà không sửa kết quả. Nhãn giữ lịch sử không tăng agreement;
nhãn train vẫn theo phiên bản được chọn. Xem [chi tiết](PSEUDO_LABEL_MEMORY.md).
Áp dụng cho lượt chạy mới; không thay đổi worker đã khởi chạy.

Nếu thiếu B nhưng muốn dùng A epoch 10 đã lưu: chọn/lưu A ở màn adaptation,
qua **Huấn luyện nguồn**, chọn chỉ **B** ở “Model cần học nguồn trong lượt này”
rồi bấm nút bắt đầu/thử lại. Tool không bắt buộc train A lại. Job B hoàn tất
sẽ được nhận đúng detector B, sau đó chọn checkpoint B tại màn adaptation.

Profile hiện hỗ trợ nguồn nuScenes chuẩn bị infos 1-sweep và checkpoint
SECOND-IoU/PV-RCNN XYZ tương ứng. Nhánh train mới là co-training source-assisted;
chưa có DSNorm/EMA/curriculum trong worker này. Tracking để chọn nhãn train còn
cần metadata sequence/timestamp/ego pose; UI hiện chạy adaptation không tracking.
Không đủ mẫu calibration hoặc không có pseudo-label dương đủ bằng chứng thì
controller dừng trước train. GPU check chứng minh đường chạy, không chứng minh
model dự đoán chính xác. Gold target không dùng để calibration/chọn nhãn/train.

**Lỗi “5/10 dự đoán source validation”**: model chỉ dự đoán 5 box của class đó,
không phải source thiếu nhãn người. Mỗi class đang chọn cần ít nhất 10 box để
fit đường calibration; đủ 10 cũng chưa bảo đảm chất lượng. Bảng calibration
hiển thị số dự đoán và số box có IoU ≥ 0.3 với nhãn nguồn. Nếu một checkpoint
đã biết thiếu mẫu, app chặn gửi lại cùng đầu vào. Chọn checkpoint tốt hơn, thêm
source validation độc lập, hoặc chủ động thu hẹp class trước vòng đầu. Không
lặp lại frame, hạ số mẫu tối thiểu hay lấy gold target để lấp thiếu hụt.
