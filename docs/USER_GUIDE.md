# AnnoLoop — hướng dẫn dùng nhanh

Recipe mới: [MMDetection3D và script chuẩn bị](MMDET3D_ADAPTER.md). Runtime
model chuẩn bị một lần; chọn checkpoint cùng recipe không phải viết JSON/YAML.
Nếu mỗi file target là một scan riêng, có thể chọn current-scan mode ở bước2/3
để dùng elapsed time0 thay cho lịch sử; đây là lựa chọn explicit có thể giảm
chất lượng, không tạo sweep giả. Chưa có kiểm chứng model MMDet thật trên GPU.

## Mở app

```bash
bash buildphase/annoloop_mvp/scripts/start_ui.sh
```

Mở **http://localhost:8502**. UI dùng venv VIN_Ai. Worker GPU dùng venv riêng tương thích với model; trên
máy này là `.venv-centerpoint` với Torch 2.0/CUDA 11.8 và các extension đã build. Không chạy `app.py`
trực tiếp bằng python. Đổi cổng bằng `ANNOLOOP_PORT=8503` nếu cần.

## Một workflow, chọn model một lần

Với **nuScenes source → KITTI target**, chọn folder source nuScenes đã chuẩn bị
và `datasets/kitti_mini` cho target. Bản mini có `pointcloud_format.json`, tool tự
đọc **x, y, z, intensity (4 cột)** và đối chiếu SHA256/số điểm. File format này
chỉ có metadata point cloud; nhãn `label_2`, calibration và `subset_manifest.json`
không được mở để chọn class/pseudo-label target. Không cần tạo thêm JSON/YAML.

Trong dự án test3 cũ, metadata format được đọc từ folder media mà không sửa
snapshot dự án. Dự án mới lưu cả feature_names và hash metadata vào inventory.
Chọn sai số cột sẽ bị chặn khi kiểm tra đầu vào; chứng nhận GPU mới gắn với
metadata format. Source và các ngưỡng class/adapter/calibration vẫn giữ nguyên.

Sidebar có 6 bước: **Dữ liệu → Class & model → Chuẩn bị model → Adaptation →
Kết quả & so sánh → Xuất CVAT**. Tên, ID dự án và phiên bản đang dùng hiện trên
trang để kiểm tra trước khi thao tác.

Ở **Kết quả & so sánh**, có thể xem gói final đã lưu, lọc class và chọn box được
giữ/ignore. Box tím là nhãn được giữ; box vàng nét đứt là ignore, kèm lý do và
score khi rê chuột. Đây là bộ lọc xem, không thay đổi nhãn train hay nhãn xuất
CVAT. So sánh trước/sau, chọn phiên bản và quay lại vòng cũ vẫn ở trang này.
Frame chưa predict được ghi rõ; số box dự đoán không phải độ chính xác.

Lần mở gói CVAT đầu tiên cần kiểm tra hash media. Các tương tác tiếp theo dùng
cache khi file không thay đổi; bấm upload vẫn kiểm tra đầy đủ. Không có model
nào tự chạy khi mở trang hoặc đổi bộ lọc.

Ngoài A/B, bước **2 · Class & model → Ghép checkpoint local** cho chọn nhiều
checkpoint hiện có, rồi chạy inference tuần tự ở bước 4 và xem sample ở bước 5.
Bảng hiện cả bốn checkpoint ST3D mới tải cùng lý do chưa tương thích. Luồng ghép
chưa train nhiều detector qua các vòng; xem [hướng dẫn ghép local](LOCAL_ENSEMBLE.md).

1. Bấm **＋ Tạo dự án mới** ở sidebar. Form mới sẽ tách khỏi dự án đang xem;
   dự án cũ và các lượt chạy được giữ nguyên. Nhập **Tên dự án mới**.
   Bấm **📂 Chọn** để duyệt folder source và target,
   hoặc dán đường dẫn nếu đã biết. Folder nằm trên máy chạy AnnoLoop.
2. Bấm **Đọc hai thư mục**. Tool đọc nhãn source, train/validation,
   media target và class schema nếu target đã có.
3. Chọn **nhãn cần gán**. Target chưa có schema vẫn dùng được: chọn từ class
   nguồn hoặc nhập tên class cần có. Đây là yêu cầu của bạn, không phải nhãn gold.
4. Bấm **Lưu dữ liệu & chọn model →**. Tool tự tạo schema/inventory trong dự án,
   không ghi đè dataset gốc. Ở bước **2 · Class & model**, chọn A/B từ checkpoint
   local hoặc danh mục trực tuyến rồi bấm **Xác nhận model & tiếp tục →**.
5. Bước **3 · Chuẩn bị model** dùng đúng lựa chọn đó: local hợp lệ thì dùng lại;
   model chưa tải chỉ tải khi bạn bấm nút; model thiếu class mới học nguồn khi
   source đủ nhãn. Model thiếu adapter train được báo riêng, không ép pretrain.
6. Bước **4 · Adaptation** chỉ hiện model đã chọn, xác nhận XYZ/mét, kiểm tra GPU
   và cấu hình các vòng. Không chọn model lại ở trang này.
7. Bước **5 · Kết quả & CVAT** xem sample/so vòng, chọn phiên bản và chuyển sang
   **Xuất CVAT**. Chọn phạm vi, bấm final predict để tạo media PCD và nhãn shape;
   sau đó nhập thông tin CVAT local và bấm tạo task. Không tự upload khi final
   predict hoàn tất. Xem [hướng dẫn bàn giao](FINAL_DELIVERY.md).

Sidebar mặc định chỉ có năm bước trên; **Công cụ & hướng dẫn** chứa Kho model,
kiểm tra model và các trang kỹ thuật. Model A/B là vai trò, không cố định tên
kiến trúc. Lựa chọn được lưu chung trong `adaptation.json` và dùng xuyên suốt,
kể cả khi bạn đóng/mở lại app. Dự án đã có vòng hoàn tất giữ model gốc để bảo
toàn lịch sử; đổi model gốc cần dự án mới.

Khi chưa lưu dự án mới, sidebar hiện **＋ Dự án mới (chưa lưu)** và khóa các
bước cần dự án. Bạn có thể chọn lại dự án cũ trong **Dự án hiện tại** để quay
về tổng quan. Bước dữ liệu không tự chọn checkpoint local; bình thường hãy
để **Chưa chỉ định** và chọn model ở bước 2 sau khi lưu.

Tùy chọn nâng cao nằm trong expander: số vòng, thủ công/tự động,
chọn phiên bản bằng tay/theo ngưỡng và epoch/ROS khi cần học class mới.

## Khi nào cần train nguồn?

### Lọc model trước khi tải

Ở **Class & model**, bộ lọc **Chỉ hiện model khớp class và đầu vào** mặc định bật.
Tool kiểm tra class checkpoint và feature đầu vào; thiếu backend không ẩn model.
Nhãn source và backend được kiểm tra riêng trước khi chạy. Tool đọc tối đa 16 frame rải
đều trên target, hiển thị số mẫu đã đọc; không đọc nhãn target hay chạy detector.

Ví dụ: PCD có `x, y, z, rgb` dùng được cho model XYZ; `rgb` không thay được cho
`intensity`. Model cần nhiều sweep bị loại khi chưa có manifest được adapter hỗ
trợ. BIN/NPY nhiều cột mà chưa khai báo ý nghĩa cột sẽ hiện chưa xác minh.

Mở **model còn bước chuẩn bị / kiểm chứng · xem lý do** để biết bước còn thiếu.
Đầu vào chưa có bằng chứng hiện **chờ xác minh**, vẫn có thể chọn để chuẩn bị;
không tự đoán theo tên model và không coi là đã tương thích.
Hiện contract đầu vào registry mới được ghi rõ cho hai gói SECOND KITTI và
PointPillars nuScenes đã kiểm tra; các gói khác vẫn có thể chưa xác minh feature.
Generic adapter chưa đầy đủ vẫn được chọn/tải nếu khớp class và đầu vào;
train/calibration/ignore còn thiếu sẽ chặn adaptation sau bước chuẩn bị.

Bỏ bộ lọc chỉ để xem/chọn gói chuẩn bị nghiên cứu; không mở quyền chạy adaptation.
Ở trang chuẩn bị, lựa chọn không phù hợp bị khóa bước tiếp và chỉ hiện nút tải
khi bạn xác nhận **Chỉ tải để nghiên cứu, không dùng chạy adaptation**. Lựa chọn
cũ được giữ để sửa, không tự thay checkpoint hoặc xóa dữ liệu. Qua bộ lọc mới
chỉ là ứng viên: vẫn cần kiểm tra GPU, axes/đơn vị và calibration trên dự án.

### Chưa có checkpoint trên máy

Chọn trực tiếp model chưa tải ở bước **Class & model**, rồi tải ở **Chuẩn bị model**.
**Kho model** trong Công cụ là thư viện tham khảo, không phải nơi chọn model của dự án.
Không giới hạn danh mục ở hai detector. Các gói có nguồn tải và class từ config,
nhưng cần adapter tương thích mới dùng được trong adaptation. Xem
[hướng dẫn Kho model](MODEL_HUB.md). Tải không khởi chạy trainer hoặc đổi model dự án.

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
Source KITTI/Waymo/format tự tạo và tự chuẩn bị infos từ nuScenes raw còn cần adapter.

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
trên source validation. Thiếu geometry thì báo blocker, không tự bịa kích thước. Runtime ST3D mặc định lấy từ workspace; quản trị viên
có thể đổi `ANNOLOOP_ST3D_RUNTIME` khi chuyển máy.

Sau khi job hoàn tất với receipt/checkpoint hợp lệ, app nhận checkpoint mới và
chuyển model sang trạng thái dùng trực tiếp. Không gửi job lại khi refresh.
Checkpoint mỗi **10 epoch + epoch cuối**. Log, PID và checkpoint cập nhật 5 giây.
**Dừng trainer** giữ artifacts; chưa có resume tự động. Ctrl+C đóng UI,
không tự dừng trainer nền. Có thể dùng thêm:

```bash
bash buildphase/lidar_qa_demo/watch_train.sh
```

Trước khi gửi trainer, app kiểm tra import dependency và các extension ST3D
bằng đúng Python worker. Có thể đổi `ANNOLOOP_WORKER_PYTHON` và
`ANNOLOOP_RUNTIME_DEPS` khi chuyển máy. CUDA toolkit cho NVRTC có thể đổi bằng
`ANNOLOOP_CUDA_TOOLKIT`; toolkit phải có `include/cuda/std/cassert`. Run lỗi có nút **Thử lại với môi trường
đã sửa**: giữ toàn bộ run cũ, tạo output mới, không tự retry.
CUDA available chưa bảo đảm mọi GPU kernel/model chạy đủ bộ nhớ. Đã kiểm tra forward/backward nhỏ với spconv và cuDNN trên GPU; chưa chứng minh
full source model đủ bộ nhớ/chạy hết epoch. Trên GPU cũ, biên dịch kernel NVRTC
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

## Tài nguyên trên máy hiện tại

Source: `buildphase/lidar_qa_demo/resources/nuscenes`

Target Day14: `Day14/K4-L2L3-Day14-Robotaxi-Tracking-Fusion-Student/outputs/Day14-J01-with-media`

Checkpoint Waymo hiện có class Vehicle: chọn car/pedestrian sẽ hiển thị cần học.
Muốn bỏ pretrain, chọn checkpoint đã hỗ trợ chính xác car/pedestrian.
Không sửa schema/weights của dự án đang chạy để vượt kiểm tra.

Dữ liệu được lưu ngoài source code, không push dataset, weights, log hoặc mật khẩu.
Docker chưa tạo. UI/backend tách module; `ANNOLOOP_DATA_DIR` đổi nơi lưu project.

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
4. Luồng mới: bấm **Tải, chuẩn bị adapter & kiểm tra model** ở Chuẩn bị model.
   Có thể kiểm tra lại bằng **Kiểm tra adapter GPU A → B**. Worker chạy inference source/target và
   forward/backward trên dữ liệu nguồn cùng sample target không nhãn. Đây là
   diagnostic **0 optimizer step**, không lưu weights mới hay nhãn để train.
   Dự án dùng giao thức mới còn kiểm tra gradient/ignore thực tế và fit
   calibration trên toàn bộ source validation trước khi mở adaptation.
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

Ví dụ lượt `fb09f628…`: A/B train xong 3 epoch, A predict198/B12. Người dùng đã
tiếp tục lượt này để predict186 frame B còn thiếu và hoàn tất M1, giữ checkpoints.
Lượt `ebacbae9…` chỉ hoàn tất A; B chưa có checkpoint target nên nếu khôi phục
lượt đó thì phải train lại riêng B.

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
## Chọn model và kiểm chứng sau tải

Ở **Class & model**, chọn checkpoint theo class và đầu vào; thiếu adapter không
còn làm checkpoint biến mất. Input chưa rõ có nhãn **chờ xác minh**.
Ở **Chuẩn bị model**, bấm **Tải, chuẩn bị adapter & kiểm tra model** để chạy
tác vụ nền. Bảng trạng thái cho biết bước nào đạt, chờ, lỗi hoặc thiếu code.
Chỉ khi predict, backward, ignore và calibration đạt mới mở adaptation.
Xem [hướng dẫn và sơ đồ chi tiết](MODEL_PREPARATION.md).

