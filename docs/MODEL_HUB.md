# Kho checkpoint theo class

Cập nhật: [recipe MMDetection3D](MMDET3D_ADAPTER.md) đã có code cho CenterHead/
Anchor3DHead trên một số model một tầng. Script tải/cache tự nhận recipe;
runtime và kiểm chứng dự án vẫn bắt buộc. Mô tả generic chưa có trainer bên
dưới áp dụng ngoài phạm vi recipe này. Chưa có chứng nhận GPU MMDet thật.

Ngoài kho generic, màn Class & model có [nguồn native ST3D/3DOD-LSF](NATIVE_SOURCES.md)
cho hai checkpoint source-only đã pin: SECOND-IoU và PV-RCNN, class Vehicle / XYZ.
Thiếu class mới học nguồn, không ép qua generic adapter rồi gặp chặn train.
Link ST3D chính thức cũng được giữ để tham khảo; chưa quảng cáo mọi checkpoint
trong repo là đã pin/tương thích, PointRCNN chưa được cấp native support.

Trong workflow dự án, chọn model ở **2 · Class & model** rồi tải/chuẩn bị tại
**3 · Chuẩn bị model**. Adaptation tự dùng lại cùng lựa chọn. **Kho model** nằm
trong **Công cụ & hướng dẫn**, dùng tham khảo/tải gói ngoài dự án; không bắt
người dùng chuyển qua thư viện rồi chọn lại model ở adaptation.

Bước Class & model mặc định lọc trước bằng `model_filter.py`: class checkpoint
và feature target. Thiếu backend không ẩn model; input chưa rõ vẫn chọn được
với trạng thái chờ xác minh. Nhãn source và quyền chạy được kiểm tra riêng sau
tải. Các model có bảng lý do riêng; raw predict/binding không được
coi là train support. Header tối đa16 frame rải đều, không load model/đọc gold.
`input_contract` của registry phải có evidence trỏ config đã pin. Hiện chỉ hai
gói SECOND KITTI và PointPillars nuScenes có contract feature được ghi rõ;
các gói còn lại chưa có contract thì giữ unknown, không đoán từ dataset/tên.
Mở rộng danh sách chỉ để chuẩn bị nghiên cứu; trang chuẩn bị khóa bước tiếp
với lựa chọn không phù hợp, nút tải cần xác nhận nghiên cứu riêng.

1. Chọn class cần gán; chọn dataset pretrained/kiến trúc nếu muốn.
2. Bật **Chỉ hiện model hỗ trợ đủ class** để tìm model đáp ứng taxonomy.
3. Xem nguồn, config, class, dung lượng và trạng thái adapter.
4. Bấm **Chọn model & chuẩn bị adapter**. Khi đó worker mới tải gói được chọn
   (hoặc dùng cache hợp lệ), rồi khởi tạo binding theo code adapter của framework.
   Xem/chuyển lựa chọn không tải, không tạo adapter, không khởi chạy GPU/trainer.
   Nút **Tải / tiếp tục tải gói model** cũng thử tạo kết nối sau khi tải xong;
   framework chưa hỗ trợ chỉ ghi báo cáo, không tạo binding giả. Xem
   [script kết nối model đã tải](CONNECT_MODEL.md).
5. File nằm ở `ANNOLOOP_DATA_DIR/model_cache/<model_id>/`. Lần sau tái sử dụng
   cache sau khi kiểm tra hash/config. Có thể dừng tải và giữ phần đã tải.

## Hiện có gì?

Danh mục ngày15/10/2026 có **62 checkpoint tải binary được**, từ hai nguồn chính
thức MMDetection3D và OpenPCDet; 19 họ/biến thể (đã gộp cách viết tên giữa hai
framework), 7 dataset. Có SECOND/SECOND-IoU, PointPillars, CenterPoint, PV-RCNN,
PointRCNN, Voxel R-CNN, VoxelNeXt, TransFusion, BEVFusion, Part-A2, 3DSSD,
FreeAnchor và các detector trong nhà. Bộ lọc mặc định chỉ dùng point cloud;
BEVFusion cần LiDAR+camera và không được đề xuất như model LiDAR thuần.

Nguồn upstream:

- [MMDetection3D model zoo](https://github.com/open-mmlab/mmdetection3d/blob/main/docs/en/model_zoo.md)
- [OpenPCDet model zoo](https://github.com/open-mmlab/OpenPCDet#model-zoo)

Mỗi mục có URL weights, config/base configs, revision Git cố định, class đọc tĩnh
từ config và bằng chứng kiểm tra link. Không chạy code config hoặc load pickle
chỉ để tìm class. Thông tin class là lời khai config upstream, chưa phải chứng
nhận đã strict-load head của từng weights. Không coi AP giữa hai dataset là
thước đo xếp hạng chung; chưa chấm chất lượng trên dữ liệu người dùng.

## Tạo adapter tự động khi người dùng chọn model

`adapter_factory.py` dùng recipe theo framework để tạo `adapter.json` gắn
weights/config/class và hash, rồi `load_adapter()` khởi tạo đối tượng
`FrameworkAdapter`. Không sinh hoặc thực thi Python tùy ý theo tên model.
Worker chỉ được gửi khi người dùng bấm chọn/chuẩn bị; code không tải model lúc
import hoặc khi mở trang. CLI tương đương: `python -m annoloop.model_hub setup MODEL_ID`.

Hai backend generic có code raw predict: MMDetection3D và OpenPCDet, gọi API
framework trong Python model qua subprocess. Chưa chạy forward thật với hai
backend mới. Phải có runtime tương ứng; OpenPCDet dùng
`ANNOLOOP_OPENPCDET_RUNTIME`, không lấy ngầm ST3D fork. Giữ feature thật,
kiểm tra thứ tự class/coverage weights, đổi origin box MMDetection3D về tâm
hình học và ghi hệ tọa độ. Model cần multi-sweep/camera nhưng input thiếu sẽ
bị chặn, không tự tạo intensity/time/sweep. Không dùng annotation target.

Binding tạo xong có trạng thái **prepared_unverified**, không phải chứng nhận
forward/backward. Raw predict không tự tạo `iou_score` từ classification score.
**Generic target training/calibration/signed-ignore cho các framework mới chưa
được nối**; adapter native adaptation A/B hiện tại vẫn riêng. Không quảng cáo
tạo binding là đã có full adaptation cho mọi detector.

## Tải được khác với chạy được

Các gói mới có trạng thái **adapter_required**. Chưa có gói registry nào được
chứng nhận native adaptation trong MVP. Model MMDetection3D và OpenPCDet cần
adapter/config/features tương ứng; ngay cả SECOND/PV-RCNN cùng tên cũng chưa
chắc tương thích với ST3D runtime. Tải không thay model dự án, không kích hoạt
source pretrain, không đổi nhãn cũ. Luồng GPU A/B hiện tại giữ readiness gate.

Class chỉ chuẩn hóa viết hoa và dấu phân cách. `Vehicle` không tự tách thành
car/truck/bus; `Cyclist` không tự đổi thành motorcycle; person/pedestrian cần
xác nhận định nghĩa. Model thiếu class không được giả là hỗ trợ đủ.

VRAM upstream là số liệu training nếu có, không phải bảo đảm inference chạy
trên GTX960M. Không tự tải/cài runtime mới hoặc sửa venv khi xem kho model.

## Kiểm tra download và cache

- Khi tạo danh mục: GET phần đầu weights và xác nhận header binary Torch;
  trang HTML/login và link lỗi bị loại. Đây là kiểm tra tại thời điểm ghi,
  không bảo đảm link tồn tại vĩnh viễn hoặc toàn file đã tải đủ.
- Khi tải: kiểm tra dung lượng HTTP nếu server cung cấp, prefix theo registry,
  config SHA256, checksum weights nhà phát hành nếu có; luôn tính SHA256 local.
- SHA256 local bảo vệ cache khỏi thay đổi, không phải chứng thực từ nhà phát hành
  khi upstream không công bố full checksum.
- Chỉ tạo `installed.json` sau khi toàn bộ gói hợp lệ. Giữ `.part` và lỗi khi
  mất mạng. Resume dùng Range + If-Range/ETag hoặc Last-Modified; nếu server
  không hỗ trợ thì tải lại từ đầu, không nối nhầm dữ liệu.
- Giấy phép repository và điều khoản weights/dataset cần xem ở upstream trước
  khi phân phối lại. App hiện tải từ nguồn chính thức, chưa tự mirror weights.

Artifacts: `download.json`, `partial.json`, `worker.log`, `weights.pth`,
`upstream/`, `installed.json`. Tất cả nằm dưới data đã gitignore; không đưa
weights, dự án hoặc database người dùng vào git.

## CLI và cập nhật danh mục

Từ root MVP (Python VIN_Ai ở hai cấp trên):

```bash
PYTHONPATH=src ../../.venv/bin/python -m annoloop.model_hub list --classes car pedestrian
PYTHONPATH=src ../../.venv/bin/python -m annoloop.model_hub download MODEL_ID
../../.venv/bin/python scripts/refresh_model_registry.py
```

Lệnh cuối dành cho maintainer, cần mạng; pin nguồn config và kiểm tra lại từng
link trước khi ghi `src/annoloop/catalog/models.json`. Mục bị loại nằm trong
`data/model_registry_refresh/excluded.json`. App đọc danh mục đi kèm package,
không scrape web khi rerun. Catalogue JSON được đóng trong wheel.

Đã tải đầy đủ một gói SECOND KITTI 3class (10.735.230bytes), config kế thừa và
kiểm tra cache offline. Full SHA256 local:
`05f67bdf31fbfc77ad7cce0d13c3cd30af92d3e12c13fadf2f6c230b38741ef9`.
Không chạy GPU hoặc đo chất lượng detector trong kiểm tra download này.

Đã tải thêm gói OpenPCDet PointPillars KITTI ba class (19.383.576bytes) từ
Google Drive của tác giả, cùng YAML/dataset config. SHA256 local:
`4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`.
Hai gói mẫu không phải xác nhận mọi model trong registry đã tải trọn/chạy GPU;
62link được probe binary, còn tải đầy đủ thực hiện khi người dùng chọn.
## Cập nhật luồng chuẩn bị model

Kho đề xuất theo class và hợp đồng đầu vào, tách khỏi quyền chạy adaptation.
Thiếu backend không ẩn checkpoint; input chưa rõ ghi chờ xác minh. Tác vụ nền
chỉ bắt đầu khi bấm **Tải, chuẩn bị adapter & kiểm tra model** ở bước chuẩn bị.
Tải/binding thành công vẫn chưa đủ: cần predict/backward/ignore/calibration.
Recipe generic còn thiếu phần train sẽ báo rõ và không mở adaptation.
Xem [MODEL_PREPARATION.md](MODEL_PREPARATION.md).

