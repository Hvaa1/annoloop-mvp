# Adapter MMDetection3D: chuẩn bị một lần, dùng lại theo recipe

Đã có code cho predict source/target, source calibration, backward, train target
kèm source loss, ignore vùng bất đồng, checkpoint mỗi10 epoch + cuối và resume
optimizer. Code này chưa được kiểm chứng với model MMDetection3D thật trên GPU;
app chỉ mở adaptation sau kiểm chứng thực tế theo dự án.

### Dựng model cho adaptation

`mmdet_model.build_model` dựng model qua registry với `model.train_cfg` gốc
trước khi cài ignore hoặc gọi loss. Không dùng `mmdet3d.apis.init_model` cho
worker adaptation: helper inference đó đặt `train_cfg=None`, khiến CenterHead
thiếu cấu hình loss và Anchor3DHead không dựng đầy đủ assigner/sampler.
Giữ cấu hình test để predict; chuyển SyncBN sang BN cho local như framework.
Thiếu train_cfg bị báo lỗi rõ ràng, không bù bằng dict rỗng hoặc giá trị tự đoán.
Constructor đọc bản sao model config; không sửa file config/checkpoint nguồn.

CPU có thể kiểm tra dựng model, nạp weights, taxonomy và cài handler ignore.
Việc đó chưa thay thế chứng nhận predict/backward/ignore trên GPU thật.

### Nạp checkpoint spconv1 / spconv2

Worker giữ `_metadata` của state_dict khi bỏ prefix `module.` và để hook
theo phiên bản của MMDetection3D chuyển layout sparse kernel. Không so shape
trước hook: checkpoint spconv1 có `(D,H,W,in,out)` còn spconv2 dùng
`(out,D,H,W,in)`. Checkpoint mới sau train giữ version metadata để không bị
chuyển layout lần thứ hai khi resume. Đây là chuyển layout lossless, không
khởi tạo lại head, không đổi số kênh hoặc tự đoán permutation cho layer khác.

`mmdet_weights.load_weights` vẫn từ chối thiếu/thừa key, tensor không khớp
sau framework load, key trùng sau bỏ prefix và NaN/Inf. Chỉ thiếu counter
`num_batches_tracked` của BatchNorm legacy được phép. Không sửa file weights
gốc. Sửa lỗi load không cấp chứng nhận GPU: cần chạy lại bước kiểm chứng
predict/backward/ignore/calibration trên app.

Tham chiếu: [hook chính thức MMDetection3D1.4](https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/mmdet3d/models/layers/spconv/overwrite_spconv/write_spconv2.py).

## Cách dùng

UI/controller vẫn dùng venv VIN_Ai. Model cần bộ thư viện MMDetection3D tương
thích; script sau chuẩn bị môi trường riêng **một lần**, không sửa venv UI:

```bash
bash buildphase/annoloop_mvp/scripts/setup_mmdet3d.sh
```

Script pin Torch2.0.1/CUDA11.8, torchvision0.15.2, MMCV2.1, MMDetection3.3,
MMDetection3D1.4. Chưa chạy script cài đặt này trong đợt viết code. Nếu máy
khác không có wheel/driver phù hợp, lỗi cài đặt phải được xử lý trước; không
được coi môi trường sẵn sàng chỉ vì checkpoint đã tải. Có thể chỉ định runtime
đã có bằng `ANNOLOOP_MMDET3D_PYTHON=/path/to/venv/bin/python`.

Sau đó chọn dataset/class/model trong app như thường. Bấm **Tải, chuẩn bị
adapter & kiểm tra model**: tải/cache → đọc config → nối recipe → kiểm chứng
GPU → predict toàn source validation → calibration. Không yêu cầu người dùng
tạo JSON/YAML. Checkpoint khác cùng recipe được nối lại bằng code có sẵn.

Ở form chuẩn bị có lựa chọn **Đầu vào MMDetection3D**:

| Chế độ | Cách xử lý |
|---|---|
| Giữ yêu cầu đầu vào gốc (mặc định) | Config cần sweep/time mà dataset không có sẽ bị chặn |
| Mỗi file là scan hiện tại | Người dùng xác nhận mỗi cloud là một scan tham chiếu; chỉ dùng scan đó, suy ra elapsed time Δt=0 |

Chế độ thứ hai giữ nguyên số kênh encoder/weights. Đây là Δt tương đối theo
scan tham chiếu, không tạo timestamp đo mới hay sweep giả. XYZ/intensity vẫn
phải có thật; dữ liệu đã có time khác0 bị từ chối ở chế độ này. Source và
target đều đi cùng chế độ khi calibration/predict/train. Thiếu lịch sử có thể
làm chất lượng giảm; không hứa tương đương kết quả multi-sweep của checkpoint.
Chế độ được lưu theo dự án, gắn vào chứng nhận GPU, checkpoint/resume; không
tự bật và không đổi sau khi đã có vòng hoàn tất.

Nút **Lưu chế độ đầu vào** chỉ lưu cấu hình, không tải, không chạy GPU/train.
Banner **Chế độ đã lưu** cho biết giá trị đang có hiệu lực. Thay radio trong
form nhưng chưa bấm Lưu/kiểm tra chưa đổi cấu hình. API chuẩn bị không truyền
mode sẽ kế thừa giá trị đã lưu, không tự trả về strict.

CLI tương đương cho bước tải/binding:

```bash
bash buildphase/annoloop_mvp/scripts/prepare_model.sh MODEL_ID
# Chỉ đọc/nối lại cache, không tải và không chạy GPU:
bash buildphase/annoloop_mvp/scripts/prepare_model.sh MODEL_ID --offline
# Kiểm tra import trong runtime model, không chạy train/GPU kernel:
bash buildphase/annoloop_mvp/scripts/prepare_model.sh MODEL_ID --offline --runtime-check
```

Exit code2 nghĩa recipe/input còn trở ngại; file đã tải vẫn được giữ. CLI không
tự chọn A/B, không ghi chứng nhận và không tự chạy adaptation của dự án.

## Phạm vi recipe

| Thành phần | Hỗ trợ trong code |
|---|---|
| Framework | MMDetection3D1.4.x / MMEngine, point cloud LiDAR |
| Head | `CenterHead`, `Anchor3DHead` |
| Model một tầng | CenterPoint, MVXFasterRCNN, DynamicMVXFasterRCNN, VoxelNet, DynamicVoxelNet; không nhánh ROI/camera |
| Đầu vào adaptation | Một frame XYZ/XYZI đúng feature; hoặc current-scan mode suy ra Δt=0 sau xác nhận rõ ràng; không tự thêm intensity |
| Source | nuScenes infos kiểu OpenPCDet đã được app chuẩn bị, có source train và validation riêng |
| Huấn luyện | Batch1, AdamW, source loss + target pseudo-label; giữ BN statistics, tắt loss velocity khi không có nhãn vận tốc |
| Ignore | Anchor negative weights hoặc heatmap negative weights; giữ positive và không train ignore thành background |
| Khôi phục | Model + AdamW tại checkpoint epoch đã lưu, không khôi phục iteration/RNG chưa lưu |
| Bàn giao | Adapter factory được nối vào final A/B inference và exporter hiện có |

Tên họ model trong kho không đủ để chọn recipe. Script đọc cấu hình đã pin,
gồm các base config, mà không thực thi Python khi chuẩn bị. Config quan trọng
không đọc được sẽ báo lỗi. Hash toàn bộ base config được kiểm lại trước chạy.
Model/head ngoài bảng, camera/ROI vẫn bị chặn rõ. Chế độ strict giữ chốt
multi-sweep; current-scan mode không phải hỗ trợ ghép sweep có ego pose. Đây không
phải adapter train phổ quát cho mọi model MMDetection3D.

## Chất lượng box khi model không có IoU head

Không sao chép confidence thành IoU. Với recipe MMDet:

1. Predict cloud gốc và cloud quay0.05 rad.
2. Đưa box quay về hệ gốc, ghép một-một theo class.
3. Dùng 3D IoU giữa hai box làm số đo ổn định hình học; box không ghép được có0.
4. Fit calibration chất lượng trên source validation có nhãn, tối thiểu10 box/class.

Trường `iou_score` được giữ để dùng chung interface nhưng có metadata
`quality_semantics=rotation_consistency_iou`, `quality_is_native_iou_head=false`.
Đây là proxy chất lượng đã calibration, **không phải IoU head hoặc bằng chứng
accuracy trên target**. Nó tốn hai forward mỗi frame và vẫn có thể ổn định mà
dự đoán sai. Không dùng gold target để fit hoặc chọn nhãn.

## Hai checkpoint test3 hiện tại

- CenterPoint đã tải: head `CenterHead`, input XYZ/intensity/time,10 sweep gồm current.
- Gói RegNet400MF đã tải: thực tế `Anchor3DHead`, không phải FreeAnchor head;
  đã sửa tên architecture/family của đúng gói này trong kho. Input config có
  elapsed time và10 sweep lịch sử cộng current.

Cả hai có code head tương ứng nhưng KITTI đơn frame XYZI không đáp ứng yêu cầu
multi-sweep gốc trong chế độ strict. Để thử trên KITTI, người dùng có thể bật
**Mỗi file là scan hiện tại**, xác nhận mỗi file là scan riêng, rồi kiểm chứng
runtime/forward/backward/ignore/calibration. PointPillars chọn XYZ+elapsed time
đúng config; CenterPoint chọn XYZ/intensity+elapsed time. Không đổi số kênh
encoder, không tạo lịch sử. Đây là chế độ suy giảm đầu vào cần đánh giá chất
lượng riêng, chưa được chạy thực nghiệm GPU trong đợt viết code này.

Kiểm tra lại binding bằng nút chung sau khi cập nhật code; cache hợp lệ không
phải tải lại weights. Snapshot, checkpoint và lựa chọn cũ không bị tự sửa.

API tham chiếu: [MMDetection3D CenterHead](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/centerpoint_head.py),
[Anchor training interface](https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/dense_heads/train_mixins.py).
