# Bàn giao UI cho Astra

15/10 final delivery: bỏ warning/nút upload placeholder, nối ui/delivery_page.py
vào Xuất CVAT. A/B chốt checkpoint vòng đang chọn; ensemble chốt lượt hoàn tất.
Scope rõ adaptation+delivery/delivery/all, final inference riêng → PCD+annotation
mapping → explicit create/upload task shape qua SDK. Không tự inference/upload
khi xem trang; credential không persist, không dùng snapshot labels làm final.
Chỉ shape, chưa track/2D. Đọc FINAL_DELIVERY.md; actual GPU/CVAT chưa kiểm chứng.

15/10: bước Class & model có nhánh **Ghép checkpoint local** đọc receipt/config
từ state app, hiện cả official ST3D downloads bị chặn. Branch chọn → chuẩn bị →
chạy → sample tích hợp trong năm bước cũ, không thêm page riêng. Inference-only,
không N-model training. Xem LOCAL_ENSEMBLE.md; không tự mở quyền cho imports,
không dùng chứng nhận A/B để giả ensemble GPU đã kiểm tra. Giữ A/B lịch sử riêng.

Workflow đã được chỉnh tiếp theo phản hồi người dùng: năm bước chính Dữ liệu →
Class & model → Chuẩn bị model → Adaptation → Kết quả/CVAT. Sidebar pages cũ
chuyển vào Công cụ & hướng dẫn. `project_models.py` là nơi duy nhất chọn A/B;
`adaptation.json.model_selection` giữ intent local/hub/source, kể cả model chưa
tải. `save_model_selection` resolve cache/nguồn sang binding; không download.
Trang chuẩn bị nhận đúng intent, tải/bind chỉ khi bấm; adaptation chỉ đọc model,
không còn dropdown chọn lại. Generic train chưa có vẫn bị chặn rõ, không nhầm
thành thiếu class để bắt pretrain. Dự án/lượt cũ giữ checkpoint gốc.

Cập nhật tiếp15/10: người dùng yêu cầu chỉ viết code, không tải model bây giờ.
Nút **Chọn model & chuẩn bị adapter** gửi worker `model_hub setup` đúng model
được chọn: tải/cache → `adapter_factory.prepare_adapter` → `adapter.json`.
Xem/chuyển lựa chọn không có side effect mạng/GPU. FrameworkAdapter có code
raw predict qua framework_worker; chưa chạy forward thật. Trạng thái binding
prepared_unverified; generic target train/calibration/signed-ignore chưa nối,
không mở native readiness chỉ vì binding được tạo. Chi tiết MODEL_HUB.md.

Cập nhật15/10: sidebar **Kho model** tại `ui/model_library.py`, backend
`model_hub.py`, danh mục `src/annoloop/catalog/models.json`. Wizard có liên kết
chuyển class sang thư viện; xem `MODEL_HUB.md`. 62gói/19họ-biến thể/7dataset,
đã probe binary; download background có progress/pause/resume. Mọi model mới
vẫn adapter_required; không mở nút native adaptation hoặc bỏ kiểm tra GPU chỉ
vì tải xong weights. Page view/refresh không tự tải hay train.

Cập nhật 13/10/2026: người dùng đã yêu cầu dựng UI. Giao diện hiện có tại `ui/app.py`;
xem `USER_GUIDE.md`. Nội dung bên dưới giữ làm định hướng; chưa tạo Docker.
Làm UI Streamlit trong `ui/` riêng; `.venv` VIN_Ai đã có Streamlit 1.65.0.
Không thay backend bằng code UI hoặc dữ liệu mẫu giả. NativeDetectorAdapter
đã nối trong adaptation.py/adaptation_worker.py; nút chạy dựa trên readiness
của dự án (checkpoint, coordinates và verification), không mở vô điều kiện.

## Phong cách đã chọn

Tên AnnoLoop. Nền trắng/ngà, tím làm màu hành động chính, chữ rõ, spacing thoáng.
Sidebar: Dữ liệu → Class & mapping → Huấn luyện nguồn → Vòng adaptation →
So sánh → CVAT → Cài đặt. Shell có switch 2D/3D; 2D hiện chỉ có schema/export
contract, chưa có detector. Không cho bật ROS 3D trên 2D.

![Preview đã sửa: class từ dataset đầu vào](mockups/source_pretrain.png)

Ảnh chỉ là reference thiết kế. Các class, đường dẫn và thống kê trong ảnh là ví dụ.

## Các màn hình và backend

| Màn hình | API/backend | Hành vi |
|---|---|---|
| Dataset | `inspect_dataset`, `discover`, `ProjectStore.create` | Folder trước, đọc schema sau; class hiển thị từ dataset; không hardcode 6 class |
| Class | `compatibility.matrix`, `source_counts`, `checkpoint_metadata` | Tách source train/val counts với checkpoint đã học; mapping rộng/hẹp hiện amber; không tự chấp nhận |
| Source pretrain | `source_config`, `jobs.start_pretrain/start_source_pair` | ROS preview, epoch/LR từ profile, 2 model lần lượt; thiếu class/anchor/source thì chặn |
| Theo dõi | `jobs.status`, `read_log`, `device.probe` | PID thật, log/epoch, checkpoints; không suy live từ log cũ; không restart khi rerun UI |
| Adaptation | `RoundController`, `ProjectStore.next_round/request_finish` | Số vòng tùy chọn; manual từng vòng hoặc automatic; dừng sau vòng đang chạy |
| Review vòng | `ProjectStore.get/select`, `SelectionPolicy` | Fixed sample A/B + consensus, trước/sau; chọn mới hoặc quay lại vòng cũ; thresholds có thể cấu hình |
| CVAT | `cvat.package/export_media/upload` | Shape/track task riêng; upload explicit; credential chỉ trong memory; không overwrite review |

`RoundController` dùng NativeDetectorAdapter qua subprocess model venv. Source/
target predict và target trainer source-assisted đã có; kiểm tra GPU A → B có
0 optimizer step. Project Test (`60fdb405fa904512`) đã có A epoch 10 và B
epoch 20, đủ car/pedestrian/motorcycle; cả hai đã kiểm tra GPU thành công ngày
14/10/2026. M1 cũ bị ngắt trong predict A, chưa hoàn tất vòng adaptation.
Final delivery/CVAT chưa enable.
Streamlit gọi worker/background service, không gọi vòng train trực tiếp trong
script page. State bền ở backend, `session_state` chỉ lưu selection/view.
Không giữ thread executor làm nguồn duy nhất xác định run đang sống.

## UX cần giữ

1. Chọn dataset, đọc class, rồi mới kiểm tra checkpoint/source. Nếu target là
   point cloud không nhãn: nhập class schema hoặc lấy schema CVAT. Không chạy
   detector để khẳng định class nào hiện diện trước bước này.
2. Hai điều khiển khác nhau: **cách chạy train** và **cách chọn phiên bản nhãn**.
   Automatic train vẫn có thể dừng chờ manual review.
3. So sánh trên cùng frame mỗi vòng. Hiện class, confidence đã calibration,
   quality, point support, origin, track, policy positive/ignore/review.
   Viewer mới đọc raw prediction A/B trước/sau train, score gốc và point cloud
   BEV. Cập nhật mỗi 5 giây, không dispatch GPU. Xem được cả lượt bị ngắt;
   frame chưa có không hiển thị thành zero detections. Chỉ snapshot nhãn chung
   đã hoàn tất mới được chọn làm phiên bản. Ưu tiên lượt có predict sau train.
4. Source validation AP/precision/recall phải hiện đúng domain; proxy target
   không gắn nhãn accuracy. Thiếu metric thì hiện chưa có, không sinh số giả.
5. Phân biệt **vòng vừa train** và **vòng nhãn đang chọn**. Rollback giữ toàn bộ
   snapshot; không delete vòng mới. Giữ checkpoint 10 epoch và cuối.
6. Nếu ngắt job: giữ log và checkpoint. Cho chọn lượt hợp lệ rồi **Tiếp tục vòng
   M…** sau khi worker cũ thoát: giữ model đã hoàn tất, dùng cache predict đúng
   hash, khôi phục model/Adam tại checkpoint epoch. Nếu chưa có checkpoint target
   thì train lại riêng model còn dở; không hứa khôi phục iteration chưa lưu/RNG.
   Resume theo execution/selection mode: tự động tiếp tục nếu nhãn được chọn,
   manual selection chờ duyệt, manual execution dừng sau một vòng. Vòng đầu
   bị automatic selection từ chối thì dừng vì chưa có nhãn để train tiếp.
   Hiện lý do cùng proxy/ngưỡng ở Vòng adaptation và So sánh phiên bản; phân biệt
   manual override với automatic accept. **Chạy lại vòng M…** là thao tác
   riêng bắt đầu toàn bộ vòng trong lượt mới. Không dispatch khi refresh.
   Monitor refresh toàn app khi
   job nền kết thúc để cập nhật readiness và mở lại các nút.
   Nếu cờ kết thúc đang bật, hiện **Cho phép chạy tiếp**; chỉ bỏ cờ sau thao tác
   người dùng và khi không còn worker, không tự dispatch.
   Class adaptation có thể là subset của schema/checkpoint trước vòng đầu.
   Calibration và nhãn dương chỉ dùng subset; dự đoán ngoài subset giữ làm
   ignore, source loss vẫn theo config gốc. Không đổi head hay ID class.
   Pseudo-label memory mặc định bật cho controller mới, riêng với tracking.
   UI có số lần xác nhận và cửa sổ giữ khi bỏ sót. So sánh phiên bản đọc memory
   snapshot có hash và hiển thị lịch sử mỗi box theo frame; không dispatch GPU.
   Nhãn giữ theo lịch sử không tăng agreement. Vòng bị từ chối được ghi vào lịch
   sử nhưng không tự đổi nhãn train. Xem PSEUDO_LABEL_MEMORY.md.
7. Màn CVAT chỉ export dataset bàn giao đã có final predictions từ version
   được chọn. Hiện rõ số frame xuất; không gán holdout vào pseudo-label train.
8. Device: NVIDIA GPU, một worker; `.venv` VIN_Ai mặc định. Thiếu CUDA/runtime
   thì hiện preflight blocker. Không có fallback CPU detector.

## Chuyển Docker sau này

UI/backend/worker tách module. Dữ liệu và runtime đặt ngoài code bằng env/mount.
Không thêm Compose/Dockerfile trong phase này. Giữ API service ổn định để có thể
thay storage/queue và chạy worker thành container riêng sau này.
