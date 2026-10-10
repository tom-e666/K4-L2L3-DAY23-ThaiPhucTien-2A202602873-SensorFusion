# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Thái Phúc Tiến
- MSSV: 2A202602873
- Email: tptien.a1.c3tqcap@gmail.com
- Link repo (fork): https://github.com/tom-e666/K4-L2L3-DAY23-ThaiPhucTien-2A202602873-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): [điền sau khi commit cuối cùng]

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`:
  `compare`, frames `[0, 198]` (199 frames), segment `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed: 0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`:
  precision: `0.9701`, recall: `0.7004`, `tp: 519`, `fp: 16`, `fn: 222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  rmse: `0.1503 m`, matches: `502`, sum_sq_err: `11.3369 m²`, ghost_track_frames: `0`, missed_gt_frames: `239`, mean_confirmed_tracks: `2.5226`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  rmse: `0.1358 m`, matches: `502`, sum_sq_err: `9.2642 m²`, ghost_track_frames: `0`, missed_gt_frames: `239`, mean_confirmed_tracks: `2.5226`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  1. Về độ chính xác ước lượng (RMSE 3D): Mode `fused` đạt RMSE 3D là `0.1358 m`, cải thiện `0.0145 m` (~9.6% sai số) so với mode `lidar` (`0.1503 m`), đồng thời tổng bình phương sai số vị trí `sum_sq_err` giảm từ `11.3369 m²` xuống `9.2642 m²`. Cả hai mode đều có RMSE ≤ 0.45 m (đạt hệ số chất lượng RMSE tối đa = 1.0 theo Rubric). Điều này chứng minh phép đo pixel 2D từ camera sau khi chiếu và cập nhật qua EKF update đã bổ sung ràng buộc hình học góc nhìn giúp tinh chỉnh ước lượng tâm xe 3D chính xác hơn.
  2. Về số lượng ghép và độ tin cậy vòng đời (matches, ghost, miss):
     - Cả hai mode đều có số cặp ghép chính xác bằng nhau (`matches = 502`), không có bất kỳ track ảo nào (`ghost_track_frames = 0`), số đối tượng bị miss bằng nhau (`missed_gt_frames = 239`), và số track xác nhận trung bình mỗi frame bằng nhau (`mean_confirmed_tracks = 2.5226`).
     - Độ chính xác track: `precision_track = matches / (matches + ghost) = 502 / (502 + 0) = 1.0` (100% > ngưỡng 75% của Rubric).
     - Độ bao phủ track: `coverage = matches / det_tp = 502 / 519 = 0.9672` (96.72% > ngưỡng 70% của Rubric).
     - Kết quả này thể hiện đúng tính chất thiết kế `track-then-fuse`: LiDAR là cảm biến chủ đạo quyết định vòng đời track (score, init, delete), còn Camera chỉ đóng vai trò cập nhật trạng thái EKF mà không gây nhiễu, không tạo ghost track và không làm mất đối tượng đã theo dõi.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - Vectơ đo $z$:
     - LiDAR: $z \in \mathbb{R}^3$ đo trực tiếp tọa độ không gian 3D $[x, y, z]^T$ của xe trong hệ tọa độ xe chủ. Mô hình đo tuyến tính $z = H x$ với ma trận $H \in \mathbb{R}^{3 \times 6}$ chọn 3 phần tử vị trí đầu tiên.
     - Camera: $z \in \mathbb{R}^2$ là tọa độ điểm ảnh pixel $[u, v]^T$ trên ảnh 2D camera trước (`FRONT`). Mô hình đo phi tuyến $h(x) = \text{project}(x, K, T_{ext})$ chuyển đổi tọa độ xe 3D sang tọa độ camera rồi chiếu pinhole thành tọa độ pixel 2D; do đó EKF phải sử dụng ma trận Jacobian phi tuyến $H = \frac{\partial h}{\partial x} \in \mathbb{R}^{2 \times 6}$ (`compute_camera_jacobian_numerical`).
   - Ma trận hiệp phương sai nhiễu đo $R$:
     - LiDAR: $R \in \mathbb{R}^{3 \times 3}$, đơn vị là $\text{m}^2$, đại diện cho độ bất định sai số khoảng cách 3D thực tế theo các trục tọa độ.
     - Camera: $R \in \mathbb{R}^{2 \times 2}$, đơn vị là $\text{pixel}^2$, đại diện cho phương sai sai số vị trí điểm ảnh 2D trên mặt phẳng ảnh (`camera_measurement_covariance` trong `camera_fusion.py`).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Gating loại bỏ sớm các đo lường ngoại lai (clutter, false alarms, đo lường quá xa) không khả thi về mặt xác suất trước khi thực hiện thuật toán gán Greedy/Hungarian, giúp giảm không gian tìm kiếm và ngăn việc gán sai đo lường rác vào track.
   - Khoảng cách Mahalanobis $d_M = \sqrt{(z - \hat{z})^T S^{-1} (z - \hat{z})}$ (với $S = H P^- H^T + R$) chuẩn hóa độ lệch theo ma trận hiệp phương sai dự báo $S$. Khác với khoảng cách Euclid thuần túy (giả định không gian đẳng hướng), khoảng cách Mahalanobis co giãn theo ellipse độ bất định của track: khi track có độ bất định $P$ lớn theo một phương (ví dụ phương di chuyển hoặc chiều sâu), một khoảng cách Euclid lớn vẫn có thể nằm trong ngưỡng tin cậy thống kê của phân phối $\chi^2$; ngược lại nếu độ bất định rất nhỏ thì sai số Euclid nhỏ cũng sẽ bị loại nếu vượt ngưỡng kiểm định $\chi^2$.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline thuộc loại **track-then-fuse**: tracker duy trì một danh sách đối tượng trạng thái 6D duy nhất. Ở mỗi frame, tracker thực hiện predict trạng thái một lần cho mọi track, sau đó lần lượt cập nhật EKF nối tiếp theo từng cảm biến (AssocL: gán và cập nhật LiDAR trước, sau đó AssocC: gán và cập nhật Camera sau trên cùng danh sách track). Pipeline không hòa trộn dữ liệu thô (point cloud và ảnh) hay gộp detection 2D+3D trước khi tracking (fuse-then-track).
   - Minh chứng từ log `grade_run.log`: trong cùng frame $k$, cả hai chế độ `lidar` và `fused` đều có các trường thống kê `valid_gt = 741`, `confirmed = 502`, `matches = 502`, `ghosts = 0`, `misses = 239` giống hệt nhau; chỉ có tổng bình phương sai số `sum_sq_err` ở mode `fused` giảm từ `11.3369` về `9.2642 m²`. Điều này chứng tỏ Camera không sinh thêm hay can thiệp vào số lượng track mà chỉ thực hiện EKF measurement update để tinh chỉnh vị trí các track do LiDAR khởi tạo và duy trì.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Khi camera bị lệch calibration (extrinsic $T_{ext}$ hoặc intrinsic $K$ bị lệch góc xoay hoặc tịnh tiến), mô hình chiếu $h(x)$ sẽ dự đoán vị trí pixel bị lệch có hệ thống (systematic bias) so với vị trí quan sát thực tế $z$.
   - Triệu chứng trên Innovation $\gamma = z - h(x)$:
     - Innovation $\gamma$ sẽ không còn có kỳ vọng bằng 0 ($\mathbb{E}[\gamma] \neq 0$) mà xuất hiện độ lệch cố định (non-zero mean bias) lệch về một phía trên mặt phẳng ảnh theo trục $u$ hoặc $v$.
     - Khoảng cách Mahalanobis tăng vọt khiến các phép đo camera bị gating loại bỏ (không gán được vào track).
     - Nếu vẫn lọt qua gate, EKF Kalman update sẽ kéo trạng thái 3D của track lệch khỏi vị trí thực theo hướng sai số calibration, làm tăng RMSE đột biến và có thể dẫn đến phân kỳ track (divergence).

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Khi frame rỗng (`meas_list == []`), toàn bộ tracks hiện có đều trở thành `unassigned_tracks`. Hàm `manage_tracks(unassigned_tracks, unassigned_meas, sensor)` vẫn bắt buộc phải được gọi để cập nhật vòng đời track (missed frames). Tham số `sensor` tường minh giúp bộ quản lý track phân biệt: nếu là `lidar`, track bị miss trong tầm quan sát sẽ bị trừ điểm score (`score -= 1/window`) và có thể bị xóa; nếu là `camera`, frame không có đo lường (hoặc track nằm ngoài FOV camera) tuyệt đối không được trừ score hay xóa track.
   - LiDAR quyết định score/init/delete: LiDAR cung cấp đầy đủ thông tin vị trí 3D $[x, y, z]$ với trường quan sát 360°, đủ bậc tự do để khởi tạo ngay trạng thái vị trí 3D ban đầu của xe với ma trận hiệp phương sai xác định.
   - Camera chỉ EKF update: Camera đơn chỉ cung cấp thông tin góc nhìn 2D $[u, v]$ trên mặt phẳng ảnh và thiếu hoàn toàn thông tin khoảng cách/độ sâu dọc tia chiếu, do đó không thể khởi tạo trạng thái 3D tin cậy nếu thiếu LiDAR. Ngoài ra, camera chỉ bao quát góc nhìn hẹp phía trước (`FRONT`); nếu để camera trừ điểm score hay xóa track thì các xe nằm ở hai bên hông hoặc phía sau (vốn vẫn được LiDAR theo dõi liên tục) sẽ bị xóa nhầm. Vì vậy, Camera chỉ đóng vai trò hỗ trợ tinh chỉnh ước lượng trạng thái EKF khi xe xuất hiện trong FOV.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Xác nhận (Confirmed)**: Track mới được khởi tạo ở trạng thái `initialized` với điểm ban đầu `score = 1/window` (mặc định window = 5 nên score ban đầu = 0.2). Mỗi khi có LiDAR hit gán thành công, điểm số tăng `score = min(1.0, score + 1/window)`. Khi `score > confirmed_threshold` (mặc định 0.6, tương đương 3 hits liên tiếp) thì chuyển thành `confirmed`.
   - **Giữ confirmed sau miss**: Khi bị miss trong tầm quan sát, điểm số bị trừ `score = score - 1/window` (trừ 0.2). Một miss đơn lẻ làm điểm số giảm nhưng vẫn lớn hơn hoặc bằng `delete_threshold` (0.4), do đó trạng thái vẫn được duy trì là `confirmed` mà không bị hạ cấp về `initialized`.
   - **Xóa track (Delete)**: Track bị xóa khi thỏa mãn một trong các điều kiện:
     1. Phương sai vị trí quá lớn: phần tử hiệp phương sai $P_{xx} > \text{max\_P}$ hoặc $P_{yy} > \text{max\_P}$ (với $\text{max\_P} = 4.0\text{ m}^2$).
     2. Điểm số suy giảm dưới ngưỡng: đối với track đã `confirmed`, bị xóa khi `score < delete_threshold` (0.4, tương đương 2-3 misses liên tiếp); đối với track chưa confirmed (`initialized`), bị xóa ngay khi `score <= 0.0` (chỉ cần 1 miss ngay sau khi vừa tạo).

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI Assistant
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ phân tích công thức toán học EKF, ma trận hiệp phương sai camera, giải thích luồng track-then-fuse và debug các trường hợp edge-case trong pytest test_track_management.py.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Tự chạy kiểm thử toàn diện `pytest student/tests` (đạt 128/128 passed), đối chiếu công thức với slide bài giảng và tài liệu kỹ thuật `docs/HUONG_DAN_KY_THUAT.md`, chạy thực tế segment Waymo với script `fusion-run-lab --fusion compare --seed 0` và kiểm tra tính nhất quán giữa `metrics.json` và `grade_run.log` bằng công cụ `tools/check_submission.py`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
