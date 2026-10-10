"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params
import numpy as np
from scipy.stats import chi2

# vi: from fusion_lab.workspace_support import get_tracking_params
# vi: from fusion_lab.workspace_loader import load_workspace_module
# vi: kalman = load_workspace_module("kalman")  # không dùng `import kalman`


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    # vi: TODO Part F — H = meas.sensor.get_H(track.x);
    # vi: gamma = kalman.innovation(...); S = kalman.innovation_covariance(...);
    # vi: return gamma.T @ inv(S) @ gamma (float scalar).
    kalman = load_workspace_module("kalman")
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    d_sq = gamma.T @ np.linalg.inv(S) @ gamma
    return float(np.asarray(d_sq).ravel()[0])


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    # vi: TODO Part F — ngưỡng chi2.ppf(gating_threshold, sensor.dim_meas) từ params.
    params = get_tracking_params()
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return bool(mhd_sq <= threshold)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    # vi: TODO Part F — khởi tạo toàn inf; kiểm tra meas.sensor.in_fov(track.x)
    # vi: trước MHD (camera sau lưng/độ sâu 0 không được chiếu); rồi kiểm tra chi2.
    n_tracks = len(track_list)
    n_meas = len(meas_list)
    cost = np.full((n_tracks, n_meas), np.inf, dtype=float)

    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            if not meas.sensor.in_fov(track.x):
                continue
            d_sq = mahalanobis_distance(track, meas)
            if chi2_gate(d_sq, meas.sensor):
                cost[i, j] = d_sq

    return np.asmatrix(cost)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    # vi: TODO Part F — chỉ lấy cặp hữu hạn nhỏ nhất rồi xóa hàng/cột tương ứng;
    # vi: ma trận rỗng/toàn inf: trả np.nan, np.nan và giữ các danh sách chưa ghép.
    mat = np.asarray(association_matrix)
    if mat.size == 0 or not np.isfinite(mat).any():
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    min_idx = np.argmin(mat)
    r, c = np.unravel_index(min_idx, mat.shape)
    if not np.isfinite(mat[r, c]):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)

    selected_track = unassigned_tracks[r]
    selected_meas = unassigned_meas[c]

    remaining_tracks = [t for i, t in enumerate(unassigned_tracks) if i != r]
    remaining_meas = [m for j, m in enumerate(unassigned_meas) if j != c]
    new_matrix = np.asmatrix(np.delete(np.delete(mat, r, axis=0), c, axis=1))

    return selected_track, selected_meas, new_matrix, remaining_tracks, remaining_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    # vi: TODO Part F — kể cả meas_list rỗng, vẫn gọi quản lý cuối lượt.
    # vi: Ghép cặp hữu hạn, filter_obj.update rồi handle_updated_track(track, sensor).
    # vi: Không bỏ qua FOV sau khi đã xóa cặp khỏi danh sách chưa ghép.
    # vi: Kết thúc manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor).
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)

    if len(unassigned_tracks) > 0 and len(unassigned_meas) > 0:
        cost_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)
        while True:
            track, meas, cost_matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
                cost_matrix, unassigned_tracks, unassigned_meas
            )
            if isinstance(track, float) and np.isnan(track):
                break
            filter_obj.update(track, meas)
            manager.handle_updated_track(track, sensor)

    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
