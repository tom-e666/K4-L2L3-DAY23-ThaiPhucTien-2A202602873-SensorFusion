"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

from fusion_lab.workspace_support import get_tracking_params
import numpy as np

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    # vi: TODO Part G — p_s = R @ p + t; loại tọa độ không hữu hạn.
    # vi: Camera cần x_s > 1e-6; FOV từ nội tại và bề rộng ảnh.
    # vi: Với cả lidar/camera: kiểm tra atan2(y_s, x_s) nằm trong sensor.fov.
    pos = np.asarray(x, dtype=float).reshape(-1)[:3]
    T = np.asarray(sensor.veh_to_sens)
    p_s = T[:3, :3] @ pos + T[:3, 3]

    if not np.isfinite(p_s).all():
        return False

    is_camera = getattr(sensor, "name", "") == "camera" or hasattr(sensor, "f_i")
    depth_eps = getattr(sensor, "depth_epsilon", 1e-6)
    if is_camera and p_s[0] <= depth_eps:
        return False

    angle = np.arctan2(p_s[1], p_s[0])
    min_fov, max_fov = min(sensor.fov), max(sensor.fov)
    return bool(min_fov <= angle <= max_fov)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    # vi: TODO Part G — tính p_s = R @ p + t; trước phép chia kiểm tra hữu hạn
    # vi: và x_s > 1e-6, ngược lại raise ValueError có tọa độ.
    # vi: u = c_i - f_i * y_s/x_s; v = c_j - f_j * z_s/x_s.
    pos = np.asarray(x, dtype=float).reshape(-1)[:3]
    T = np.asarray(sensor.veh_to_sens)
    p_s = T[:3, :3] @ pos + T[:3, 3]

    depth_eps = getattr(sensor, "depth_epsilon", 1e-6)
    if not np.isfinite(p_s).all() or p_s[0] <= depth_eps:
        raise ValueError(
            f"Camera projection needs finite coordinates and positive depth > {depth_eps:g}; "
            f"sensor position={p_s.tolist()}"
        )

    x_s, y_s, z_s = p_s[0], p_s[1], p_s[2]
    u = sensor.c_i - sensor.f_i * y_s / x_s
    v = sensor.c_j - sensor.f_j * z_s / x_s
    return np.asmatrix([[u], [v]])


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    # vi: TODO Part G — z mat 2x1; R diag sigma_cam_i^2, sigma_cam_j^2 từ params.
    params = get_tracking_params()
    z_mat = np.asmatrix(np.asarray(z, dtype=float).reshape(2, 1))
    R = np.matrix(
        np.diag([float(params.sigma_cam_i**2), float(params.sigma_cam_j**2)]),
        dtype=float,
    )
    return {"z": z_mat, "R": R, "sensor": sensor}
