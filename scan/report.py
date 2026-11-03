"""稀疏模型质量分析

把 COLMAP 的二进制模型翻译成一组能读懂的数，用于判断重建是否可信。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .colmap_io import Model


@dataclass
class ModelReport:
    n_cameras: int = 0
    n_images: int = 0
    n_points: int = 0

    error_mean: float = 0.0
    error_median: float = 0.0
    error_p90: float = 0.0
    track_mean: float = 0.0
    track_min: int = 0
    track_max: int = 0
    short_tracks: int = 0

    bbox: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cam_spread: float = 0.0
    object_distance: float = 0.0
    parallax_ratio: float = 0.0
    azimuth_coverage: float = 0.0
    azimuth_gap: float = 0.0
    planarity: float = 0.0
    """相机分布的最小/最大主成分方差之比。接近 0 表示相机几乎共面，
    此时方位角覆盖才有意义。"""

    warnings: list[str] = field(default_factory=list)


def analyze_model(model: Model) -> ModelReport:
    rep = ModelReport()
    rep.n_cameras = len(model.cameras)
    rep.n_images = len(model.images)
    pts = model.points
    rep.n_points = len(pts)

    if rep.n_points == 0 or rep.n_images == 0:
        rep.warnings.append("模型为空：没有注册图像或没有三维点。")
        return rep

    rep.error_mean = float(pts.error.mean())
    rep.error_median = float(np.median(pts.error))
    rep.error_p90 = float(np.percentile(pts.error, 90))
    rep.track_mean = float(pts.track_len.mean())
    rep.track_min = int(pts.track_len.min())
    rep.track_max = int(pts.track_len.max())
    rep.short_tracks = int((pts.track_len <= 2).sum())

    rep.bbox = pts.xyz.max(axis=0) - pts.xyz.min(axis=0)

    centers = model.camera_centers()
    if len(centers) >= 2:
        rep.cam_spread = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    scene_center = np.median(pts.xyz, axis=0)
    if len(centers) >= 1:
        rep.object_distance = float(np.median(np.linalg.norm(centers - scene_center, axis=1)))
        if rep.object_distance > 1e-9:
            rep.parallax_ratio = rep.cam_spread / rep.object_distance

    if len(centers) >= 3:
        cov = np.cov((centers - centers.mean(axis=0)).T)
        evals, vecs = np.linalg.eigh(cov)
        rep.planarity = float(evals[0] / evals[-1]) if evals[-1] > 1e-12 else 0.0
        basis = vecs[:, -2:]  # 方差最大的两个方向 = 相机所在平面
        proj = (centers - centers.mean(axis=0)) @ basis
        az = np.degrees(np.arctan2(proj[:, 1], proj[:, 0]))
        az_sorted = np.sort(az)
        gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360.0]]))
        rep.azimuth_gap = float(gaps.max())
        rep.azimuth_coverage = float(360.0 - gaps.max())

    _diagnose(rep)
    return rep


def _diagnose(rep: ModelReport) -> None:
    if rep.error_mean > 2.0:
        rep.warnings.append(
            f"平均重投影误差 {rep.error_mean:.2f} px 偏大（健康值 < 1.5）。位姿可能不准确。"
        )

    if rep.n_images >= 30 and rep.track_mean > 10:
        rep.warnings.append(
            f"平均轨迹长度 {rep.track_mean:.1f} 偏高（单物体通常 3–6）。"
            "说明存在大量近重复帧，连拍或帧率过高会导致这种情况。"
        )

    if rep.short_tracks:
        ratio = rep.short_tracks / max(rep.n_points, 1)
        rep.warnings.append(
            f"有 {rep.short_tracks} 个点（{ratio:.1%}）只被 1–2 张图观测到，可靠性低，建议在清洗时剔除。"
        )

    if rep.parallax_ratio < 0.5:
        rep.warnings.append(
            f"视差比 {rep.parallax_ratio:.2f} 偏小（< 0.5）。相机位移相对物距不足，深度估计不可靠。"
            "常见原因：原地转身、绕的圈太小。"
        )

    if rep.planarity > 0.25:
        rep.warnings.append(
            f"相机分布不够共面（主成分比 {rep.planarity:.2f}，理想 < 0.25），"
            "方位角覆盖估计可能不可靠——相机高度变化过大时该指标会失真。"
        )
    elif 0 < rep.azimuth_coverage < 270:
        rep.warnings.append(
            f"相机在最佳拟合平面上只覆盖了 {rep.azimuth_coverage:.0f}°"
            f"（最大空洞 {rep.azimuth_gap:.0f}°）。"
            "物体有相当一部分没有被拍到，模型会缺失。建议完整绕一圈。"
        )

    if rep.n_cameras > 1:
        rep.warnings.append(
            f"模型里有 {rep.n_cameras} 个相机（理想为 1）。检查是否混入了不同分辨率的图片。"
        )
