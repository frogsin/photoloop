"""点云清洗与法向校正

这两步是本项目的核心贡献：

1. 清洗：用 COLMAP 自带的两个质量指标剔除离群点，避免 Poisson 重建被
   远处的孤立点撑大包围盒、生成大量气泡。

2. 法向校正：PCA 算出的法向只有「方向」没有「正负」。用观测到该点的
   相机位置来定向，是 SfM 领域的标准做法。实测中约 70% 的法向需要翻转。
"""

from __future__ import annotations

import numpy as np

from .colmap_io import Model


def filter_points(
    model: Model,
    min_track: int = 4,
    max_error: float = 1.5,
    keep_percentile: float = 85.0,
) -> np.ndarray:
    """返回保留下来的点的索引。

    参数
    ----
    min_track : int
        点的最小轨迹长度（被多少张图观测到）。只有 2 张图看到的点最不可靠。
    max_error : float
        最大重投影误差（像素）。
    keep_percentile : float
        在通过质量过滤后，再按「到点云中心距离」保留的百分比。
        设为 100 表示不做空间裁剪。
    """
    pts = model.points
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    mask = (pts.track_len >= min_track) & (pts.error <= max_error)
    idx = np.flatnonzero(mask)

    if idx.size == 0:
        return idx

    if keep_percentile < 100.0:
        xyz = pts.xyz[idx]
        center = np.median(xyz, axis=0)
        dist = np.linalg.norm(xyz - center, axis=1)
        thr = np.percentile(dist, keep_percentile)
        idx = idx[dist <= thr]

    return idx


def estimate_normals(
    points: np.ndarray,
    k: int = 32,
    cell_div: float = 25.0,
) -> np.ndarray:
    """用邻域 PCA 估计法向。

    返回 (N, 3)，每个法向单位化，但**正负号未定**（这就是需要校正的地方）。
    """
    n = points.shape[0]
    if n < 3:
        return np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))

    bbox = points.max(axis=0) - points.min(axis=0)
    cell = max(float(np.mean(bbox)) / cell_div, 1e-9)
    ijk = np.floor((points - points.min(axis=0)) / cell).astype(np.int64)

    keys = ijk[:, 0] * 1_000_003 + ijk[:, 1] * 1_000_033 + ijk[:, 2]
    order = np.argsort(keys)
    keys_sorted = keys[order]
    uniq, starts = np.unique(keys_sorted, return_index=True)
    ends = np.append(starts[1:], len(keys_sorted))

    # 邻居候选（每个点最多 k 个）
    nbr = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        x, y, z = ijk[i]
        found = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (x + dx) * 1_000_003 + (y + dy) * 1_000_033 + (z + dz)
                    pos = np.searchsorted(uniq, key)
                    if pos < len(uniq) and uniq[pos] == key:
                        found.append(order[starts[pos]:ends[pos]])
        if not found:
            continue
        cand = np.concatenate(found)
        if cand.size > k:
            d = np.linalg.norm(points[cand] - points[i], axis=1)
            cand = cand[np.argpartition(d, k - 1)[:k]]
        nbr[i, :cand.size] = cand

    # 批量 PCA
    valid = nbr >= 0
    safe = np.where(valid, nbr, 0)
    neigh = points[safe]
    w = valid[:, :, None].astype(np.float64)
    cnt = np.maximum(w.sum(axis=1), 1.0)
    mean = (neigh * w).sum(axis=1) / cnt
    centered = (neigh - mean[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", centered, centered) / cnt[:, :, None]
    _, evecs = np.linalg.eigh(cov)

    normals = evecs[:, :, 0]
    bad = np.linalg.norm(normals, axis=1) < 1e-9
    normals[bad] = (0.0, 0.0, 1.0)
    return normals


def orient_normals(
    normals: np.ndarray,
    points: np.ndarray,
    camera_centers: np.ndarray,
) -> tuple[np.ndarray, int]:
    """把法向朝向观测到该点的相机。

    返回 (校正后的法向, 被翻转的数量)。
    """
    view = camera_centers - points
    flip = np.einsum("ni,ni->n", normals, view) < 0
    out = normals.copy()
    out[flip] *= -1.0
    return out, int(flip.sum())


def build_cloud(
    model: Model,
    min_track: int = 4,
    max_error: float = 1.5,
    keep_percentile: float = 85.0,
    with_normals: bool = True,
    k: int = 32,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """完整流程：清洗 → 估计法向 → 用相机定向。

    返回 (PLY 属性字典, 统计信息)。
    """
    pts = model.points
    idx = filter_points(model, min_track, max_error, keep_percentile)

    xyz = pts.xyz[idx]
    rgb = pts.rgb[idx]
    first_img = pts.first_image_id[idx]

    stats: dict[str, float] = {
        "total": float(len(pts)),
        "kept": float(len(idx)),
        "removed": float(len(pts) - len(idx)),
    }

    data: dict[str, np.ndarray] = {
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "red": rgb[:, 0], "green": rgb[:, 1], "blue": rgb[:, 2],
    }

    if with_normals and len(idx) >= 3:
        normals = estimate_normals(xyz, k=k)

        cam = np.zeros_like(xyz)
        for j, iid in enumerate(first_img):
            im = model.images.get(int(iid))
            cam[j] = im.center if im is not None else np.nan
        valid = ~np.isnan(cam).any(axis=1)
        if valid.any():
            fixed, n_flip = orient_normals(normals[valid], xyz[valid], cam[valid])
            normals[valid] = fixed
            stats["flipped"] = float(n_flip)
            stats["flip_ratio"] = float(n_flip) / max(int(valid.sum()), 1)

        data["nx"] = normals[:, 0]
        data["ny"] = normals[:, 1]
        data["nz"] = normals[:, 2]

    return data, stats
