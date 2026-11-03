"""COLMAP 数据库体检

在跑重建之前，从 database.db 里读出足以判断成败的信息。
这是本工具区别于「只是调用 COLMAP」的核心能力之一。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseReport:
    path: str

    # 图像与相机
    n_images: int = 0
    n_cameras: int = 0
    cameras: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    # (camera_id, width, height, 图像数, ...)
    singletons: list[tuple[str, int]] = field(default_factory=list)

    # 特征
    n_images_extracted: int = 0
    n_keypoints: int = 0
    kp_min: int = 0
    kp_max: int = 0
    weak_images: list[tuple[str, int]] = field(default_factory=list)

    # 匹配
    n_pairs: int = 0
    n_pairs_matched: int = 0
    n_matches: int = 0
    n_pairs_verified: int = 0
    n_inliers: int = 0

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def extract_rate(self) -> float:
        return self.n_images_extracted / self.n_images if self.n_images else 0.0

    @property
    def verify_rate(self) -> float:
        return self.n_pairs_verified / self.n_pairs if self.n_pairs else 0.0

    @property
    def mean_inliers(self) -> float:
        return self.n_inliers / self.n_pairs_verified if self.n_pairs_verified else 0.0

    @property
    def mean_keypoints(self) -> float:
        return self.n_keypoints / self.n_images_extracted if self.n_images_extracted else 0.0


def analyze_database(path: str | Path, weak_kp: int = 2000) -> DatabaseReport:
    """读取 database.db 并生成体检报告。只读打开，不修改原文件。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据库不存在：{p}")

    rep = DatabaseReport(path=str(p))

    uri = p.as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        cur = con.cursor()

        def one(sql: str) -> int:
            cur.execute(sql)
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0

        rep.n_images = one("SELECT COUNT(*) FROM images")
        rep.n_cameras = one("SELECT COUNT(*) FROM cameras")

        cur.execute("""
            SELECT cameras.camera_id, cameras.width, cameras.height, COUNT(images.image_id)
            FROM cameras LEFT JOIN images ON images.camera_id = cameras.camera_id
            GROUP BY cameras.camera_id ORDER BY cameras.camera_id
        """)
        rep.cameras = [(int(a), int(b), int(c), int(d), 0) for a, b, c, d in cur.fetchall()]

        cur.execute("""
            SELECT images.name, images.camera_id FROM images
            WHERE images.camera_id IN (
                SELECT camera_id FROM images GROUP BY camera_id HAVING COUNT(*) = 1
            ) ORDER BY images.name
        """)
        rep.singletons = [(n, int(c)) for n, c in cur.fetchall()]

        rep.n_images_extracted = one("SELECT COUNT(*) FROM keypoints")
        rep.n_keypoints = one("SELECT SUM(rows) FROM keypoints")

        cur.execute("""
            SELECT images.name, keypoints.rows
            FROM keypoints JOIN images ON images.image_id = keypoints.image_id
            WHERE keypoints.rows < ? ORDER BY keypoints.rows ASC LIMIT 10
        """, (weak_kp,))
        rep.weak_images = [(n, int(r)) for n, r in cur.fetchall()]

        if rep.n_images_extracted:
            cur.execute("SELECT MIN(rows), MAX(rows) FROM keypoints")
            rep.kp_min, rep.kp_max = (int(v or 0) for v in cur.fetchone())

        rep.n_pairs = one("SELECT COUNT(*) FROM matches")
        rep.n_pairs_matched = one("SELECT COUNT(*) FROM matches WHERE rows > 0")
        rep.n_matches = one("SELECT SUM(rows) FROM matches")
        rep.n_pairs_verified = one("SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0")
        rep.n_inliers = one("SELECT SUM(rows) FROM two_view_geometries")
    finally:
        con.close()

    _diagnose(rep)
    return rep


def _diagnose(rep: DatabaseReport) -> None:
    """把数字翻译成可执行的建议。"""
    if rep.n_images == 0:
        rep.errors.append("数据库里没有图像。请先在 COLMAP 中完成「新建项目」并指定图片目录。")
        return

    if rep.extract_rate < 1.0:
        lost = rep.n_images - rep.n_images_extracted
        rep.errors.append(
            f"有 {lost} 张图没有完成特征提取。常见原因：图片损坏、格式不支持、路径含非 ASCII 字符。"
        )

    # 多相机：最常见的数据卫生问题
    if rep.n_cameras > 1:
        rep.warnings.append(
            f"检测到 {rep.n_cameras} 个相机（理想应为 1 个）。"
            "通常是混入了不同分辨率的图片，或部分图片缺少 EXIF。"
            "多相机会让光束平差不稳定，建议统一图片来源与分辨率。"
        )
        if rep.singletons:
            names = ", ".join(n for n, _ in rep.singletons[:5])
            more = "" if len(rep.singletons) <= 5 else f" 等 {len(rep.singletons)} 张"
            rep.warnings.append(f"以下图片单独构成了相机（可疑）：{names}{more}")

    # 匹配质量
    if rep.n_pairs and rep.verify_rate < 0.40:
        rep.warnings.append(
            f"图像对几何验证通过率仅 {rep.verify_rate:.1%}（建议 > 40%）。"
            "说明相邻照片的共同可见区域太小，通常意味着拍摄间隔过大或重叠不足。"
        )
    elif rep.n_pairs and rep.verify_rate > 0.90 and rep.n_images > 40:
        rep.warnings.append(
            f"图像对验证通过率高达 {rep.verify_rate:.1%}。"
            "通过率过高反而可能说明视角变化太小（帧冗余），检查是否存在连拍重复帧。"
        )

    if rep.n_pairs_verified and rep.mean_inliers < 150:
        rep.warnings.append(
            f"每个验证通过的图像对平均只有 {rep.mean_inliers:.0f} 个内点（健康值 > 150）。匹配偏弱。"
        )

    # 图像数量
    if rep.n_images < 20:
        rep.warnings.append(
            f"只有 {rep.n_images} 张图。单物体重建通常需要 40 张以上，否则三角化率会很低。"
        )

    # 弱特征图
    if rep.weak_images:
        names = ", ".join(f"{n}({k})" for n, k in rep.weak_images[:5])
        rep.warnings.append(
            f"有 {len(rep.weak_images)} 张图特征点少于 2000，可能是模糊、欠曝或大面积纯色：{names}"
        )
