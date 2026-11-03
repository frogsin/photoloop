"""COLMAP 稀疏模型读取（BIN 格式，零外部依赖）

COLMAP 的稀疏模型目录通常包含：
    cameras.bin    相机内参
    images.bin     已注册图像的位姿
    points3D.bin   三维点及其观测轨迹
    rigs.bin / frames.bin   COLMAP 4.x 新增，本模块忽略

本模块只依赖标准库与 numpy，不需要安装 pycolmap。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COLMAP 的相机模型编号 -> (名称, 参数个数)
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


class ModelError(RuntimeError):
    """模型文件缺失或格式不符。"""


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    @property
    def focal(self) -> float:
        """返回像素焦距的近似值（不同模型位置不同，取第一个参数）。"""
        return float(self.params[0]) if self.params else 0.0


@dataclass
class Image:
    image_id: int
    camera_id: int
    name: str
    qvec: np.ndarray   # (qw, qx, qy, qz)
    tvec: np.ndarray   # 世界 -> 相机
    center: np.ndarray  # 相机中心（世界坐标）

    @property
    def rotation(self) -> np.ndarray:
        """由四元数还原旋转矩阵 R（世界 -> 相机）。"""
        qw, qx, qy, qz = self.qvec
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])


@dataclass
class PointCloud:
    """稀疏点云。每个点记录被多少张图观测，以及第一张观测它的图像。"""

    xyz: np.ndarray       # (N, 3) float64
    rgb: np.ndarray       # (N, 3) uint8
    error: np.ndarray     # (N,)   float64，重投影误差（像素）
    track_len: np.ndarray  # (N,)   int32，轨迹长度
    first_image_id: np.ndarray  # (N,) int32，第一个观测者

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[int, Image]
    points: PointCloud

    def camera_centers(self) -> np.ndarray:
        if not self.images:
            return np.zeros((0, 3))
        return np.stack([im.center for im in self.images.values()])


# --------------------------------------------------------------------------
# 二进制读取
# --------------------------------------------------------------------------

def _u32(f):
    return struct.unpack("<I", f.read(4))[0]


def _i32(f):
    return struct.unpack("<i", f.read(4))[0]


def _u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def _doubles(f, n):
    return np.frombuffer(f.read(8 * n), dtype="<f8")


def _read_name(f) -> str:
    buf = bytearray()
    while True:
        ch = f.read(1)
        if not ch or ch == b"\x00":
            break
        buf += ch
    return buf.decode("utf-8", errors="replace")


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with open(path, "rb") as f:
        n = _u64(f)
        for _ in range(n):
            cid = _u32(f)
            model_id = _i32(f)
            width = _u64(f)
            height = _u64(f)
            name, nparams = CAMERA_MODELS.get(model_id, (f"UNKNOWN_{model_id}", 8))
            params = tuple(_doubles(f, nparams).tolist())
            cameras[cid] = Camera(cid, name, int(width), int(height), params)
    return cameras


def read_images(path: Path) -> dict[int, Image]:
    """读取已注册图像。为了速度，跳过每张图的二维观测点。"""
    images: dict[int, Image] = {}
    with open(path, "rb") as f:
        n = _u64(f)
        for _ in range(n):
            image_id = _u32(f)
            qvec = _doubles(f, 4)
            tvec = _doubles(f, 3)
            camera_id = _u32(f)
            name = _read_name(f)
            num_points2d = _u64(f)
            # 每个观测点占 8(x)+8(y)+8(point3D_id) 字节，直接跳过
            f.seek(num_points2d * 24, 1)

            im = Image(image_id, camera_id, name, qvec, tvec, np.zeros(3))
            # 相机中心 C = -R^T t
            im.center = -im.rotation.T @ tvec
            images[image_id] = im
    return images


def read_points3d(path: Path) -> PointCloud:
    """读取三维点。为了速度，每条轨迹只保留第一个观测者。"""
    xs, cols, errs, tls, firsts = [], [], [], [], []
    with open(path, "rb") as f:
        n = _u64(f)
        for _ in range(n):
            _pid = _u64(f)
            xyz = _doubles(f, 3)
            rgb = np.frombuffer(f.read(3), dtype=np.uint8)
            err = struct.unpack("<d", f.read(8))[0]
            track_len = _u64(f)
            if track_len > 0:
                first_img = _u32(f)
                _u32(f)  # point2D_idx，不需要
                if track_len > 1:
                    f.seek((track_len - 1) * 8, 1)
            else:
                first_img = -1
            xs.append(xyz)
            cols.append(rgb)
            errs.append(err)
            tls.append(track_len)
            firsts.append(first_img)

    if not xs:
        return PointCloud(np.zeros((0, 3)), np.zeros((0, 3), np.uint8),
                          np.zeros(0), np.zeros(0, np.int32), np.zeros(0, np.int32))

    return PointCloud(
        xyz=np.stack(xs).astype(np.float64),
        rgb=np.stack(cols).astype(np.uint8),
        error=np.asarray(errs, dtype=np.float64),
        track_len=np.asarray(tls, dtype=np.int32),
        first_image_id=np.asarray(firsts, dtype=np.int32),
    )


def read_model(model_dir: str | Path) -> Model:
    """读取一个 COLMAP 稀疏模型目录（例如 sparse/0）。"""
    d = Path(model_dir)
    if not d.is_dir():
        raise ModelError(f"模型目录不存在：{d}")

    required = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [n for n in required if not (d / n).exists()]
    if missing:
        hint = ""
        if (d / "cameras.txt").exists():
            hint = "\n检测到 TXT 格式，请先用 `colmap model_converter --output_type BIN` 转换。"
        raise ModelError(f"缺少文件：{', '.join(missing)}{hint}")

    return Model(
        cameras=read_cameras(d / "cameras.bin"),
        images=read_images(d / "images.bin"),
        points=read_points3d(d / "points3D.bin"),
    )
