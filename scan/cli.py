"""PhotoLoop Scan —— 照片三维重建的诊断与后处理工具

用法示例
--------
    python -m scan check  D:\\recon\\ob01\\database.db
    python -m scan report D:\\recon\\ob01\\sparse\\0
    python -m scan cloud  D:\\recon\\ob01\\sparse\\0 -o cloud.ply

设计目标
--------
不做重建本身（那交给 COLMAP），而是补齐它缺的两块：
  1. 事前诊断 —— 在跑重建之前，从数据库判断这组照片有没有希望
  2. 事后修整 —— 把稀疏点云变成能用于 Poisson 重建的干净点云
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import cloud as cloud_mod
from . import dbcheck, ply_io
from .colmap_io import ModelError, read_model
from .report import analyze_model

RULE = "─" * 62


def _title(text: str) -> None:
    print()
    print(RULE)
    print(f"  {text}")
    print(RULE)


def _row(label: str, value, unit: str = "") -> None:
    print(f"  {label:<26}{str(value):>18}{(' ' + unit) if unit else ''}")


def _warns(items: list[str], kind: str = "WARN") -> None:
    if not items:
        return
    print()
    for w in items:
        print(f"  [{kind}] {w}")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def cmd_check(args) -> int:
    _title("数据库体检 / DATABASE CHECK")
    t0 = time.time()
    rep = dbcheck.analyze_database(args.database)
    print(f"  文件    {rep.path}")
    print(f"  耗时    {time.time() - t0:.2f}s")

    print()
    print("  【图像与相机】")
    _row("图像数量", rep.n_images)
    _row("相机数量", rep.n_cameras)

    if rep.cameras:
        print()
        print(f"  {'相机':<8}{'分辨率':<16}{'图像数':>8}")
        for cid, w, h, cnt, _ in rep.cameras:
            print(f"  {cid:<8}{f'{w}x{h}':<16}{cnt:>8}")

    if rep.singletons:
        print()
        print(f"  单独成相机的图像（{len(rep.singletons)} 张）：")
        for name, cid in rep.singletons[:10]:
            print(f"    - {name}   (相机 {cid})")

    print()
    print("  【特征】")
    _row("完成提取的图像", f"{rep.n_images_extracted} / {rep.n_images}")
    _row("特征点总数", f"{rep.n_keypoints:,}")
    _row("平均每张", f"{rep.mean_keypoints:.0f}")
    _row("最少 / 最多", f"{rep.kp_min:,} / {rep.kp_max:,}")

    print()
    print("  【匹配】")
    _row("图像对", f"{rep.n_pairs:,}")
    _row("有匹配的对", f"{rep.n_pairs_matched:,}")
    _row("几何验证通过", f"{rep.n_pairs_verified:,}  ({rep.verify_rate:.1%})")
    _row("内点总数", f"{rep.n_inliers:,}")
    _row("平均内点/对", f"{rep.mean_inliers:.0f}")

    _warns(rep.errors, "ERROR")
    _warns(rep.warnings)

    if not rep.errors and not rep.warnings:
        print()
        print("  [ OK ] 未发现问题，可以继续重建。")

    print()
    return 1 if rep.errors else 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def cmd_report(args) -> int:
    _title("模型质量报告 / MODEL REPORT")
    t0 = time.time()
    model = read_model(args.model)
    rep = analyze_model(model)
    print(f"  模型    {args.model}")
    print(f"  耗时    {time.time() - t0:.2f}s")

    print()
    print("  【规模】")
    _row("相机", rep.n_cameras)
    _row("注册图像", rep.n_images)
    _row("三维点", f"{rep.n_points:,}")

    print()
    print("  【精度】")
    _row("重投影误差 均值", f"{rep.error_mean:.3f}", "px")
    _row("重投影误差 中位", f"{rep.error_median:.3f}", "px")
    _row("重投影误差 90分位", f"{rep.error_p90:.3f}", "px")

    print()
    print("  【点质量】")
    _row("平均轨迹长度", f"{rep.track_mean:.2f}")
    _row("轨迹长度 范围", f"{rep.track_min} – {rep.track_max}")
    _row("短轨迹点(<=2)", f"{rep.short_tracks:,}")

    print()
    print("  【几何】")
    b = rep.bbox
    _row("点云包围盒", f"{b[0]:.2f} x {b[1]:.2f} x {b[2]:.2f}")
    _row("相机分布直径", f"{rep.cam_spread:.2f}")
    _row("相机到场景距离", f"{rep.object_distance:.2f}")
    _row("视差比", f"{rep.parallax_ratio:.2f}")
    _row("共面性(主成分比)", f"{rep.planarity:.3f}")
    _row("方位角覆盖", f"{rep.azimuth_coverage:.1f}", "°")
    _row("方位角最大空洞", f"{rep.azimuth_gap:.1f}", "°")

    _warns(rep.warnings)

    if not rep.warnings:
        print()
        print("  [ OK ] 各项指标正常。")

    print()
    return 0


# --------------------------------------------------------------------------
# cloud
# --------------------------------------------------------------------------

def cmd_cloud(args) -> int:
    _title("生成可用点云 / BUILD CLOUD")
    t0 = time.time()
    model = read_model(args.model)
    print(f"  模型    {args.model}")
    print(f"  过滤    track>={args.min_track}  error<={args.max_error}  keep<={args.keep}%")
    print(f"  法向    {'校正（朝向观测相机）' if not args.no_normals else '跳过'}")

    data, stats = cloud_mod.build_cloud(
        model,
        min_track=args.min_track,
        max_error=args.max_error,
        keep_percentile=args.keep,
        with_normals=not args.no_normals,
        k=args.knn,
    )

    out = Path(args.output)
    ply_io.write_ply(out, data, ascii=args.ascii)

    print()
    print("  【结果】")
    _row("原始点数", f"{int(stats['total']):,}")
    _row("保留点数", f"{int(stats['kept']):,}")
    _row("剔除", f"{int(stats['removed']):,}  ({stats['removed'] / max(stats['total'], 1):.1%})")
    if "flipped" in stats:
        _row("翻转法向", f"{int(stats['flipped']):,}  ({stats['flip_ratio']:.1%})")
    _row("输出文件", str(out))
    _row("文件大小", f"{out.stat().st_size / 1024 / 1024:.2f}", "MB")

    print()
    print(f"  耗时    {time.time() - t0:.2f}s")
    print()
    print("  下一步：在 MeshLab 中导入该 PLY，直接运行 Screened Poisson")
    print("         重建（不要再点「计算法向」，否则会覆盖已校正的法向）。")
    print()
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scan",
        description="PhotoLoop Scan —— 照片三维重建的诊断与后处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-v", "--version", action="version", version="photoloop-scan 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="体检 database.db：跑重建之前先判断有没有希望")
    c.add_argument("database", help="database.db 的路径")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("report", help="分析稀疏模型的质量指标")
    r.add_argument("model", help="稀疏模型目录，例如 sparse/0")
    r.set_defaults(func=cmd_report)

    k = sub.add_parser("cloud", help="清洗点云并校正法向，输出带法向的 PLY")
    k.add_argument("model", help="稀疏模型目录，例如 sparse/0")
    k.add_argument("-o", "--output", required=True, help="输出 PLY 路径")
    k.add_argument("--min-track", type=int, default=4, help="最小轨迹长度（默认 4）")
    k.add_argument("--max-error", type=float, default=1.5, help="最大重投影误差（默认 1.5 px）")
    k.add_argument("--keep", type=float, default=85.0, help="按距离保留的百分比（默认 85）")
    k.add_argument("--knn", type=int, default=32, help="法向估计的邻域大小（默认 32）")
    k.add_argument("--no-normals", action="store_true", help="跳过法向估计")
    k.add_argument("--ascii", action="store_true", help="输出 ASCII PLY（默认二进制）")
    k.set_defaults(func=cmd_cloud)

    return p


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认可能是 GBK，中文会触发 UnicodeEncodeError。
    # 这里统一成 UTF-8，并允许无法编码的字符降级输出而不是崩溃。
    for stream in (sys.stdout, sys.stderr):
        try:
            if (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ModelError, FileNotFoundError) as e:
        print(f"\n  [ERROR] {e}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n  已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
