"""PhotoLoop Scan

照片三维重建的诊断与后处理工具。

模块
----
colmap_io   COLMAP 稀疏模型（BIN）读写
dbcheck     database.db 体检
cloud       点云清洗与法向校正
report      模型质量分析
ply_io      PLY 读写
cli         命令行入口
"""

__version__ = "0.1.0"

from .colmap_io import Model, ModelError, read_model  # noqa: F401

__all__ = ["Model", "ModelError", "read_model", "__version__"]
