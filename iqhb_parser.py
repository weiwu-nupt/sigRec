"""
iqhb_parser.py
==============
.iqh / .iqb 格式 IQ 文件解析器插件

文件格式说明：
  .iqh  文本头文件，包含中心频率、采样率、数据格式等元数据
  .iqb  二进制数据文件，I/Q 交织存储

数据格式（由 .iqh 中 NumberFormat 字段指定）：
  IQ_Int8   : int8,  每样本 2 字节 (I 1B + Q 1B)
  IQ_Int16  : int16, 每样本 4 字节 (I 2B + Q 2B)，实际值 = raw / 32768 * Scale
  IQ_Single : float32, 每样本 8 字节 (I 4B + Q 4B)，实际值 = raw * Scale

用法（作为插件集成到 signal_tfmap_generator.py）：
  from iqhb_parser import IQHBParser
  # 注册到 PARSER_REGISTRY
  PARSER_REGISTRY["iqhb"] = IQHBParser()

独立测试：
  python iqhb_parser.py --header path/to/file.iqh
"""

import os
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional, List

log = logging.getLogger(__name__)

# ── 尝试导入主模块的基类，若独立运行则本地定义 ──
try:
    from signal_tfmap_generator import IQParser, IQMeta
except ImportError:
    # 独立运行时的轻量版基类
    import dataclasses

    @dataclasses.dataclass
    class IQMeta:
        sample_rate: float
        center_freq: float
        sample_count: int
        data_type: str
        iq_file_path: str
        scale_factor: float = 1.0
        extra: dict = dataclasses.field(default_factory=dict)

        @property
        def duration_s(self):
            return self.sample_count / self.sample_rate

        @property
        def samples_per_10ms(self):
            return int(self.sample_rate * 0.01)

    class IQParser:
        FORMAT_NAME = "base"

        def parse(self, meta_path): raise NotImplementedError
        def read_samples(self, meta, start_sample, num_samples, iq_search_dirs=None):
            raise NotImplementedError

        @staticmethod
        def _resolve_iq_path(iq_path, search_dirs):
            if os.path.isfile(iq_path):
                return iq_path
            filename = os.path.basename(iq_path.replace("\\", "/"))
            for d in (search_dirs or []):
                candidate = os.path.join(d, filename)
                if os.path.isfile(candidate):
                    return candidate
            raise FileNotFoundError(f"IQ 文件未找到: {iq_path}")


# ─────────────────────────────────────────────
# .iqh / .iqb 解析器
# ─────────────────────────────────────────────

class IQHBParser(IQParser):
    """
    解析 .iqh（文本头）+ .iqb（二进制体）格式的 IQ 文件。

    支持的 NumberFormat：
      IQ_Int8    int8  交织，每样本 2 字节
      IQ_Int16   int16 交织，每样本 4 字节，实际值 = raw / 32768 * Scale
      IQ_Single  float32 交织，每样本 8 字节，实际值 = raw * Scale
    """

    FORMAT_NAME = "iqhb"

    # NumberFormat → (numpy dtype, bytes_per_component, 归一化分母)
    _FORMAT_MAP = {
        "iq_int8":   (np.int8,    1, 128.0),
        "iq_int16":  (np.int16,   2, 32768.0),
        "iq_single": (np.float32, 4, 1.0),
    }

    # ── 解析 .iqh 头文件 ──

    def parse(self, meta_path: str) -> IQMeta:
        """
        解析 .iqh 文件，返回标准化的 IQMeta。

        .iqb 文件与 .iqh 同名同目录，自动推断路径。
        """
        meta_path = os.path.normpath(meta_path)
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f".iqh 文件不存在: {meta_path}")

        fields = {}
        with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]

        # 第一行必须是固定标识
        if not lines or "IQ Record Header" not in lines[0]:
            raise ValueError(
                f".iqh 文件格式错误：第一行应为 'IQ Record Header'，实际为: {lines[0]!r}"
            )

        for line in lines[1:]:
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()

        required = ["CenterFrequency", "SampleRate", "NumberFormat",
                    "NumberSamples"]
        missing = [k for k in required if k not in fields]
        if missing:
            raise ValueError(f".iqh 缺少必要字段: {missing}")

        # 推断 .iqb 路径（同目录同文件名）
        stem = Path(meta_path).stem
        iqb_path = str(Path(meta_path).parent / (stem + ".iqb"))

        return IQMeta(
            sample_rate  = float(fields["SampleRate"]),
            center_freq  = float(fields["CenterFrequency"]),
            sample_count = int(fields["NumberSamples"]),
            data_type    = fields["NumberFormat"],
            iq_file_path = iqb_path,
            scale_factor = float(fields.get("Scale", 1.0)),
            extra={
                "acq_bandwidth":  fields.get("AcqBandwidth", ""),
                "record_time":    fields.get("RecordTime", ""),
                "reference_level": fields.get("ReferenceLevel", ""),
            },
        )

    # ── 读取 .iqb 二进制数据 ──

    def read_samples(self, meta: IQMeta, start_sample: int, num_samples: int,
                     iq_search_dirs: Optional[List[str]] = None) -> np.ndarray:
        """
        从 .iqb 文件读取 num_samples 个复数样本，返回 complex64 数组。

        参数:
            meta           : parse() 返回的 IQMeta
            start_sample   : 起始样本序号（0-based）
            num_samples    : 读取样本数
            iq_search_dirs : 额外搜索目录（.iqb 不在 .iqh 同目录时使用）
        """
        # 定位 .iqb 文件
        search_dirs = list(iq_search_dirs or [])
        search_dirs.append(str(Path(meta.iq_file_path).parent))
        iqb_path = self._resolve_iq_path(meta.iq_file_path, search_dirs)

        fmt_key = meta.data_type.lower().replace("-", "_")
        if fmt_key not in self._FORMAT_MAP:
            raise ValueError(
                f"不支持的 NumberFormat: {meta.data_type}  "
                f"(支持: {list(self._FORMAT_MAP.keys())})"
            )
        np_dtype, bytes_per, norm_denom = self._FORMAT_MAP[fmt_key]

        bytes_per_sample = bytes_per * 2   # I + Q
        offset = start_sample * bytes_per_sample
        read_bytes = num_samples * bytes_per_sample

        with open(iqb_path, "rb") as f:
            f.seek(offset)
            raw = np.frombuffer(f.read(read_bytes), dtype=np_dtype)

        actual = len(raw) // 2
        if actual < num_samples:
            log.warning(
                f"请求 {num_samples} 个样本，实际只读到 {actual} 个"
            )
            num_samples = actual

        I = raw[0::2][:num_samples].astype(np.float32)
        Q = raw[1::2][:num_samples].astype(np.float32)

        # 归一化 + 缩放
        scale = meta.scale_factor / norm_denom if norm_denom != 1.0 else meta.scale_factor
        iq = (I + 1j * Q).astype(np.complex64) * scale

        return iq


# ─────────────────────────────────────────────
# 自动注册到主模块
# ─────────────────────────────────────────────

def register(registry: dict):
    """
    将 IQHBParser 注册到 signal_tfmap_generator 的解析器注册表，
    并更新 detect_format 的格式识别逻辑。

    在 signal_tfmap_generator.py 末尾调用：
        import iqhb_parser
        iqhb_parser.register(PARSER_REGISTRY)
    """
    registry["iqhb"] = IQHBParser()
    log.debug("IQHBParser 已注册到解析器注册表")


def patch_detect_format():
    """
    猴子补丁：将 .iqh 格式识别注入 signal_tfmap_generator.detect_format。
    在 signal_tfmap_generator.py 导入后调用一次即可。
    """
    import signal_tfmap_generator as _m

    _original_detect = _m.detect_format

    def _patched_detect(meta_path: str):
        ext = Path(meta_path).suffix.lower()
        if ext == ".iqh":
            return _m.PARSER_REGISTRY.get("iqhb", IQHBParser())
        return _original_detect(meta_path)

    _m.detect_format = _patched_detect
    log.debug("detect_format 已补丁支持 .iqh 格式")


# ─────────────────────────────────────────────
# 独立测试 CLI
# ─────────────────────────────────────────────

def _test_parse(header_path: str):
    """打印 .iqh 解析结果，验证头文件是否正确读取"""
    parser = IQHBParser()
    meta = parser.parse(header_path)

    print("=" * 50)
    print(f"文件        : {header_path}")
    print(f"中心频率    : {meta.center_freq / 1e6:.4f} MHz")
    print(f"采样率      : {meta.sample_rate / 1e3:.2f} kHz  ({meta.sample_rate:.0f} sps)")
    print(f"总采样点数  : {meta.sample_count:,}")
    print(f"录制时长    : {meta.duration_s:.3f} 秒")
    print(f"每10ms样本  : {meta.samples_per_10ms:,}")
    print(f"数据格式    : {meta.data_type}")
    print(f"缩放因子    : {meta.scale_factor}")
    print(f"采集带宽    : {meta.extra.get('acq_bandwidth', 'N/A')} Hz")
    print(f"录制时间    : {meta.extra.get('record_time', 'N/A')}")
    print(f"参考电平    : {meta.extra.get('reference_level', 'N/A')} dBm")
    print(f"推断.iqb路径: {meta.iq_file_path}")

    # 检查 .iqb 文件是否存在
    if os.path.isfile(meta.iq_file_path):
        size = os.path.getsize(meta.iq_file_path)
        fmt_map = {"iq_int8": 2, "iq_int16": 4, "iq_single": 8}
        bps = fmt_map.get(meta.data_type.lower(), 4)
        expected_bytes = meta.sample_count * bps
        print(f".iqb 大小   : {size:,} 字节  (预期 {expected_bytes:,} 字节)")
        if abs(size - expected_bytes) > 16:
            print(f"  ⚠ 大小不匹配，差 {size - expected_bytes:+,} 字节")
        else:
            print(f"  ✓ 大小匹配")
    else:
        print(f"  ⚠ .iqb 文件未找到（路径: {meta.iq_file_path}）")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    p = argparse.ArgumentParser(
        description=".iqh/.iqb 格式解析器独立测试工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--header", required=True,
                   help=".iqh 头文件路径")
    args = p.parse_args()

    _test_parse(args.header)