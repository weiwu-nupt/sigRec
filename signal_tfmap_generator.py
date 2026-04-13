"""
signal_tfmap_generator.py
=========================
通用 IQ 信号时频图生成模块
支持: DVB (SignalHound .xml/.iq), 后续可扩展 5G/4G 等格式

使用方法:
  # 直接指定文件列表
  python signal_tfmap_generator.py \
      --files file1.xml file2.xml ... \
      --output_dir ./output \
      --images_per_file 10

  # 从 txt 文件读取路径列表（每行一个 .xml 路径）
  python signal_tfmap_generator.py \
      --filelist paths.txt \
      --output_dir ./output \
      --images_per_file 10

作者: Claude
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import importlib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 数据类: 解析后的 IQ 元数据
# ─────────────────────────────────────────────

@dataclass
class IQMeta:
    """标准化的 IQ 录制元数据（信号格式无关）"""
    sample_rate: float          # 采样率 (Hz)
    center_freq: float          # 中心频率 (Hz)
    sample_count: int           # 总采样数
    data_type: str              # 原始数据类型描述
    iq_file_path: str           # 二进制 .iq 文件路径（可能需重映射）
    scale_factor: float = 1.0   # 幅度缩放因子
    extra: dict = field(default_factory=dict)  # 格式特有字段

    @property
    def duration_s(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def samples_per_10ms(self) -> int:
        return int(self.sample_rate * 0.01)


# ─────────────────────────────────────────────
# 格式解析器基类
# ─────────────────────────────────────────────

class IQParser:
    """所有格式解析器的基类，子类只需实现 parse() 和 read_samples()"""

    FORMAT_NAME = "base"

    def parse(self, meta_path: str) -> IQMeta:
        """解析元数据文件，返回 IQMeta"""
        raise NotImplementedError

    def read_samples(self, meta: IQMeta, start_sample: int, num_samples: int,
                     iq_search_dirs: Optional[List[str]] = None) -> np.ndarray:
        """
        从二进制 IQ 文件读取 num_samples 个复数样本 (complex64)
        iq_search_dirs: 当 iq_file_path 为绝对路径但文件不存在时，在此目录列表搜索同名文件
        """
        raise NotImplementedError

    @staticmethod
    def _resolve_iq_path(iq_path: str, search_dirs: Optional[List[str]]) -> str:
        """尝试定位实际的 .iq 文件"""
        if os.path.isfile(iq_path):
            return iq_path
        filename = os.path.basename(iq_path.replace("\\", "/"))
        if search_dirs:
            for d in search_dirs:
                candidate = os.path.join(d, filename)
                if os.path.isfile(candidate):
                    return candidate
        raise FileNotFoundError(
            f"IQ 文件未找到: {iq_path}\n"
            f"  搜索目录: {search_dirs}\n"
            f"  如果 .iq 文件与 .xml 在同一目录，请将该目录加入 --iq_dirs"
        )


# ─────────────────────────────────────────────
# DVB / SignalHound 解析器
# ─────────────────────────────────────────────

class SignalHoundParser(IQParser):
    """
    解析 SignalHound IQ 录制文件 (.xml + .iq)
    DataType 支持:
      - "Complex Short"  : int16 交织 I/Q，每样本 4 字节
      - "Complex Float"  : float32 交织 I/Q，每样本 8 字节（预留）
    """

    FORMAT_NAME = "signalhound_dvb"

    _DTYPE_MAP = {
        "complex short":  (np.int16,   2),   # (numpy dtype, bytes per component)
        "complex float":  (np.float32, 4),
    }

    def parse(self, meta_path: str) -> IQMeta:
        tree = ET.parse(meta_path)
        root = tree.getroot()

        def get(tag, cast=str):
            el = root.find(tag)
            if el is None:
                raise ValueError(f"XML 缺少字段: <{tag}>")
            return cast(el.text.strip())

        return IQMeta(
            sample_rate   = get("SampleRate",      float),
            center_freq   = get("CenterFrequency", float),
            sample_count  = get("SampleCount",     int),
            data_type     = get("DataType"),
            iq_file_path  = get("IQFileName"),
            scale_factor  = get("ScaleFactor",     float),
            extra={
                "device":   root.findtext("DeviceType", ""),
                "serial":   root.findtext("SerialNumber", ""),
                "epoch_ns": root.findtext("EpochNanos", ""),
            },
        )

    def read_samples(self, meta: IQMeta, start_sample: int, num_samples: int,
                     iq_search_dirs: Optional[List[str]] = None) -> np.ndarray:
        iq_path = self._resolve_iq_path(meta.iq_file_path, iq_search_dirs)
        dtype_key = meta.data_type.lower()
        if dtype_key not in self._DTYPE_MAP:
            raise ValueError(f"不支持的 DataType: {meta.data_type}")

        np_dtype, bytes_per = self._DTYPE_MAP[dtype_key]
        bytes_per_sample = bytes_per * 2      # I + Q
        offset = start_sample * bytes_per_sample

        with open(iq_path, "rb") as f:
            f.seek(offset)
            raw = np.frombuffer(f.read(num_samples * bytes_per_sample), dtype=np_dtype)

        if len(raw) < num_samples * 2:
            raise IOError(f"读取到的样本数不足 (期望 {num_samples*2}, 实际 {len(raw)})")

        iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
        # 归一化到 int16 满量程
        if dtype_key == "complex short":
            iq /= 32768.0
        return iq


# ─────────────────────────────────────────────
# 5G NR / 4G LTE 解析器占位（后续扩展）
# ─────────────────────────────────────────────

class NR5GParser(IQParser):
    """
    5G NR IQ 文件解析器（待实现）
    典型格式: SigMF (.sigmf-meta + .sigmf-data) 或自定义二进制
    """
    FORMAT_NAME = "5g_nr"

    def parse(self, meta_path: str) -> IQMeta:
        # TODO: 解析 SigMF JSON 元数据
        raise NotImplementedError("5G NR 解析器待实现")

    def read_samples(self, meta, start_sample, num_samples, iq_search_dirs=None):
        raise NotImplementedError("5G NR 解析器待实现")


class LTE4GParser(IQParser):
    """4G LTE IQ 文件解析器（待实现）"""
    FORMAT_NAME = "4g_lte"

    def parse(self, meta_path: str) -> IQMeta:
        raise NotImplementedError("4G LTE 解析器待实现")

    def read_samples(self, meta, start_sample, num_samples, iq_search_dirs=None):
        raise NotImplementedError("4G LTE 解析器待实现")


# ─────────────────────────────────────────────
# 格式自动检测 & 注册表
# ─────────────────────────────────────────────

PARSER_REGISTRY: dict[str, IQParser] = {
    "signalhound_dvb": SignalHoundParser(),
    "5g_nr":           NR5GParser(),
    "4g_lte":          LTE4GParser(),
}


def detect_format(meta_path: str) -> IQParser:
    """根据文件内容自动识别信号格式，返回对应 Parser"""
    ext = Path(meta_path).suffix.lower()

    if ext == ".xml":
        # 快速探测: 检查 XML 根标签
        try:
            tree = ET.parse(meta_path)
            root = tree.getroot()
            if root.tag == "SignalHoundIQFile":
                return PARSER_REGISTRY["signalhound_dvb"]
        except ET.ParseError:
            pass

    if ext in (".json", ".sigmf-meta"):
        return PARSER_REGISTRY["5g_nr"]

    raise ValueError(f"无法识别文件格式: {meta_path}")


# ─────────────────────────────────────────────
# 时频图生成核心
# ─────────────────────────────────────────────

def generate_spectrogram(
    iq_samples: np.ndarray,
    sample_rate: float,
    nfft: int = 256,
    overlap: float = 0.75,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算短时傅里叶变换 (STFT)，返回 (time_axis, freq_axis, power_dB)

    参数:
        iq_samples  : 复数 IQ 样本 (complex64)
        sample_rate : 采样率 (Hz)
        nfft        : FFT 点数
        overlap     : 相邻帧重叠比例 [0, 1)
        window      : 窗函数名 ('hann', 'hamming', 'blackman', ...)

    返回:
        t_axis   : 时间轴 (秒)，shape (num_frames,)
        f_axis   : 频率轴 (Hz, 相对带宽)，shape (nfft,)
        power_db : 功率谱 (dBFS)，shape (nfft, num_frames)
    """
    hop = max(1, int(nfft * (1 - overlap)))
    win = np.hanning(nfft) if window == "hann" else np.blackman(nfft)

    n_frames = (len(iq_samples) - nfft) // hop + 1
    if n_frames <= 0:
        raise ValueError(f"样本数 ({len(iq_samples)}) 不足一帧 (nfft={nfft})")

    # 预分配频谱矩阵
    spec = np.zeros((nfft, n_frames), dtype=np.float32)
    for i in range(n_frames):
        seg = iq_samples[i*hop : i*hop + nfft] * win
        spectrum = np.fft.fftshift(np.fft.fft(seg, n=nfft))
        spec[:, i] = np.abs(spectrum) ** 2

    # 转 dBFS，防 log(0)
    power_db = 10 * np.log10(spec + 1e-12)

    # 频率轴: [-fs/2, +fs/2]
    f_axis = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0/sample_rate))
    t_axis = np.arange(n_frames) * hop / sample_rate

    return t_axis, f_axis, power_db


def save_spectrogram_image(
    power_db: np.ndarray,
    out_path: str,
    img_size: Tuple[int, int] = (224, 224),
    colormap: str = "viridis",
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
):
    """
    将功率谱矩阵渲染为无标注图像并保存 (PNG)

    参数:
        power_db        : (freq_bins, time_frames) 功率谱 dBFS
        out_path        : 输出文件路径
        img_size        : (width, height) 像素，默认 224×224
        colormap        : matplotlib colormap 名称
        vmin_percentile : 动态范围下限百分位
        vmax_percentile : 动态范围上限百分位
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    vmin = np.percentile(power_db, vmin_percentile)
    vmax = np.percentile(power_db, vmax_percentile)

    # 归一化到 [0, 1]
    norm = np.clip((power_db - vmin) / (vmax - vmin + 1e-12), 0, 1)

    # 应用 colormap -> RGBA
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap)  # matplotlib < 3.7 fallback
    rgba = cmap(norm)             # (H, W, 4)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)

    # Resize to target dimensions (freq axis = height, time axis = width)
    from PIL import Image
    img = Image.fromarray(rgb)
    img = img.resize(img_size, Image.LANCZOS)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG", optimize=False)


# ─────────────────────────────────────────────
# 批量生成主函数
# ─────────────────────────────────────────────

def process_file(
    meta_path: str,
    output_dir: str,
    images_per_file: int = 10,
    img_size: Tuple[int, int] = (224, 224),
    nfft: int = 256,
    overlap: float = 0.75,
    colormap: str = "viridis",
    iq_search_dirs: Optional[List[str]] = None,
    file_index: int = 0,
    global_start: int = 0,
) -> List[str]:
    """
    对单个 meta 文件生成 images_per_file 张时频图

    返回: 已保存的图像路径列表
    """
    log.info(f"[{file_index}] 处理: {meta_path}")

    parser = detect_format(meta_path)
    meta = parser.parse(meta_path)

    samples_10ms = meta.samples_per_10ms
    max_start = meta.sample_count - samples_10ms
    if max_start <= 0:
        raise ValueError(f"文件采样数 ({meta.sample_count}) 不足 10ms ({samples_10ms})")

    # 均匀分布采样起始点（跳过首尾各 5% 的边缘）
    margin = int(max_start * 0.05)
    usable = max_start - 2 * margin
    if images_per_file > 1:
        starts = [margin + int(usable * i / (images_per_file - 1))
                  for i in range(images_per_file)]
    else:
        starts = [margin + usable // 2]

    stem = Path(meta_path).stem
    saved = []

    # 确定 IQ 搜索目录（自动加入 xml 所在目录）
    dirs = list(iq_search_dirs or [])
    dirs.append(str(Path(meta_path).parent))

    for idx, start in enumerate(starts):
        try:
            iq = parser.read_samples(meta, start, samples_10ms, dirs)
        except FileNotFoundError as e:
            log.error(str(e))
            log.warning(f"  → 使用 PreviewTrace 模拟 IQ（仅供演示，精度有限）")
            iq = _simulate_iq_from_preview(meta_path, samples_10ms)

        _, _, pdb = generate_spectrogram(iq, meta.sample_rate, nfft=nfft, overlap=overlap)

        out_name = f"{global_start + idx + 1}.png"
        out_path = os.path.join(output_dir, out_name)
        save_spectrogram_image(pdb, out_path, img_size=img_size, colormap=colormap)
        log.info(f"  → 已保存: {out_path}")
        saved.append(out_path)

    return saved


def _simulate_iq_from_preview(xml_path: str, n_samples: int) -> np.ndarray:
    """
    当 .iq 文件不可用时，用 XML 中的 PreviewTrace（功率谱）反推模拟 IQ。
    仅用于测试/演示，不具备精确相位信息。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    preview_text = root.findtext("PreviewTrace", "")
    if not preview_text.strip():
        # 无 PreviewTrace，生成白噪声
        return (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)).astype(np.complex64)

    # PreviewTrace 是功率 dBm 随频率的向量
    pwr_db = np.array([float(x) for x in preview_text.split(",") if x.strip()])
    pwr_lin = 10 ** (pwr_db / 10)
    pwr_lin = pwr_lin / pwr_lin.max()

    # IFFT 从频谱合成时域信号（随机相位）
    n_fft = len(pwr_lin)
    phase = np.exp(1j * np.random.uniform(0, 2*np.pi, n_fft))
    spectrum = np.sqrt(pwr_lin) * phase
    td = np.fft.ifft(spectrum)

    # 循环平铺到所需长度
    repeats = (n_samples + n_fft - 1) // n_fft
    td_tiled = np.tile(td, repeats)[:n_samples]
    return td_tiled.astype(np.complex64)


# ─────────────────────────────────────────────
# 目录扫描: 叶子目录 XML 自动发现
# ─────────────────────────────────────────────

def scan_leaf_dirs(root_dir: str, xml_ext: str = ".xml") -> List[str]:
    """
    遍历 root_dir，找出所有「最深叶子目录」（即不含任何子目录的目录），
    每个叶子目录按文件名排序后取第一个 XML 文件。

    参数:
        root_dir : 根目录路径，例如 C:/Users/.../GSM/DCS1800/Downlink
        xml_ext  : 元数据文件扩展名，默认 .xml

    返回: XML 文件路径列表（每个叶子目录一个）
    """
    root_dir = os.path.normpath(root_dir)
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"目录不存在: {root_dir}")

    result = []

    for dirpath, subdirs, files in os.walk(root_dir):
        # 过滤掉隐藏目录（以 . 开头），避免误入 .git 等
        subdirs[:] = [d for d in subdirs if not d.startswith(".")]

        # 叶子目录：没有任何子目录
        if subdirs:
            continue

        xml_files = sorted(
            f for f in files if f.lower().endswith(xml_ext)
        )
        if not xml_files:
            log.warning(f"叶子目录无 XML 文件，跳过: {dirpath}")
            continue

        chosen = os.path.join(dirpath, xml_files[0])
        log.info(f"  叶子目录: {dirpath}")
        log.info(f"    → 选取: {xml_files[0]}"
                 + (f"  (共 {len(xml_files)} 个 XML)" if len(xml_files) > 1 else ""))
        result.append(chosen)

    result.sort()  # 按路径排序，保证跨平台一致性
    return result



def batch_generate(
    meta_paths: List[str],
    output_dir: str,
    images_per_file: int = 10,
    img_size: Tuple[int, int] = (224, 224),
    nfft: int = 256,
    overlap: float = 0.75,
    colormap: str = "viridis",
    iq_search_dirs: Optional[List[str]] = None,
) -> List[str]:
    """
    批量处理多个 meta 文件，生成时频图

    参数:
        meta_paths      : 元数据文件路径列表（.xml / .json 等）
        output_dir      : 图片输出目录
        images_per_file : 每个文件生成的图片数量
        img_size        : 输出图像尺寸 (width, height)
        nfft            : STFT 的 FFT 点数
        overlap         : STFT 帧重叠比例
        colormap        : 色彩映射方案 (viridis/plasma/inferno/magma 等)
        iq_search_dirs  : 额外的 .iq 文件搜索目录

    返回: 所有已保存图像的路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    all_saved = []

    for i, path in enumerate(meta_paths):
        try:
            saved = process_file(
                meta_path=path,
                output_dir=output_dir,
                images_per_file=images_per_file,
                img_size=img_size,
                nfft=nfft,
                overlap=overlap,
                colormap=colormap,
                iq_search_dirs=iq_search_dirs,
                file_index=i,
                global_start=i * images_per_file,
            )
            all_saved.extend(saved)
        except Exception as e:
            log.error(f"处理 {path} 时出错: {e}")

    log.info(f"\n完成! 共生成 {len(all_saved)} 张图像 → {output_dir}")
    return all_saved


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="通用 IQ 信号时频图批量生成工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--files",          nargs="+", default=None,
                   help="元数据文件路径列表 (.xml / .sigmf-meta 等)")
    p.add_argument("--filelist",        default=None,
                   help="包含文件路径的 txt 文件，每行一个路径（与 --files 二选一）")
    p.add_argument("--scan_dir",        nargs="+", default=None,
                   help="自动扫描目录：找出所有最深叶子目录，每个目录取第一个 XML 文件")
    p.add_argument("--output_dir",     default="./tfmap_output",
                   help="图像输出目录")
    p.add_argument("--images_per_file", type=int, default=10,
                   help="每个文件生成的图像数量")
    p.add_argument("--img_size",       nargs=2, type=int, default=[224, 224],
                   metavar=("W", "H"),
                   help="输出图像尺寸 (宽 高)")
    p.add_argument("--nfft",           type=int, default=256,
                   help="STFT FFT 点数（时频分辨率权衡）")
    p.add_argument("--overlap",        type=float, default=0.75,
                   help="STFT 帧重叠比例 [0, 1)")
    p.add_argument("--colormap",       default="viridis",
                   help="Matplotlib colormap 名称")
    p.add_argument("--iq_dirs",        nargs="*", default=None,
                   help="额外的 .iq 文件搜索目录（文件不在 xml 同目录时使用）")

    args = p.parse_args()

    # 收集文件路径：--files / --filelist / --scan_dir 可任意组合使用
    meta_paths = list(args.files or [])

    if args.filelist:
        with open(args.filelist, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    meta_paths.append(line)

    if args.scan_dir:
        for root_dir in args.scan_dir:
            found = scan_leaf_dirs(root_dir)
            log.info(f"扫描 {root_dir} → 找到 {len(found)} 个叶子目录，共 {len(found)} 个 XML")
            meta_paths.extend(found)

    if not meta_paths:
        p.error("请通过 --files、--filelist 或 --scan_dir 提供至少一个文件路径")

    batch_generate(
        meta_paths      = meta_paths,
        output_dir      = args.output_dir,
        images_per_file = args.images_per_file,
        img_size        = tuple(args.img_size),
        nfft            = args.nfft,
        overlap         = args.overlap,
        colormap        = args.colormap,
        iq_search_dirs  = args.iq_dirs,
    )


if __name__ == "__main__":
    main()