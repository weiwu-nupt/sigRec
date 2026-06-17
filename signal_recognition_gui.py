"""
signal_recognition_gui_qt6.py
=============================
频谱仪黑色风格 · 信号体制识别演示界面 (PyQt6)

功能：
  1. 浏览选择 IQ 数据文件 (.iqh/.iqb 或 SignalHound .xml/.iq)
  2. 频谱显示（功率谱密度）
  3. 一键生成时频图（整段数据中的一段 10ms 切片，真实 STFT）
  4. 导入训练好的 ResNet 权重 (.pth)，真正用 PyTorch 前向推理
  5. 输出信号类型 (4G/5G/DVB/GSM/WiFi) 及各类概率

依赖：
  pip install PyQt6 numpy matplotlib torch torchvision pillow

复用（可选，放同目录即自动复用）：
  signal_tfmap_generator.py   —— IQMeta / detect_format / generate_spectrogram / auto_nfft
  iqhb_parser.py              —— .iqh/.iqb 解析插件

若上述脚本不在同目录，本程序内置等价的轻量实现，可独立运行。

运行：
  python signal_recognition_gui_qt6.py
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np

# ── PyQt6 ──
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal, QThread

# ── Matplotlib 嵌入 ──
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _fill_axes(fig, left=0.08, right=0.985, top=0.88, bottom=0.20):
    """让 axes 撑满整个 figure（留出标签边距）。
    比 constrained_layout 在 Qt 后端更稳定地填满画布、随窗口伸缩。
    """
    try:
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    except Exception:
        pass

# ─────────────────────────────────────────────
# 尝试复用用户已有脚本（同目录）
# ─────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import signal_tfmap_generator as STG          # 复用解析 + STFT
if hasattr(STG, "_load_plugins"):
    try:
        STG._load_plugins()
    except Exception:
        pass
detect_format       = STG.detect_format
generate_spectrogram = STG.generate_spectrogram
auto_nfft           = STG.auto_nfft
IQMeta              = STG.IQMeta
_BACKEND = "signal_tfmap_generator"

# ═════════════════════════════════════════════
# 类别定义
# ═════════════════════════════════════════════
CLASS_NAMES = ["4G", "5G", "DVB", "GSM", "WiFi"]
CLASS_COLOR = {"4G": "#ffb454", "5G": "#ff5d9e", "DVB": "#33ff9d",
               "GSM": "#bc8cff", "WiFi": "#39d0ff"}

# 频谱仪配色
C_BG      = "#05070a"
C_PANEL   = "#0a0e14"
C_PANEL2  = "#0d1219"
C_LINE    = "#1c2b3f"
C_GRID    = "#142235"
C_PHOS    = "#33ff9d"
C_PHOSDIM = "#1c8a57"
C_AMBER   = "#ffb454"
C_TXT     = "#a9c4d8"
C_TXTDIM  = "#5a7388"
C_TXTBRT  = "#dceaf3"
C_WARN    = "#ff5d5d"


# ═════════════════════════════════════════════
# PyTorch 模型加载与推理（后台线程）
# ═════════════════════════════════════════════
def build_resnet(num_classes=5, backbone="resnet50"):
    """与 signal_classify_resnet.py 中 build_model 结构一致，便于直接加载权重。"""
    import torch.nn as nn
    from torchvision import models
    if backbone == "resnet18":
        m = models.resnet18(weights=None); feat = 512
    elif backbone == "resnet34":
        m = models.resnet34(weights=None); feat = 512
    else:
        m = models.resnet50(weights=None); feat = 2048
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat, num_classes))
    return m


class InferenceEngine:
    """封装 .pth 加载 + 时频图 → 概率向量 的真实推理。"""

    def __init__(self):
        self.model = None
        self.device = None
        self.backbone = "resnet50"
        self.img_size = 224

    def load(self, pth_path):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(pth_path, map_location=self.device)
        # 支持 {'state_dict':...} 或纯 state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        # 自动探测 backbone：看 fc 前一层权重维度
        backbone = "resnet50"
        for k, v in state.items():
            if k.endswith("fc.1.weight") or k.endswith("fc.weight"):
                in_feat = v.shape[1]
                backbone = {512: "resnet18", 2048: "resnet50"}.get(in_feat, "resnet50")
                break
        self.backbone = backbone

        model = build_resnet(len(CLASS_NAMES), backbone)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        self.model = model
        return self.device.type, backbone, len(missing), len(unexpected)

    def infer(self, rgb_uint8):
        """rgb_uint8: (H,W,3) uint8 时频图 → softmax 概率 dict"""
        import torch
        from torchvision import transforms
        from PIL import Image
        tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = Image.fromarray(rgb_uint8).convert("RGB")
        x = tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return {c: float(probs[i]) for i, c in enumerate(CLASS_NAMES)}


class InferWorker(QThread):
    """后台推理线程，避免界面卡顿。"""
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, engine, rgb):
        super().__init__()
        self.engine = engine
        self.rgb = rgb

    def run(self):
        try:
            probs = self.engine.infer(self.rgb)
            self.done.emit(probs)
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


# ═════════════════════════════════════════════
# 自定义控件：磷光概率条
# ═════════════════════════════════════════════
class ProbBar(QtWidgets.QWidget):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.value = 0.0
        self.target = 0.0
        self.lead = False
        self.setFixedHeight(40)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._anim)

    def set_value(self, v, lead=False):
        self.target = v
        self.lead = lead
        self._timer.start(16)

    def _anim(self):
        self.value += (self.target - self.value) * 0.18
        if abs(self.target - self.value) < 0.001:
            self.value = self.target
            self._timer.stop()
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        # 文本
        col = C_AMBER if self.lead else C_TXT
        p.setPen(QtGui.QColor(col))
        f = QtGui.QFont("Consolas", 10); p.setFont(f)
        p.drawText(0, 0, w, 16, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.name)
        pct = f"{self.value*100:.1f}%"
        p.setPen(QtGui.QColor(C_AMBER if self.lead else C_PHOS))
        p.drawText(0, 0, w, 16, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, pct)
        # 条底
        by = 24
        p.setBrush(QtGui.QColor("#0e1826")); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, by, w, 7, 3, 3)
        # 条填充
        fw = int(w * self.value)
        if fw > 0:
            grad = QtGui.QLinearGradient(0, 0, w, 0)
            if self.lead:
                grad.setColorAt(0, QtGui.QColor(C_AMBER))
                grad.setColorAt(1, QtGui.QColor("#ffd486"))
            else:
                c = QtGui.QColor(CLASS_COLOR.get(self.name, C_PHOS))
                c2 = QtGui.QColor(c); c2.setAlpha(120)
                grad.setColorAt(0, c2); grad.setColorAt(1, c)
            p.setBrush(grad)
            p.drawRoundedRect(0, by, fw, 7, 3, 3)


# ═════════════════════════════════════════════
# 主窗口
# ═════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SPECTRA-NET · 信号体制识别终端")
        self.resize(1380, 880)

        # 状态
        self.meta = None
        self.parser = None
        self.iq_dirs = []
        self.iq_slice = None        # 当前 10ms 切片 complex64
        self.tf_rgb = None          # 时频图 RGB uint8
        self._last_tf = None        # 缓存最近一次时频图数据，供 resize 重绘
        self.start_sample = 0
        self.engine = InferenceEngine()
        self.model_loaded = False
        self.worker = None

        self._build_ui()
        self._apply_style()
        self.log(f"终端初始化完成 · 解析后端: {_BACKEND}", "ok")
        try:
            import torch  # noqa
            self.log("PyTorch 可用 · 推理引擎就绪", "ok")
        except Exception:
            self.log("未检测到 PyTorch，导入模型前请先 pip install torch torchvision", "warn")

    # ── 构建界面 ──
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        root.addWidget(self._build_header())

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left(), 0)
        body.addWidget(self._build_center(), 1)
        body.addWidget(self._build_right(), 0)
        root.addLayout(body, 1)

    def _panel(self, title):
        box = QtWidgets.QFrame()
        box.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QtWidgets.QLabel("● " + title)
        head.setObjectName("phead")
        lay.addWidget(head)
        inner = QtWidgets.QWidget()
        inner.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                            QtWidgets.QSizePolicy.Policy.Expanding)
        ilay = QtWidgets.QVBoxLayout(inner)
        ilay.setContentsMargins(13, 11, 13, 13)
        ilay.setSpacing(11)
        lay.addWidget(inner, 1)
        return box, ilay

    def _build_header(self):
        bar = QtWidgets.QFrame(); bar.setObjectName("header")
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(18, 10, 18, 10)
        title = QtWidgets.QLabel("SPECTRA-NET")
        title.setObjectName("logo")
        sub = QtWidgets.QLabel("  信号体制识别终端")
        sub.setObjectName("logosub")
        lay.addWidget(title); lay.addWidget(sub)
        lay.addStretch(1)

        def stat(k):
            w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(14, 0, 14, 0); v.setSpacing(1)
            kk = QtWidgets.QLabel(k); kk.setObjectName("statk")
            vv = QtWidgets.QLabel("—"); vv.setObjectName("statv")
            v.addWidget(kk); v.addWidget(vv)
            return w, vv

        wfs, self.hdrFs = stat("采样率 Fs")
        wfc, self.hdrFc = stat("中心频率 Fc")
        wst, self.hdrStat = stat("状态"); self.hdrStat.setText("● READY")
        self.hdrStat.setStyleSheet(f"color:{C_PHOS}")
        lay.addWidget(wfs); lay.addWidget(wfc); lay.addWidget(wst)
        return bar

    def _build_left(self):
        wrap = QtWidgets.QWidget(); wrap.setFixedWidth(300)
        lay = QtWidgets.QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)

        # 数据源
        box, il = self._panel("数据源 · IQ SOURCE")
        self.btnBrowse = QtWidgets.QPushButton("⊟  浏览选择 IQ 文件")
        self.btnBrowse.setObjectName("filebtn")
        self.btnBrowse.clicked.connect(self.on_browse_iq)
        il.addWidget(self.btnBrowse)
        self.lblFile = QtWidgets.QLabel("支持 .iqh+.iqb / SignalHound .xml")
        self.lblFile.setObjectName("hint"); self.lblFile.setWordWrap(True)
        il.addWidget(self.lblFile)

        self.metaForm = QtWidgets.QFormLayout()
        self.metaForm.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.metaForm.setHorizontalSpacing(10); self.metaForm.setVerticalSpacing(5)
        self.mv = {}
        for k in ["采样率", "中心频率", "采样点数", "时长", "数据类型"]:
            lk = QtWidgets.QLabel(k); lk.setObjectName("mk")
            lv = QtWidgets.QLabel("—"); lv.setObjectName("mvv")
            lv.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.metaForm.addRow(lk, lv); self.mv[k] = lv
        mw = QtWidgets.QWidget(); mw.setLayout(self.metaForm)
        il.addWidget(mw)
        lay.addWidget(box)

        # 采集参数
        box2, il2 = self._panel("采集参数 · ACQUISITION")
        il2.addWidget(self._small_label("10 ms 切片起点"))
        srow = QtWidgets.QHBoxLayout()
        self.startSld = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.startSld.setRange(0, 1000); self.startSld.setValue(420)
        self.startSld.setEnabled(False)
        self.startSld.valueChanged.connect(self.on_slide)
        self.lblStart = QtWidgets.QLabel("— ms"); self.lblStart.setObjectName("amberval")
        self.lblStart.setMinimumWidth(70); self.lblStart.setAlignment(Qt.AlignmentFlag.AlignRight)
        srow.addWidget(self.startSld); srow.addWidget(self.lblStart)
        il2.addLayout(srow)

        il2.addWidget(self._small_label("STFT 窗长 nFFT"))
        self.cmbNfft = QtWidgets.QComboBox()
        self.cmbNfft.addItems(["自适应 (推荐)", "256", "512", "1024", "2048"])
        il2.addWidget(self.cmbNfft)

        il2.addWidget(self._small_label("色彩映射 Colormap"))
        self.cmbCmap = QtWidgets.QComboBox()
        self.cmbCmap.addItems(["viridis", "magma", "inferno", "plasma", "turbo"])
        il2.addWidget(self.cmbCmap)
        lay.addWidget(box2)

        # 模型
        box3, il3 = self._panel("神经网络 · MODEL")
        self.btnModel = QtWidgets.QPushButton("◈  导入网络权重 (.pth)")
        self.btnModel.setObjectName("filebtn")
        self.btnModel.clicked.connect(self.on_load_model)
        il3.addWidget(self.btnModel)
        self.lblModel = QtWidgets.QLabel("ResNet · 5 类 4G/5G/DVB/GSM/WiFi")
        self.lblModel.setObjectName("hint"); self.lblModel.setWordWrap(True)
        il3.addWidget(self.lblModel)
        lay.addWidget(box3)

        lay.addStretch(1)
        return wrap

    def _build_center(self):
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)

        box, il = self._panel("实时显示 · DISPLAY")

        _EXPAND = QtWidgets.QSizePolicy.Policy.Expanding

        # 频谱图：尺寸由 Qt 后端自动同步到画布；axes 用固定边距撑满 figure
        self.figSpec = Figure(facecolor=C_BG)
        self.canvasSpec = FigureCanvas(self.figSpec)
        self.canvasSpec.setMinimumHeight(150)
        self.canvasSpec.setSizePolicy(_EXPAND, _EXPAND)
        self.axSpec = self.figSpec.add_subplot(111)
        self._style_ax(self.axSpec, "SPECTRUM 频谱")
        _fill_axes(self.figSpec)
        il.addWidget(self.canvasSpec, 2)   # 频谱占 2 份高度

        # 时频图
        self.figTF = Figure(facecolor=C_BG)
        self.canvasTF = FigureCanvas(self.figTF)
        self.canvasTF.setMinimumHeight(200)
        self.canvasTF.setSizePolicy(_EXPAND, _EXPAND)
        self.axTF = self.figTF.add_subplot(111)
        self._style_ax(self.axTF, "SPECTROGRAM 时频图 · 10 ms")
        _fill_axes(self.figTF)
        il.addWidget(self.canvasTF, 3)     # 时频图占 3 份高度

        # 按键（固定高度，不抢占画布空间）
        brow = QtWidgets.QHBoxLayout()
        self.btnGen = QtWidgets.QPushButton("⟿  生成时频图")
        self.btnGen.setObjectName("amberbtn"); self.btnGen.setEnabled(False)
        self.btnGen.setMinimumHeight(40)
        self.btnGen.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Fixed)
        self.btnGen.clicked.connect(self.on_generate)
        self.btnRun = QtWidgets.QPushButton("▶  运行识别")
        self.btnRun.setObjectName("phosbtn"); self.btnRun.setEnabled(False)
        self.btnRun.setMinimumHeight(40)
        self.btnRun.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                  QtWidgets.QSizePolicy.Policy.Fixed)
        self.btnRun.clicked.connect(self.on_run)
        brow.addWidget(self.btnGen); brow.addWidget(self.btnRun)
        il.addLayout(brow, 0)              # 拉伸权重 0：高度固定
        lay.addWidget(box, 1)

        # 控制台
        cbox, cil = self._panel("系统日志 · CONSOLE")
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setFixedHeight(120)
        cil.addWidget(self.console)
        lay.addWidget(cbox)
        return wrap

    def _build_right(self):
        wrap = QtWidgets.QWidget(); wrap.setFixedWidth(300)
        lay = QtWidgets.QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(12)

        box, il = self._panel("识别结果 · CLASSIFICATION")
        self.lblVerdict = QtWidgets.QLabel("———")
        self.lblVerdict.setObjectName("verdict")
        self.lblVerdict.setAlignment(Qt.AlignmentFlag.AlignCenter)
        il.addWidget(self.lblVerdict)
        self.lblConf = QtWidgets.QLabel("等待推理 · awaiting inference")
        self.lblConf.setObjectName("conf"); self.lblConf.setAlignment(Qt.AlignmentFlag.AlignCenter)
        il.addWidget(self.lblConf)

        div = QtWidgets.QFrame(); div.setObjectName("divider"); div.setFixedHeight(1)
        il.addWidget(div)

        self.bars = {}
        for c in CLASS_NAMES:
            b = ProbBar(c); self.bars[c] = b; il.addWidget(b)

        il.addStretch(1)
        self.lblRightHint = QtWidgets.QLabel(
            "流程：浏览 IQ → 生成时频图 → 导入 .pth → 运行识别。\n"
            "未导入模型时「运行识别」不可用。")
        self.lblRightHint.setObjectName("hint"); self.lblRightHint.setWordWrap(True)
        il.addWidget(self.lblRightHint)
        lay.addWidget(box)
        return wrap

    def _small_label(self, t):
        l = QtWidgets.QLabel(t); l.setObjectName("flabel"); return l

    def _style_ax(self, ax, title):
        ax.set_facecolor("#03060a")
        for s in ax.spines.values():
            s.set_color(C_LINE)
        ax.tick_params(colors=C_TXTDIM, labelsize=7)
        ax.set_title(title, color=C_PHOS, fontsize=8, loc="left", pad=4,
                     fontfamily="Microsoft YaHei")
        ax.grid(True, color=C_GRID, alpha=0.5, linewidth=0.5)

    # ════════════ 业务逻辑 ════════════
    def log(self, msg, kind="info"):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        color = {"ok": C_PHOS, "info": C_TXT, "warn": C_AMBER, "err": C_WARN}[kind]
        self.console.appendHtml(
            f'<span style="color:{C_TXTDIM}">{ts}</span> '
            f'<span style="color:{color}">{msg}</span>')
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum())

    @staticmethod
    def _fmt_hz(v):
        if v >= 1e9: return f"{v/1e9:.3f} GHz"
        if v >= 1e6: return f"{v/1e6:.3f} MHz"
        if v >= 1e3: return f"{v/1e3:.2f} kHz"
        return f"{v:.0f} Hz"

    def on_browse_iq(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 IQ 数据文件", "",
            "IQ 数据 (*.iq *.iqb *.bin);;所有文件 (*.*)")
        if not path:
            return

        iq_path = path
        iq_dir = Path(path).parent
        stem = Path(path).stem

        # 在同目录自动定位头文件：先同名 .xml / .iqh，再退而求其次扫描目录
        header = None
        for ext in (".xml", ".iqh"):
            cand = iq_dir / (stem + ext)
            if cand.is_file():
                header = str(cand); break
        if header is None:
            # 同名找不到：扫描目录里唯一的 .xml 或 .iqh
            for ext in (".xml", ".iqh"):
                hits = sorted(iq_dir.glob("*" + ext))
                if len(hits) == 1:
                    header = str(hits[0]); break
                elif len(hits) > 1:
                    self.log(f"同目录有多个 {ext} 头文件，请手动选择", "warn")
                    header, _ = QtWidgets.QFileDialog.getOpenFileName(
                        self, "选择对应的头文件", str(iq_dir),
                        f"头文件 (*{ext});;所有文件 (*.*)")
                    break
        if not header:
            self.log("未找到同目录头文件 (.xml/.iqh)", "err")
            QtWidgets.QMessageBox.warning(
                self, "缺少头文件",
                f"未在同目录找到 {stem}.xml 或 {stem}.iqh。\n"
                f"请确保头文件与 .iq 数据文件在同一目录。")
            return

        try:
            self.parser = detect_format(header)
            self.meta = self.parser.parse(header)
        except Exception as e:
            self.log(f"头文件解析失败: {e}", "err")
            QtWidgets.QMessageBox.critical(self, "解析失败", str(e))
            return

        # 用用户实际选中的 .iq 覆盖头文件里记录的路径（以实际为准）
        self.meta.iq_file_path = iq_path
        self.iq_dirs = [str(iq_dir)]
        self.meta_path = iq_path           # 显示用：.iq 文件名
        self.header_path = header
        self.log(f"载入: {os.path.basename(iq_path)} · 头文件: {os.path.basename(header)}", "info")
        self._apply_meta()

        # 直接读首个 10ms 切片并画频谱
        try:
            self._load_slice()
            self._draw_spectrum()
            self.btnGen.setEnabled(True)
            self.log(f"元数据就绪 · Fs={self._fmt_hz(self.meta.sample_rate)} · "
                     f"{self.meta.sample_count:,} 点 · {self.meta.duration_s*1000:.1f} ms", "ok")
        except FileNotFoundError as e:
            self.log(f"IQ 数据读取失败: {e}", "err")
            QtWidgets.QMessageBox.warning(
                self, "IQ 文件读取失败", f"{e}")
        except Exception as e:
            self.log(f"读取切片失败: {e}", "err")

    def _apply_meta(self):
        m = self.meta
        self.lblFile.setText(os.path.basename(self.meta_path))
        self.mv["采样率"].setText(self._fmt_hz(m.sample_rate))
        self.mv["中心频率"].setText(self._fmt_hz(m.center_freq))
        self.mv["采样点数"].setText(f"{m.sample_count:,}")
        self.mv["时长"].setText(f"{m.duration_s*1000:.1f} ms")
        self.mv["数据类型"].setText(m.data_type)
        self.hdrFs.setText(self._fmt_hz(m.sample_rate))
        self.hdrFc.setText(self._fmt_hz(m.center_freq))

        n10 = m.samples_per_10ms
        max_start = max(0, m.sample_count - n10)
        self.startSld.setEnabled(max_start > 0)
        self.start_sample = int(max_start * 0.42)
        self.startSld.blockSignals(True)
        self.startSld.setValue(420)
        self.startSld.blockSignals(False)
        self._update_start_label()

    def _update_start_label(self):
        if not self.meta:
            return
        ms = self.start_sample / self.meta.sample_rate * 1000
        self.lblStart.setText(f"{ms:.1f} ms")

    def on_slide(self, val):
        if not self.meta:
            return
        n10 = self.meta.samples_per_10ms
        max_start = max(0, self.meta.sample_count - n10)
        self.start_sample = int(max_start * val / 1000)
        self._update_start_label()
        try:
            self._load_slice()
            self._draw_spectrum()
        except Exception as e:
            self.log(f"切片更新失败: {e}", "warn")

    def _load_slice(self):
        """真实读取 10ms 二进制样本。"""
        n10 = self.meta.samples_per_10ms
        self.iq_slice = self.parser.read_samples(
            self.meta, self.start_sample, n10, self.iq_dirs)
        if self.iq_slice is None or len(self.iq_slice) < 16:
            raise IOError("读取到的样本数过少")

    def _draw_spectrum(self):
        iq = self.iq_slice
        fs = self.meta.sample_rate
        nfft = 1024 if len(iq) >= 1024 else 256
        # Welch 式平均
        hop = nfft
        n_seg = max(1, len(iq) // nfft)
        acc = np.zeros(nfft)
        win = np.hanning(nfft)
        for i in range(n_seg):
            seg = iq[i*nfft:(i+1)*nfft]
            if len(seg) < nfft:
                break
            sp = np.fft.fftshift(np.fft.fft(seg * win))
            acc += np.abs(sp) ** 2
        acc /= n_seg
        pdb = 10 * np.log10(acc + 1e-12)
        f = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0/fs)) / 1e6  # MHz

        ax = self.axSpec
        ax.clear()
        self._style_ax(ax, "SPECTRUM 频谱")
        ax.fill_between(f, pdb, pdb.min()-5, color=C_PHOS, alpha=0.18)
        ax.plot(f, pdb, color=C_PHOS, linewidth=1.0)
        ax.axvline(0, color=C_AMBER, alpha=0.4, linewidth=0.8, linestyle="--")
        ax.set_xlabel("Freq offset (MHz)", color=C_TXTDIM, fontsize=7)
        ax.set_ylabel("PSD (dB)", color=C_TXTDIM, fontsize=7)
        ax.set_xlim(f[0], f[-1])
        self.canvasSpec.draw_idle()

    # ── 生成时频图 ──
    def on_generate(self):
        if self.iq_slice is None:
            return
        self.btnGen.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        try:
            fs = self.meta.sample_rate
            sel = self.cmbNfft.currentText()
            nfft = auto_nfft(fs) if sel.startswith("自适应") else int(sel)
            if sel.startswith("自适应"):
                self.log(f"自适应 nFFT = {nfft} (Fs {fs/1e6:.2f} MHz)", "info")
            nfft = min(nfft, 1024)

            t, fa, pdb = generate_spectrogram(self.iq_slice, fs, nfft=nfft, overlap=0.75)
            self._render_tf(t, fa, pdb)
            self.log(f"时频图就绪 · {pdb.shape[1]} 帧 × {pdb.shape[0]} bins", "ok")
            self.btnRun.setEnabled(self.model_loaded)
            if not self.model_loaded:
                self.log("提示: 导入 .pth 模型后即可运行识别", "warn")
        except Exception as e:
            self.log(f"生成时频图失败: {e}", "err")
        finally:
            self.btnGen.setEnabled(True)

    def _render_tf(self, t, f, pdb):
        self._last_tf = (t, f, pdb)
        ax = self.axTF
        ax.clear()
        self._style_ax(ax, "SPECTROGRAM 时频图 · 10 ms")
        cmap = self.cmbCmap.currentText()
        vmin = np.percentile(pdb, 5); vmax = np.percentile(pdb, 99)
        extent = [t[0]*1000, t[-1]*1000, f[0]/1e6, f[-1]/1e6]
        im = ax.imshow(pdb, aspect="auto", origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, extent=extent,
                       interpolation="antialiased", resample=True)
        im.set_interpolation_stage("rgba")
        ax.set_xlabel("Time (ms)", color=C_TXTDIM, fontsize=7)
        ax.set_ylabel("Freq (MHz)", color=C_TXTDIM, fontsize=7)
        self.canvasTF.draw_idle()

        # 渲染成 RGB uint8 供模型推理（无坐标、纯图，匹配训练数据风格）
        self.tf_rgb = self._tf_to_rgb(pdb, cmap)

    def _tf_to_rgb(self, pdb, cmap_name):
        """将功率谱渲染为 RGB uint8（与训练时的无标注 PNG 一致）。"""
        import matplotlib.cm as cm
        vmin = np.percentile(pdb, 1); vmax = np.percentile(pdb, 99)
        norm = np.clip((pdb - vmin) / (vmax - vmin + 1e-12), 0, 1)
        try:
            cmap = matplotlib.colormaps[cmap_name]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(cmap_name)
        rgba = cmap(norm[::-1])  # 频率向上
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    # ── 模型导入 ──
    def on_load_model(self):
        try:
            import torch  # noqa
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "缺少 PyTorch",
                "未安装 PyTorch。请运行:\n\npip install torch torchvision")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择模型权重", "", "PyTorch 权重 (*.pth *.pt);;所有文件 (*.*)")
        if not path:
            return
        try:
            self.log(f"加载模型: {os.path.basename(path)} …", "info")
            QtWidgets.QApplication.processEvents()
            dev, backbone, miss, unexp = self.engine.load(path)
            self.model_loaded = True
            self.lblModel.setText(f"✓ {os.path.basename(path)}\n{backbone} · {dev.upper()}")
            self.lblModel.setStyleSheet(f"color:{C_PHOS}")
            self.log(f"模型就绪 · {backbone} · 设备 {dev.upper()} · "
                     f"缺失 {miss} / 多余 {unexp} 键", "ok")
            if miss or unexp:
                self.log("键不完全匹配，已按 strict=False 加载（通常因 fc 结构差异）", "warn")
            if self.tf_rgb is not None:
                self.btnRun.setEnabled(True)
        except Exception as e:
            self.log(f"模型加载失败: {e}", "err")
            QtWidgets.QMessageBox.critical(self, "加载失败", str(e))

    # ── 运行识别 ──
    def on_run(self):
        if self.tf_rgb is None:
            self.log("请先生成时频图", "warn"); return
        if not self.model_loaded:
            self.log("请先导入模型", "warn"); return
        self.btnRun.setEnabled(False)
        self.lblVerdict.setText("···")
        self.lblConf.setText("前向传播 · forward pass")
        self.hdrStat.setText("● INFER"); self.hdrStat.setStyleSheet(f"color:{C_AMBER}")
        self.log("运行推理 (PyTorch forward) …", "info")

        self.worker = InferWorker(self.engine, self.tf_rgb)
        self.worker.done.connect(self._on_infer_done)
        self.worker.failed.connect(self._on_infer_fail)
        self.worker.start()

    def _on_infer_done(self, probs):
        order = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        for i, (c, p) in enumerate(order):
            self.bars[c].set_value(p, lead=(i == 0))
        best, bp = order[0]
        self.lblVerdict.setText(best)
        self.lblVerdict.setStyleSheet(
            f"color:{CLASS_COLOR[best]}; font-size:30px; font-weight:700; letter-spacing:4px;")
        self.lblConf.setText(f"置信度 {bp*100:.1f}% · imported model")
        self.hdrStat.setText("● READY"); self.hdrStat.setStyleSheet(f"color:{C_PHOS}")
        self.log(f"识别结果 ▸ {best}  ({bp*100:.1f}%)", "ok")
        self.btnRun.setEnabled(True)

    def _on_infer_fail(self, msg):
        self.log(f"推理失败: {msg.splitlines()[0]}", "err")
        self.hdrStat.setText("● ERROR"); self.hdrStat.setStyleSheet(f"color:{C_WARN}")
        QtWidgets.QMessageBox.critical(self, "推理失败", msg)
        self.btnRun.setEnabled(True)

    # ── 样式表 ──
    def _apply_style(self):
        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background:{C_BG}; color:{C_TXT};
            font-family:Consolas,'Courier New',monospace; font-size:12px;
        }}
        QFrame#header {{
            background:{C_PANEL2}; border:1px solid {C_LINE}; border-radius:6px;
        }}
        QLabel#logo {{ color:{C_TXTBRT}; font-size:16px; font-weight:bold; letter-spacing:2px; }}
        QLabel#logosub {{ color:{C_PHOS}; font-size:10px; letter-spacing:3px; }}
        QLabel#statk {{ color:{C_TXTDIM}; font-size:9px; letter-spacing:1px; }}
        QLabel#statv {{ color:{C_PHOS}; font-size:13px; }}
        QFrame#panel {{
            background:{C_PANEL2}; border:1px solid {C_LINE}; border-radius:6px;
        }}
        QLabel#phead {{
            color:{C_TXTDIM}; font-size:10px; letter-spacing:2px; padding:9px 13px;
            border-bottom:1px solid {C_LINE};
        }}
        QLabel#flabel {{ color:{C_TXTDIM}; font-size:9px; letter-spacing:1px; }}
        QLabel#mk {{ color:{C_TXTDIM}; font-size:11px; }}
        QLabel#mvv {{ color:{C_PHOS}; font-size:11px; }}
        QLabel#hint {{ color:{C_TXTDIM}; font-size:9px; }}
        QLabel#amberval {{ color:{C_AMBER}; font-size:11px; }}
        QLabel#verdict {{ color:{C_TXTDIM}; font-size:30px; font-weight:bold; letter-spacing:4px; padding:8px; }}
        QLabel#conf {{ color:{C_TXTDIM}; font-size:11px; padding-bottom:6px; }}
        QFrame#divider {{ background:{C_LINE}; }}
        QPushButton#filebtn {{
            background:{C_PANEL}; border:1px dashed {C_LINE}; border-radius:5px;
            color:{C_TXT}; padding:12px; font-size:11px;
        }}
        QPushButton#filebtn:hover {{ border-color:{C_PHOSDIM}; background:#08120d; color:{C_PHOS}; }}
        QPushButton#amberbtn {{
            background:#1f1708; border:1px solid #7a5418; border-radius:5px;
            color:{C_AMBER}; padding:11px; font-size:11px; letter-spacing:2px;
        }}
        QPushButton#amberbtn:hover:enabled {{ background:#2a1f0a; }}
        QPushButton#amberbtn:disabled {{ color:{C_TXTDIM}; border-color:{C_LINE}; background:{C_PANEL}; }}
        QPushButton#phosbtn {{
            background:#0c1f17; border:1px solid {C_PHOSDIM}; border-radius:5px;
            color:{C_PHOS}; padding:11px; font-size:11px; letter-spacing:2px;
        }}
        QPushButton#phosbtn:hover:enabled {{ background:#11301f; }}
        QPushButton#phosbtn:disabled {{ color:{C_TXTDIM}; border-color:{C_LINE}; background:{C_PANEL}; }}
        QComboBox {{
            background:{C_PANEL}; border:1px solid {C_LINE}; border-radius:5px;
            color:{C_TXT}; padding:6px 8px; font-size:11px;
        }}
        QComboBox:hover {{ border-color:{C_PHOSDIM}; }}
        QComboBox QAbstractItemView {{
            background:{C_PANEL}; color:{C_TXT}; selection-background-color:{C_PHOSDIM};
            border:1px solid {C_LINE};
        }}
        QSlider::groove:horizontal {{ height:3px; background:{C_LINE}; border-radius:2px; }}
        QSlider::handle:horizontal {{
            width:13px; height:13px; margin:-6px 0; border-radius:7px; background:{C_PHOS};
        }}
        QSlider::sub-page:horizontal {{ background:{C_PHOSDIM}; border-radius:2px; }}
        QPlainTextEdit#console {{
            background:{C_PANEL}; border:1px solid {C_LINE}; border-radius:5px;
            color:{C_TXT}; font-size:11px; padding:6px;
        }}
        QScrollBar:vertical {{ background:{C_PANEL}; width:8px; }}
        QScrollBar::handle:vertical {{ background:{C_LINE}; border-radius:4px; }}
        QMessageBox {{ background:{C_PANEL2}; }}
        """)


def main():
    # 高 DPI 支持：让控件在高分屏上锐利（PyQt6 默认开启 DPI 缩放，
    # 这里补充高清 pixmap，不放大图形尺寸）
    try:
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    # 暗色调色板
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(C_BG))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(C_PANEL))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(C_TXT))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(C_TXT))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()