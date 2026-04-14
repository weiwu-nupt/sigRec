"""
开集信号体制识别 —— 背景类训练法
═══════════════════════════════════════════════════════════════
核心思路：
  训练时直接用 6 个类：4G / 5G / DVB / GSM / WiFi / Unknown
  其中 Unknown 类由训练集里的 TETRA + DMR 图片组成
  测试时直接看 softmax 输出，预测为 Unknown 类就是开集样本

目录结构：
  SigPic/
    train/
      4G/  5G/  DVB/  GSM/  WiFi/   ← 5 个已知类
      DMR/  TETRA/                   ← 这两个在训练时合并为 Unknown 类
    test/
      4G/  5G/  DVB/  GSM/  WiFi/   ← 已知类测试
      DMR/  TETRA/                   ← 未知类测试（标签统一为 Unknown）

优点：
  - 网络训练时就见过 Unknown 长什么样，判决边界更准
  - 不需要阈值调参，直接 softmax argmax 输出结果
  - GSM/WiFi/DVB 等容易误判的类会显著改善
═══════════════════════════════════════════════════════════════
"""

import os, time, copy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms, models
from PIL import Image

from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
TRAIN_DIR    = r"C:\Users\weiwu\Desktop\SigPic\train"
TEST_DIR     = r"C:\Users\weiwu\Desktop\SigPic\test"
OUTPUT_DIR   = r"C:\Users\weiwu\Desktop\SigPic_output_openset"
MODEL_SAVE   = os.path.join(OUTPUT_DIR, "best_resnet_openset.pth")

# 已知类（字母序，需与文件夹名完全一致）
KNOWN_CLASSES   = ["4G", "5G", "DVB", "GSM", "WiFi"]
# 训练集里作为 Unknown 类的文件夹（合并为一个 Unknown 类）
UNKNOWN_TRAIN   = ["DMR", "TETRA"]
# 测试集里同样作为 Unknown 的文件夹
UNKNOWN_TEST    = ["DMR", "TETRA"]

NUM_KNOWN    = len(KNOWN_CLASSES)       # 5
NUM_CLASSES  = NUM_KNOWN + 1            # 6（加上 Unknown）
UNKNOWN_LABEL = NUM_KNOWN               # Unknown 的标签 = 5

IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
SEED         = 42
NUM_WORKERS  = 0


# ═══════════════════════════════════════════════════════════════
# 数据集：训练集（6类）
# ═══════════════════════════════════════════════════════════════
class SixClassDataset(torch.utils.data.Dataset):
    """
    读取训练目录：
      已知类文件夹 → 标签 0~4
      UNKNOWN_TRAIN 里的文件夹 → 统一标签 5（Unknown）
    """
    def __init__(self, root, known_classes, unknown_folders, transform):
        self.transform = transform
        self.samples   = []   # [(path, label)]

        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder in known_classes:
                label = known_classes.index(folder)
            elif folder in unknown_folders:
                label = UNKNOWN_LABEL
            else:
                print(f"  [警告] 忽略未预期文件夹: {folder}")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                    self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


# ═══════════════════════════════════════════════════════════════
# 数据集：测试集（7类真实标签 → 合并为6类评估）
# ═══════════════════════════════════════════════════════════════
class OpenSetTestDataset(torch.utils.data.Dataset):
    """
    读取测试目录：
      已知类 → 标签 0~4
      DMR/TETRA → 统一标签 5（Unknown）
    同时保留原始类名用于 t-SNE 细粒度着色。
    """
    def __init__(self, root, known_classes, unknown_folders, transform):
        self.transform  = transform
        self.samples    = []   # [(path, label)]
        self.fine_names = []   # 每个样本的真实类名

        for folder in sorted(os.listdir(root)):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder in known_classes:
                label = known_classes.index(folder)
            elif folder in unknown_folders:
                label = UNKNOWN_LABEL
            else:
                print(f"  [警告] 测试集忽略未预期文件夹: {folder}")
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                    self.samples.append((os.path.join(folder_path, fname), label))
                    self.fine_names.append(folder)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


# ═══════════════════════════════════════════════════════════════
# val 子集工具
# ═══════════════════════════════════════════════════════════════
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        return self.transform(Image.open(path).convert("RGB")), label


# ═══════════════════════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════════════════════
def build_resnet(num_classes: int) -> nn.Module:
    model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feat_dim = 2048
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat_dim, num_classes))
    return model


# ═══════════════════════════════════════════════════════════════
# 训练 / 验证 epoch
# ═══════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    is_train = (phase == "train")
    model.train() if is_train else model.eval()
    total_loss = total_correct = total = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss    += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total         += imgs.size(0)

    return total_loss / total, total_correct / total


# ═══════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════
DARK_BG = "#0f1117"
DARK_AX = "#1a1d27"
# 已知类5色 + Unknown红色 + 细粒度未知类橙色
PAL = ["#58a6ff","#f78166","#3fb950","#d29922","#bc8cff","#ff4444","#ffa040"]

def _style(ax):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.12, color="#30363d")


def save_loss_curve(history, path):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes: _style(ax)

    axes[0].plot(ep, history["train_loss"], color="#58a6ff", lw=2, label="Train")
    axes[0].plot(ep, history["val_loss"],   color="#f78166", lw=2, label="Val")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor="#21262d", labelcolor="#c9d1d9")

    axes[1].plot(ep, [a*100 for a in history["train_acc"]], color="#3fb950", lw=2, label="Train")
    axes[1].plot(ep, [a*100 for a in history["val_acc"]],   color="#d29922", lw=2, label="Val")
    axes[1].set_title("Accuracy Curve (6-class incl. Unknown)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc (%)")
    axes[1].legend(facecolor="#21262d", labelcolor="#c9d1d9")
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()


def save_confusion(y_true, y_pred, names, path, title="Confusion Matrix"):
    labels_range = list(range(len(names)))
    cm      = confusion_matrix(y_true, y_pred, labels=labels_range)
    cm_norm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-9)
    annot   = np.array([[f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]})"
                         for j in range(len(names))]
                        for i in range(len(names))])
    sz = max(7, len(names))
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=names, yticklabels=names,
                ax=ax, linewidths=0.5, vmin=0, vmax=1,
                annot_kws={"fontsize": 8})
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_tsne(feats, labels, names, path, title="t-SNE"):
    perp = min(30, len(feats) - 1)
    print(f"  t-SNE 降维中 (n={len(feats)}, perplexity={perp})...")
    xy = TSNE(2, perplexity=perp, max_iter=1000,
              random_state=SEED, init="pca", learning_rate="auto").fit_transform(feats)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG); _style(ax)
    for idx, name in enumerate(names):
        mask = labels == idx
        if mask.sum() == 0: continue
        # 已知类用圆点，未知类用三角
        marker = "^" if idx >= NUM_KNOWN else "o"
        ax.scatter(xy[mask,0], xy[mask,1], c=PAL[idx % len(PAL)],
                   label=name, s=22, alpha=0.75, linewidths=0, marker=marker)
    ax.legend(title="Signal", facecolor="#21262d", labelcolor="#c9d1d9",
              title_fontsize=9, fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=13, pad=10, color="#e6edf3")
    ax.set_xlabel("t-SNE Dim 1", color="#8b949e")
    ax.set_ylabel("t-SNE Dim 2", color="#8b949e")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备  : {DEVICE}")
    print(f"训练类别  : {KNOWN_CLASSES} + Unknown（{UNKNOWN_TRAIN}）")

    # ── Transform ────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # ── 训练集（6类）─────────────────────────────────────────
    full_train = SixClassDataset(TRAIN_DIR, KNOWN_CLASSES, UNKNOWN_TRAIN, train_tf)
    counts     = np.bincount([s[1] for s in full_train.samples], minlength=NUM_CLASSES)
    class_display = KNOWN_CLASSES + ["Unknown"]
    print("训练集各类样本数:")
    for name, cnt in zip(class_display, counts):
        print(f"  {name:10s}: {cnt}")

    # train / val 划分
    n_val   = int(len(full_train) * VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_raw, val_raw = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_set = TransformSubset(val_raw, eval_tf)

    # WeightedRandomSampler 处理类别不平衡
    train_labels  = [full_train.samples[i][1] for i in train_raw.indices]
    cls_counts    = np.bincount(train_labels, minlength=NUM_CLASSES)
    cls_w         = 1.0 / (cls_counts + 1e-9)
    sample_w      = [cls_w[l] for l in train_labels]
    sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_raw, BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── 测试集（6类标签：已知0~4，Unknown=5）──────────────────
    test_dataset = OpenSetTestDataset(TEST_DIR, KNOWN_CLASSES, UNKNOWN_TEST, eval_tf)
    test_loader  = DataLoader(test_dataset, BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_labels  = np.array([s[1] for s in test_dataset.samples])
    n_k = (test_labels < NUM_KNOWN).sum()
    n_u = (test_labels == UNKNOWN_LABEL).sum()
    print(f"\n测试集: 已知 {n_k}  未知 {n_u}")

    # ── 模型：ResNet-50，输出6类 ──────────────────────────────
    model     = build_resnet(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── 训练循环 ──────────────────────────────────────────────
    history      = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())

    print("\n开始训练（6类：5已知 + Unknown）...\n" + "="*60)
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      DEVICE, "val")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        tag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, MODEL_SAVE)
            tag = " ← best"

        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  |  "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%  "
              f"({time.time()-t0:.1f}s){tag}")

    print(f"\n最佳验证准确率: {best_val_acc*100:.2f}%")
    model.load_state_dict(best_state)
    model.eval()

    # ── Loss 曲线 ─────────────────────────────────────────────
    save_loss_curve(history, os.path.join(OUTPUT_DIR, "01_loss_curve.png"))
    print(f"Loss 曲线已保存")

    # ── 测试集推理（直接 softmax argmax）─────────────────────
    print("\n开始测试...")
    all_preds = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
    all_preds = np.array(all_preds)

    # ── 分类报告 ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("测试集分类报告（含 Unknown 类）:")
    print(classification_report(test_labels, all_preds,
                                labels=list(range(NUM_CLASSES)),
                                target_names=class_display,
                                zero_division=0))

    # TKR / TUR / KP
    is_k_gt   = test_labels < NUM_KNOWN
    is_u_gt   = test_labels == UNKNOWN_LABEL
    is_k_pred = all_preds   < NUM_KNOWN
    is_u_pred = all_preds   == UNKNOWN_LABEL

    TK = int((is_k_gt & is_k_pred).sum())
    TU = int((is_u_gt & is_u_pred).sum())
    FK = int((is_u_gt & is_k_pred).sum())
    FU = int((is_k_gt & is_u_pred).sum())
    CK = int(((all_preds == test_labels) & is_k_gt).sum())

    TKR = TK / (TK + FU + 1e-9)
    TUR = TU / (TU + FK + 1e-9)
    KP  = CK / (TK + FK + 1e-9)
    print(f"TKR (True Known Rate)   : {TKR*100:.2f}%")
    print(f"TUR (True Unknown Rate) : {TUR*100:.2f}%")
    print(f"KP  (Known Precision)   : {KP*100:.2f}%")
    print(f"TK={TK}  TU={TU}  FK={FK}  FU={FU}  CK={CK}")

    # ── 混淆矩阵 ──────────────────────────────────────────────
    save_confusion(test_labels, all_preds, class_display,
                   os.path.join(OUTPUT_DIR, "02_confusion_matrix.png"),
                   "Open-Set Confusion Matrix (Row-Normalized)")
    print(f"\n混淆矩阵已保存")

    # ── t-SNE（细粒度：DMR和TETRA分开着色）───────────────────
    print("\n绘制 t-SNE...")
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    feature_extractor.eval()
    feats_all = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            f = feature_extractor(imgs.to(DEVICE)).flatten(1).cpu().numpy()
            feats_all.append(f)
    feats_all = np.concatenate(feats_all, 0)

    # 细粒度标签：已知类0~4，DMR=5，TETRA=6
    fine_names  = KNOWN_CLASSES + UNKNOWN_TEST   # ["4G","5G","DVB","GSM","WiFi","DMR","TETRA"]
    fine_labels = []
    for name in test_dataset.fine_names:
        fine_labels.append(fine_names.index(name))
    fine_labels = np.array(fine_labels)

    save_tsne(feats_all, fine_labels, fine_names,
              os.path.join(OUTPUT_DIR, "03_tsne_all.png"),
              "t-SNE: Known (●) & Unknown (▲) | ResNet-50")

    # 仅已知类
    km = fine_labels < NUM_KNOWN
    save_tsne(feats_all[km], fine_labels[km], KNOWN_CLASSES,
              os.path.join(OUTPUT_DIR, "04_tsne_known_only.png"),
              "t-SNE: Known Classes Only")

    print("\n✅ 全部完成！输出目录:", OUTPUT_DIR)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")