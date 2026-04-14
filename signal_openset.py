"""
基于 ResNet-50 + S3R 开集识别
═══════════════════════════════════════════════════════════════
骨干网络：ResNet-50（与之前闭集训练相同）
开集方法：S3R 语义分类器（论文 IEEE TIFS 2024）

闭集（Known）：4G / 5G / WiFi / GSM / DVB
开集（Unknown）：TETRA / DMR  ← 仅出现在测试集

关键改动（相对于纯闭集脚本）：
1. 损失函数新增 中心损失(Lcen) + 聚类损失(Lclu)，使特征空间语义更紧凑
2. 测试阶段用马氏距离 + 3σ自适应阈值（S3R论文公式29）判决 Known/Unknown
3. 输出 TKR / TUR / KP、ROC曲线、开集混淆矩阵、t-SNE

目录结构：
  SigPic/
    train/  4G/ 5G/ DVB/ GSM/ WiFi/
    test/   4G/ 5G/ DVB/ GSM/ WiFi/ TETRA/ DMR/
═══════════════════════════════════════════════════════════════
"""

import os, time, copy, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms, models
from PIL import Image

from sklearn.manifold import TSNE
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import seaborn as sns

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 超参数配置
# ═══════════════════════════════════════════════════════════════
TRAIN_DIR    = r"C:\Users\weiwu\Desktop\SigPic\train"
TEST_DIR     = r"C:\Users\weiwu\Desktop\SigPic\test"
OUTPUT_DIR   = r"C:\Users\weiwu\Desktop\SigPic_output_openset"
MODEL_SAVE   = os.path.join(OUTPUT_DIR, "best_resnet_openset.pth")

KNOWN_CLASSES   = ["4G", "5G", "DVB", "GSM", "WiFi"]  # 必须与 ImageFolder 字母序一致
UNKNOWN_CLASSES = ["DMR", "TETRA"]
NUM_KNOWN       = len(KNOWN_CLASSES)   # 5

IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
SEED         = 42
NUM_WORKERS  = 0       # Windows 必须为 0

# S3R 损失权重（论文 Table II）
ETA1         = 5e-3    # 中心损失权重
ETA2         = 1.0     # 聚类损失权重
ETA3         = 0.1     # 交叉熵损失权重（主损失）
MARGIN       = 8.0     # 聚类损失 margin ϑ
CENTER_LR    = 1e-4    # 中心向量更新学习率
THRESHOLD_PERCENTILE = 99  # 阈值百分位（越大越宽松→更多判已知；越小越严格→更多判Unknown）


# ═══════════════════════════════════════════════════════════════
# 数据集工具
# ═══════════════════════════════════════════════════════════════
class TransformSubset(torch.utils.data.Dataset):
    """对 random_split Subset 重新应用 transform（val 无数据增强）。"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        return self.transform(Image.open(path).convert("RGB")), label


class OpenSetTestDataset(torch.utils.data.Dataset):
    """
    测试集：已知类标签 0~K-1，未知类（TETRA/DMR）统一标为 NUM_KNOWN。
    同时记录原始类名，方便 t-SNE 细粒度着色。
    """
    def __init__(self, root, known_classes, transform):
        self.transform     = transform
        self.known_classes = known_classes
        self.samples       = []   # [(path, label)]
        self.class_names   = []   # 每个样本的真实类名

        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            label = known_classes.index(cls) if cls in known_classes else NUM_KNOWN
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                    self.samples.append((os.path.join(cls_dir, fname), label))
                    self.class_names.append(cls)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), label


# ═══════════════════════════════════════════════════════════════
# ResNet-50 模型（与闭集脚本完全相同）
# ═══════════════════════════════════════════════════════════════
def build_resnet(num_classes: int, backbone="resnet50") -> nn.Module:
    if backbone == "resnet18":
        model    = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = 512
    elif backbone == "resnet34":
        model    = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feat_dim = 512
    else:
        model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = 2048
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(feat_dim, num_classes))
    return model


# ═══════════════════════════════════════════════════════════════
# S3R 中心损失（论文公式 20-21）
# ═══════════════════════════════════════════════════════════════
class CenterLoss(nn.Module):
    """
    可学习的类中心 μk；前向计算中心损失 Lcen = Σ ||z - μy||²
    中心用独立 SGD 优化器更新（不影响主网络梯度）。
    """
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.centers[labels]                        # (B, D)
        return ((features - c) ** 2).sum(1).mean()


# ═══════════════════════════════════════════════════════════════
# S3R 聚类损失（论文公式 22）
# ═══════════════════════════════════════════════════════════════
def clustering_loss(features: torch.Tensor,
                    labels: torch.Tensor,
                    centers: torch.Tensor,
                    margin: float = MARGIN) -> torch.Tensor:
    """拉开不同类语义之间的距离：max(0, ϑ - ||z - μk||) for k≠y"""
    loss = torch.tensor(0.0, device=features.device)
    K = centers.shape[0]
    for i in range(len(features)):
        y = labels[i].item()
        for k in range(K):
            if k != y:
                d = torch.norm(features[i] - centers[k])
                loss = loss + F.relu(margin - d)
    return loss / len(features)


# ═══════════════════════════════════════════════════════════════
# 训练 / 验证 epoch
# ═══════════════════════════════════════════════════════════════
def run_epoch(model, feature_extractor, loader, ce_fn,
              center_loss_fn, optimizer, center_opt, device, phase="train"):
    """
    model            : 完整 ResNet（输出 logits）
    feature_extractor: 去掉 fc 层的 ResNet（输出 2048 维特征）
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss = total_correct = total = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # 提取 GAP 特征（用于中心损失和聚类损失）
            feats  = feature_extractor(imgs).flatten(1)   # (B, 2048)
            logits = model.fc(feats)                       # (B, num_classes)

            Lce  = ce_fn(logits, labels)
            Lcen = center_loss_fn(feats, labels)
            Lclu = clustering_loss(feats, labels, center_loss_fn.centers)
            loss = ETA3 * Lce + ETA1 * Lcen + ETA2 * Lclu

            if is_train:
                optimizer.zero_grad()
                center_opt.zero_grad()
                loss.backward()
                optimizer.step()
                center_opt.step()

            total_loss    += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += imgs.size(0)

    return total_loss / total, total_correct / total


# ═══════════════════════════════════════════════════════════════
# S3R 语义分类器（论文 Section IV-C）
# ═══════════════════════════════════════════════════════════════
def mahal_dist(z: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    d = z - mu
    return float(np.sqrt(d @ cov_inv @ d))


def outlier_threshold_3sigma(distances: np.ndarray) -> float:
    """
    论文公式 (29)：在降序排列的距离中找到第一个不是离群点的值作为阈值。
    3σ 规则：距离 < 3 * σ 即非离群点。
    """
    if len(distances) == 0:
        return np.inf
    sorted_d = np.sort(distances)[::-1]   # 降序
    sigma    = float(np.std(distances))
    for d in sorted_d:
        if d < 3.0 * sigma:
            return float(d)
    return float(sorted_d[-1])            # 所有都是离群点时取最小


class SoftmaxThresholdClassifier:
    """
    开集分类器：Softmax 最大值阈值法（最简单有效）

    逻辑：
      softmax_max > threshold → 已知类（argmax 的类别）
      softmax_max ≤ threshold → Unknown

    threshold 从训练集 softmax 分布自动确定：
      取所有训练样本 softmax 最大值的第 THRESHOLD_PERCENTILE 百分位。
      这样 THRESHOLD_PERCENTILE% 的训练样本会被判为已知，
      低置信度的测试样本（未知类）会落在阈值以下。
    """
    def __init__(self, known_classes):
        self.known_classes = known_classes
        self.threshold = None

    def fit(self, model, loader, device):
        """用训练集 softmax 分布确定阈值"""
        print(f"\n拟合 Softmax 阈值分类器...")
        model.eval()
        all_max_probs = []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(device)
                feats  = nn.Sequential(*list(model.children())[:-1])(imgs).flatten(1)
                logits = model.fc(feats)
                probs  = torch.softmax(logits, dim=1)
                max_p  = probs.max(dim=1).values.cpu().numpy()
                all_max_probs.extend(max_p)

        all_max_probs  = np.array(all_max_probs)
        self.threshold = float(np.percentile(all_max_probs, 100 - THRESHOLD_PERCENTILE))
        print(f"  Softmax 最大值分布: mean={all_max_probs.mean():.3f}  "
              f"min={all_max_probs.min():.3f}  max={all_max_probs.max():.3f}")
        print(f"  自动阈值 (第{100-THRESHOLD_PERCENTILE}百分位): {self.threshold:.4f}")
        print(f"  含义: softmax最大值 > {self.threshold:.4f} → 已知类，否则 → Unknown")

    def predict_batch(self, model, loader, device):
        """返回 (preds, max_probs)"""
        model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs   = imgs.to(device)
                feats  = nn.Sequential(*list(model.children())[:-1])(imgs).flatten(1)
                logits = model.fc(feats)
                probs  = torch.softmax(logits, dim=1)
                max_p, argmax = probs.max(dim=1)
                max_p  = max_p.cpu().numpy()
                argmax = argmax.cpu().numpy()
                for p, cls in zip(max_p, argmax):
                    if p > self.threshold:
                        all_preds.append(int(cls))
                    else:
                        all_preds.append(NUM_KNOWN)  # Unknown
                all_probs.extend(max_p)
        return np.array(all_preds), np.array(all_probs)


# ═══════════════════════════════════════════════════════════════
# 绘图辅助
# ═══════════════════════════════════════════════════════════════
DARK_BG = "#0f1117"
DARK_AX = "#1a1d27"
PAL = ["#58a6ff","#f78166","#3fb950","#d29922","#bc8cff",
       "#ff5050","#ffa040","#aaaaaa"]   # 前5已知，后2未知

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
    axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor="#21262d", labelcolor="#c9d1d9")

    axes[1].plot(ep, [a*100 for a in history["train_acc"]], color="#3fb950", lw=2, label="Train")
    axes[1].plot(ep, [a*100 for a in history["val_acc"]],   color="#d29922", lw=2, label="Val")
    axes[1].set_title("Accuracy Curve"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc (%)")
    axes[1].legend(facecolor="#21262d", labelcolor="#c9d1d9")
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()


def save_confusion(y_true, y_pred, names, path, title="Confusion Matrix"):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    cm_norm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-9)
    annot   = np.array([[f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]})"
                         for j in range(len(names))]
                        for i in range(len(names))])
    sz = max(7, len(names))
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=names, yticklabels=names,
                ax=ax, linewidths=0.5, vmin=0, vmax=1, annot_kws={"fontsize": 8})
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_roc(y_binary, scores, path):
    fpr, tpr, _ = roc_curve(y_binary, scores)
    auc = roc_auc_score(y_binary, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(DARK_BG); _style(ax)
    ax.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1], [0,1], "--", color="#555", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("Open-Set ROC (Known vs Unknown)", color="#e6edf3")
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return auc


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
        marker = "o" if idx < NUM_KNOWN else "^"
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
    print(f"闭集类别  : {KNOWN_CLASSES}")
    print(f"开集类别  : {UNKNOWN_CLASSES}")

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

    # ── 训练集（仅已知类）────────────────────────────────────
    full_train = datasets.ImageFolder(root=TRAIN_DIR, transform=train_tf)
    assert sorted(full_train.classes) == sorted(KNOWN_CLASSES), \
        f"train 目录类别 {full_train.classes} 与 KNOWN_CLASSES {KNOWN_CLASSES} 不符！"

    counts = np.bincount([s[1] for s in full_train.samples])
    print(f"训练集各类: {dict(zip(full_train.classes, counts.tolist()))}")

    n_val   = int(len(full_train) * VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_raw, val_raw = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_set = TransformSubset(val_raw, eval_tf)

    train_labels  = [full_train.samples[i][1] for i in train_raw.indices]
    cls_w         = 1.0 / np.bincount(train_labels)
    sample_w      = [cls_w[l] for l in train_labels]
    sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_raw, BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_set,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── 测试集（含未知类）────────────────────────────────────
    test_dataset = OpenSetTestDataset(TEST_DIR, full_train.classes, eval_tf)
    test_loader  = DataLoader(test_dataset, BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    n_k = sum(1 for _,l in test_dataset.samples if l < NUM_KNOWN)
    n_u = sum(1 for _,l in test_dataset.samples if l == NUM_KNOWN)
    print(f"测试集: 已知 {n_k}  未知 {n_u}")

    # ── 模型：ResNet-50 ───────────────────────────────────────
    model = build_resnet(NUM_KNOWN, backbone="resnet50").to(DEVICE)

    # 特征提取器：去掉 fc，保留到 GAP 输出 (B,2048,1,1)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)

    # S3R 中心损失（2048 维特征空间）
    center_loss_fn = CenterLoss(NUM_KNOWN, feat_dim=2048).to(DEVICE)
    ce_fn          = nn.CrossEntropyLoss()

    # 主优化器（ResNet 参数）
    optimizer   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # 中心独立优化器（SGD，按论文）
    center_opt  = optim.SGD(center_loss_fn.parameters(), lr=CENTER_LR)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── 训练循环 ──────────────────────────────────────────────
    history      = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())

    print("\n开始训练（ResNet + 中心损失 + 聚类损失）...\n" + "="*65)
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, feature_extractor, train_loader, ce_fn,
                                    center_loss_fn, optimizer, center_opt, DEVICE, "train")
        vl_loss, vl_acc = run_epoch(model, feature_extractor, val_loader, ce_fn,
                                    center_loss_fn, None, None, DEVICE, "val")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        tag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save({"model": best_state,
                        "known_classes": full_train.classes}, MODEL_SAVE)
            tag = " ← best"

        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"Acc {tr_acc*100:.1f}%/{vl_acc*100:.1f}%  "
              f"({time.time()-t0:.1f}s){tag}")

    print(f"\n最佳验证准确率: {best_val_acc*100:.2f}%")
    model.load_state_dict(best_state)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)

    # ── Loss 曲线 ─────────────────────────────────────────────
    loss_path = os.path.join(OUTPUT_DIR, "01_loss_curve.png")
    save_loss_curve(history, loss_path)
    print(f"\nLoss 曲线: {loss_path}")

    # ═══ Softmax 阈值分类器拟合 ═════════════════════════════
    # 用无增强 transform 重新跑一遍训练集，统计 softmax 分布确定阈值
    train_eval_set = TransformSubset(train_raw, eval_tf)
    train_eval_ldr = DataLoader(train_eval_set, BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS)

    clf = SoftmaxThresholdClassifier(full_train.classes)
    clf.fit(model, train_eval_ldr, DEVICE)

    # ═══ 测试集推理 ══════════════════════════════════════════
    print("\n开始开集测试...")
    # 同时收集 labels 和 fine_labels（用于 t-SNE 细粒度着色）
    labels_test = np.array([l for _, l in test_dataset.samples])
    preds, max_probs = clf.predict_batch(model, test_loader, DEVICE)

    # ── 获取每个测试样本的真实类名（用于细粒度 t-SNE）────────
    fine_class_names = full_train.classes + UNKNOWN_CLASSES
    fine_labels      = []
    for cls_name in test_dataset.class_names:
        if cls_name in fine_class_names:
            fine_labels.append(fine_class_names.index(cls_name))
        else:
            fine_labels.append(len(fine_class_names) - 1)
    fine_labels = np.array(fine_labels)

    # ── 评估指标 ──────────────────────────────────────────────
    all_names = full_train.classes + ["Unknown"]
    print("\n" + "="*65)
    print("开集分类报告：")
    print(classification_report(labels_test, preds,
                                labels=list(range(NUM_KNOWN + 1)),
                                target_names=all_names,
                                zero_division=0))

    # TKR / TUR / KP（论文公式 36）
    is_k_gt   = labels_test < NUM_KNOWN
    is_u_gt   = labels_test == NUM_KNOWN
    is_k_pred = preds < NUM_KNOWN
    is_u_pred = preds == NUM_KNOWN

    TK = int((is_k_gt  & is_k_pred).sum())
    TU = int((is_u_gt  & is_u_pred).sum())
    FK = int((is_u_gt  & is_k_pred).sum())
    FU = int((is_k_gt  & is_u_pred).sum())
    CK = int(((preds == labels_test) & is_k_gt).sum())

    TKR = TK / (TK + FU + 1e-9)
    TUR = TU / (TU + FK + 1e-9)
    KP  = CK / (TK + FK + 1e-9)

    print(f"TKR (True Known Rate)   : {TKR*100:.2f}%")
    print(f"TUR (True Unknown Rate) : {TUR*100:.2f}%")
    print(f"KP  (Known Precision)   : {KP*100:.2f}%")
    print(f"TK={TK}  TU={TU}  FK={FK}  FU={FU}  CK={CK}")

    # ── 混淆矩阵 ──────────────────────────────────────────────
    cm_path = os.path.join(OUTPUT_DIR, "02_confusion_matrix_openset.png")
    save_confusion(labels_test, preds, all_names, cm_path,
                   "Open-Set Confusion Matrix (Row-Normalized)")
    print(f"\n混淆矩阵: {cm_path}")

    # ── ROC 曲线 ──────────────────────────────────────────────
    binary_gt    = (labels_test < NUM_KNOWN).astype(int)  # 1=known
    known_scores = max_probs                               # softmax最大值越大越像已知
    auc = save_roc(binary_gt, known_scores,
                   os.path.join(OUTPUT_DIR, "03_roc_curve.png"))
    print(f"ROC AUC: {auc:.4f}")

    # ── t-SNE（提取特征后可视化）────────────────────────────
    print("\n绘制 t-SNE（提取特征中）...")
    feature_extractor_eval = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)
    feature_extractor_eval.eval()
    feats_test_np = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            f = feature_extractor_eval(imgs.to(DEVICE)).flatten(1).cpu().numpy()
            feats_test_np.append(f)
    feats_test_np = np.concatenate(feats_test_np, 0)

    save_tsne(feats_test_np, fine_labels, fine_class_names,
              os.path.join(OUTPUT_DIR, "04_tsne_all.png"),
              "t-SNE: Known (●) & Unknown (▲) | ResNet-50 Features")

    # 仅已知类
    km = fine_labels < NUM_KNOWN
    save_tsne(feats_test_np[km], fine_labels[km], full_train.classes,
              os.path.join(OUTPUT_DIR, "05_tsne_known_only.png"),
              "t-SNE: Known Classes Only")

    print("\n✅ 全部完成！输出目录:", OUTPUT_DIR)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")