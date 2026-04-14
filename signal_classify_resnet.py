"""
信号体制分类 —— ResNet 训练脚本
支持类别：WiFi / 5G / 4G / DVB / GSM
数据结构：
    SigPic/train/<class>/xxx.png
    SigPic/test/<class>/xxx.png
功能：训练 ResNet、绘制 Loss/Acc 曲线、混淆矩阵（概率）、t-SNE 特征分布图
"""

import os
import time
import copy
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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ─────────────────────────────────────────────
# 超参数 & 路径配置
# ─────────────────────────────────────────────
TRAIN_DIR    = r"C:\Users\weiwu\Desktop\SigPic\train"  # 训练集根目录
TEST_DIR     = r"C:\Users\weiwu\Desktop\SigPic\test"   # 测试集根目录
OUTPUT_DIR   = r"C:\Users\weiwu\Desktop\SigPic_output" # 输出目录
MODEL_SAVE   = os.path.join(OUTPUT_DIR, "best_resnet.pth")

NUM_CLASSES  = 5
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15   # 从 train 中划出 15% 做验证集
IMG_SIZE     = 224
SEED         = 42
NUM_WORKERS  = 0      # Windows 下必须为 0


# ─────────────────────────────────────────────
# 辅助：对 Subset 重新应用 transform（val 无增强）
# ─────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ─────────────────────────────────────────────
# 模型构建
# ─────────────────────────────────────────────
def build_model(num_classes: int, backbone: str = "resnet50") -> nn.Module:
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


# ─────────────────────────────────────────────
# 单 epoch 训练 / 验证
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, phase="train"):
    is_train = (phase == "train")
    model.train() if is_train else model.eval()
    running_loss = running_correct = total = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss    += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total           += imgs.size(0)

    return running_loss / total, running_correct / total


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    # ── Transform ─────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── 加载 train / test ─────────────────────
    full_train = datasets.ImageFolder(root=TRAIN_DIR, transform=train_tf)
    test_set_raw = datasets.ImageFolder(root=TEST_DIR,  transform=val_tf)

    CLASS_NAMES = full_train.classes
    print(f"检测到类别: {CLASS_NAMES}")

    train_counts = np.bincount([s[1] for s in full_train.samples])
    test_counts  = np.bincount([s[1] for s in test_set_raw.samples])
    print("训练集各类样本数:", dict(zip(CLASS_NAMES, train_counts.tolist())))
    print("测试集各类样本数:", dict(zip(CLASS_NAMES, test_counts.tolist())))

    # 确保 train/test 类别一致
    assert full_train.classes == test_set_raw.classes, \
        "train 和 test 的类别不一致，请检查文件夹结构！"

    # ── 从 train 划出验证集 ───────────────────
    n_total = len(full_train)
    n_val   = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_raw, val_raw = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_set = TransformSubset(val_raw, val_tf)   # val 不做增强

    print(f"训练集: {len(train_raw)}  验证集: {len(val_set)}  测试集: {len(test_set_raw)}")

    # ── WeightedRandomSampler（应对类别不平衡）─
    train_labels  = [full_train.samples[i][1] for i in train_raw.indices]
    class_counts  = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_w      = [class_weights[l] for l in train_labels]
    sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_raw,     batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_set,       batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_set_raw,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── 模型 / 优化器 ─────────────────────────
    model     = build_model(NUM_CLASSES, backbone="resnet50").to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── 训练循环 ──────────────────────────────
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    print("\n开始训练...\n" + "=" * 60)
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, DEVICE, "train")
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, None,      DEVICE, "val")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_SAVE)
            tag = " ← best"
        else:
            tag = ""

        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}%  |  "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}%  "
              f"({time.time()-t0:.1f}s){tag}")

    print(f"\n最佳验证准确率: {best_val_acc*100:.2f}%  模型已保存至: {MODEL_SAVE}")

    # ── Loss / Accuracy 曲线 ──────────────────
    epochs_range = range(1, NUM_EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    axes[0].plot(epochs_range, history["train_loss"], color="#58a6ff", lw=2, label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"],   color="#f78166", lw=2, label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend(facecolor="#21262d", labelcolor="#c9d1d9", framealpha=0.8)
    axes[0].grid(True, alpha=0.15)

    axes[1].plot(epochs_range, [a*100 for a in history["train_acc"]], color="#3fb950", lw=2, label="Train Acc")
    axes[1].plot(epochs_range, [a*100 for a in history["val_acc"]],   color="#d29922", lw=2, label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend(facecolor="#21262d", labelcolor="#c9d1d9", framealpha=0.8)
    axes[1].grid(True, alpha=0.15)
    axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))

    plt.tight_layout()
    loss_fig_path = os.path.join(OUTPUT_DIR, "loss_accuracy_curve.png")
    plt.savefig(loss_fig_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nLoss/Acc 曲线已保存: {loss_fig_path}")

    # ── 测试集评估 & 混淆矩阵（概率）────────────
    model.load_state_dict(best_weights)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs  = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n测试集分类报告:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot   = np.array([[f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]})"
                         for j in range(cm.shape[1])]
                        for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5, vmin=0, vmax=1,
                annot_kws={"fontsize": 9})
    ax.set_title("Confusion Matrix (Row-Normalized)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存: {cm_path}")

    # ── t-SNE 特征可视化（使用完整测试集）───────
    print("\n提取特征向量用于 t-SNE（使用测试集）...")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(DEVICE).eval()

    feats_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            f    = feature_extractor(imgs).view(imgs.size(0), -1).cpu().numpy()
            feats_list.append(f)
            labels_list.extend(labels.numpy())

    feats     = np.concatenate(feats_list, axis=0)
    labels_np = np.array(labels_list)

    perplexity = min(30, len(feats) - 1)
    print(f"特征矩阵形状: {feats.shape}  perplexity={perplexity}  开始 t-SNE 降维...")
    tsne     = TSNE(n_components=2, perplexity=perplexity, max_iter=1000,
                   random_state=SEED, init="pca", learning_rate="auto")
    feats_2d = tsne.fit_transform(feats)
    print("t-SNE 降维完成。")

    PALETTE = ["#58a6ff", "#f78166", "#3fb950", "#d29922", "#bc8cff"]
    fig, ax  = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#c9d1d9")
    ax.title.set_color("#e6edf3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    for idx, cls in enumerate(CLASS_NAMES):
        mask = labels_np == idx
        ax.scatter(feats_2d[mask, 0], feats_2d[mask, 1],
                   c=PALETTE[idx], label=cls, s=20, alpha=0.75, linewidths=0)

    ax.legend(title="Signal Type", facecolor="#21262d",
              labelcolor="#c9d1d9", title_fontsize=10, fontsize=9, framealpha=0.85)
    ax.set_title("t-SNE Feature Distribution (Test Set)", fontsize=14, pad=12, color="#e6edf3")
    ax.set_xlabel("t-SNE Dim 1", color="#8b949e")
    ax.set_ylabel("t-SNE Dim 2", color="#8b949e")
    ax.grid(True, alpha=0.1, color="#30363d")

    plt.tight_layout()
    tsne_path = os.path.join(OUTPUT_DIR, "tsne_visualization.png")
    plt.savefig(tsne_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"t-SNE 图已保存: {tsne_path}")

    print("\n✅ 全部完成！输出文件：")
    print(f"  模型权重:  {MODEL_SAVE}")
    print(f"  Loss曲线:  {loss_fig_path}")
    print(f"  混淆矩阵:  {cm_path}")
    print(f"  t-SNE图:   {tsne_path}")