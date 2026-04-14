"""
S3R 官方代码精确移植版
═══════════════════════════════════════════════════════════════
来源：https://github.com/DaftJun/S3R
论文：Open Set Learning for RF-Based Drone Recognition via Signal Semantics
      IEEE TIFS 2024

与原始代码的差异（适配你的数据）：
  原始：输入 512×512 单通道 .npy 矩阵，x/y/z 三路分别是原图、原图、转置
  本版：输入 224×224 RGB PNG 图片，转为灰度后处理，结构完全一致

网络结构（完全照抄 train.py 的 NET 类）：
  TE：三路膨胀卷积 (d=1,3,5) → GAP → 拼接 → MLP → x_semantic
  PE：
    y = 图像行方向 (T×W→T×64) + 位置编码 → TransformerEncoder → 展平 → MLP → y_semantic
    z = 图像列方向 (W×T→W×64) + 位置编码 → TransformerEncoder → 展平 → MLP → z_semantic
  融合：cat(x,y,z) → MLP → semantic (128维)

损失（完全照抄 contLoss.py 的 ContLoss）：
  loss_center：同类到中心的欧氏距离均值
  loss_cluster：MarginRankingLoss（异类分离）

开集分类器（完全照抄 test_stage_1.py + test_stage_2.py）：
  Stage-1：pinv 协方差矩阵 + 马氏距离 + 3σ outlier_check → Known/-1
  Stage-2：MinMaxScaler + KMeans + Davies-Bouldin 选 u

数据目录：
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
TRAIN_DIR    = r"C:\Users\weiwu\Desktop\SigPic\train"
TEST_DIR     = r"C:\Users\weiwu\Desktop\SigPic\test"
OUTPUT_DIR   = r"C:\Users\weiwu\Desktop\SigPic_output_s3r"
MODEL_SAVE   = os.path.join(OUTPUT_DIR, "s3r_best.pth")

KNOWN_CLASSES   = ["4G", "5G", "DVB", "GSM", "WiFi"]
UNKNOWN_CLASSES = ["DMR", "TETRA"]
NUM_KNOWN       = len(KNOWN_CLASSES)   # 5

IMG_SIZE     = 224      # 图片大小
SEMANTIC_DIM = 128      # 语义维度，对应原代码 semantic_dim=128
MARGIN       = 8        # ContLoss margin，对应原代码 margin=8（可以调大到16让异类更分散）
BATCH_SIZE   = 32
NUM_EPOCHS   = 250      # 对应原代码 max_epoch=251
EVAL_INTERVAL= 50       # 每隔多少 epoch 评估一次，对应原代码 interval=50
LR           = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT    = 0.15
SEED         = 42
NUM_WORKERS  = 0

# 损失权重（完全照抄原代码 line 286）
# 损失权重
# 原代码是 [0.05, 1, 0.1]，但 Center Loss 过大说明同类太分散，需加大约束
ETA1 = 0.5    # center loss（调大，让同类聚拢）
ETA2 = 1.0    # cluster loss
ETA3 = 0.1    # ce loss


# ═══════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════
class SignalImageDataset(Dataset):
    """
    读取 PNG 时频图，转为灰度后归一化到 [0,1]。
    原始代码输入是 .npy 矩阵，这里改为从图片读取。
    同时提供 x（原图）、y（原图用于时域SA）、z（转置用于频域SA）
    对应原代码 __getitem__ 返回的 x, x, x.permute(1,0)
    """
    def __init__(self, root, known_classes, unknown_classes, img_size=IMG_SIZE):
        self.img_size       = img_size
        self.samples        = []
        self.fine_names     = []

        for folder in sorted(os.listdir(root)):
            fp = os.path.join(root, folder)
            if not os.path.isdir(fp): continue
            if   folder in known_classes:   label = known_classes.index(folder)
            elif folder in unknown_classes: label = NUM_KNOWN
            elif unknown_classes is None:   continue
            else:                           continue
            for fname in sorted(os.listdir(fp)):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                    self.samples.append((os.path.join(fp, fname), label))
                    self.fine_names.append(folder)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # 转灰度，归一化到 [0,1]，reshape 为 (img_size, img_size)
        img = Image.open(path).convert("L").resize((self.img_size, self.img_size))
        x   = torch.FloatTensor(np.array(img) / 255.0)   # (H, W)
        # 原代码：return x.unsqueeze(0), x, x.permute(1,0), label
        # x.unsqueeze(0)：(1,H,W) 输入卷积网络
        # x：(H,W) = (T,W) 用于时域 SA，每行是一个时间步
        # x.permute(1,0)：(W,H) = (W,T) 用于频域 SA，每行是一个频率步
        return x.unsqueeze(0), x, x.permute(1, 0), label


class TransformSubset(Dataset):
    """val 子集不做随机增强，直接返回 SignalImageDataset 的样本。"""
    def __init__(self, subset):
        self.subset = subset
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        # subset 的 dataset 是 SignalImageDataset，直接透传
        return self.subset[idx]


# ═══════════════════════════════════════════════════════════════
# ContLoss（完全照抄 contLoss.py）
# ═══════════════════════════════════════════════════════════════
class ContLoss(nn.Module):
    def __init__(self, num_classes, device, feat_dims, margin):
        super(ContLoss, self).__init__()
        self.num_classes  = num_classes
        self.device       = device
        self.margin       = margin
        self.feat_dim     = feat_dims
        self.centers      = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim).to(device))
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, x, labels):
        batch_size = x.size(0)
        # 计算每个样本到每个类中心的欧氏距离矩阵
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(mat1=x, mat2=self.centers.t(), alpha=-2, beta=1)
        distmat = torch.sqrt(distmat.clamp(min=1e-12))

        classes  = torch.arange(self.num_classes).long().to(self.device)
        labels_e = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask     = labels_e.eq(classes.expand(batch_size, self.num_classes))
        mis_mask = labels_e.ne(classes.expand(batch_size, self.num_classes))

        # 同类损失：到同类中心的距离均值
        dist = []
        for i in range(batch_size):
            val = distmat[i][mask[i]].clamp(min=1e-12, max=1e+12)
            dist.append(val)
        dist   = torch.cat(dist)
        loss_1 = dist.mean()

        # 异类损失：MarginRankingLoss（希望异类距离大于 margin）
        ss     = distmat[mis_mask]
        num    = self.num_classes * batch_size - batch_size
        loss_2 = self.ranking_loss(ss,
                                   torch.zeros(num).to(self.device),
                                   torch.ones(num).to(self.device))
        return loss_1, loss_2


# ═══════════════════════════════════════════════════════════════
# 位置编码（完全照抄 train.py position_coding）
# ═══════════════════════════════════════════════════════════════
def position_coding(x):
    """x: (B, N, d) → 返回位置编码 (1, N, d)"""
    num_token, num_dims = x.size(-2), x.size(-1)
    p = torch.zeros((1, num_token, num_dims))
    t = torch.arange(num_token, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(1e4, torch.arange(0, num_dims, 2, dtype=torch.float32) / num_dims)
    p[:, :, 0::2] = torch.sin(t)
    p[:, :, 1::2] = torch.cos(t[:, :num_dims//2] if num_dims % 2 == 0 
                               else t[:, :(num_dims+1)//2])[:, :p[:, :, 1::2].shape[-1]]
    return p


# ═══════════════════════════════════════════════════════════════
# NET（完全照抄 train.py NET 类，适配 224×224 输入）
# ═══════════════════════════════════════════════════════════════
class NET(nn.Module):
    def __init__(self, in_channels, input_size, semantic_dim, num_class, device):
        """
        input_size: [H, W] = [224, 224]
        原代码是 [512, 512]，这里适配 224×224
        """
        super(NET, self).__init__()
        self.input_size   = input_size   # [T, W]
        self.semantic_dim = semantic_dim
        self.num_class    = num_class
        self.device       = device

        # ── PE：位置编码 + Transformer（完全照抄）────────────
        # SA_1：时域，输入序列长度 T，每步特征维度 64
        self.SA_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8,
                                       batch_first=True, dim_feedforward=256),
            num_layers=3)
        # SA_2：频域，输入序列长度 W，每步特征维度 64
        self.SA_2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=8,
                                       batch_first=True, dim_feedforward=256),
            num_layers=3)

        # 把时频图每行（W维）压到 64 维
        self.encoding_to_sa1 = nn.Sequential(
            nn.Linear(input_size[1], 128), nn.ReLU(),
            nn.Linear(128, 64),           nn.ReLU()
        )
        # 把时频图每列（T维）压到 64 维
        self.encoding_to_sa2 = nn.Sequential(
            nn.Linear(input_size[0], 128), nn.ReLU(),
            nn.Linear(128, 64),            nn.ReLU()
        )

        # SA 输出展平后映射到语义维度
        self.sa1_to_semantic = nn.Sequential(
            nn.Linear(input_size[0] * 64, 512, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, semantic_dim, bias=False),
            nn.BatchNorm1d(semantic_dim), nn.ReLU()
        )
        self.sa2_to_semantic = nn.Sequential(
            nn.Linear(input_size[1] * 64, 512, bias=False),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, semantic_dim, bias=False),
            nn.BatchNorm1d(semantic_dim), nn.ReLU()
        )

        # 三路语义融合
        self.total_semantic = nn.Sequential(
            nn.Linear(semantic_dim * 3, semantic_dim, bias=False),
            nn.BatchNorm1d(semantic_dim), nn.ReLU(),
            nn.Linear(semantic_dim, semantic_dim, bias=False),
            nn.BatchNorm1d(semantic_dim), nn.ReLU()
        )

        # ── TE：三路膨胀卷积（完全照抄）─────────────────────
        def make_dcl(dilation):
            pad = dilation
            return nn.Sequential(
                nn.Conv2d(in_channels, 4,   3, 1, pad, dilation), nn.BatchNorm2d(4),   nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(4,  8,  3, 1, pad, dilation), nn.BatchNorm2d(8),   nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(8,  16, 3, 1, pad, dilation), nn.BatchNorm2d(16),  nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(16, 32, 3, 1, pad, dilation), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(32, 64, 3, 1, pad, dilation), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128, 3, 1, 1),             nn.BatchNorm2d(128), nn.ReLU(), nn.AvgPool2d(2,2),
            )

        self.encoder_d1 = make_dcl(1)
        self.encoder_d3 = make_dcl(3)
        self.encoder_d5 = make_dcl(5)

        # TE 输出 128×3=384 维 → 语义维度
        self.encoder_to_semantic = nn.Sequential(
            nn.Linear(128 * 3, semantic_dim * 2, bias=False),
            nn.BatchNorm1d(semantic_dim * 2), nn.ReLU(),
            nn.Linear(semantic_dim * 2, semantic_dim, bias=False),
            nn.BatchNorm1d(semantic_dim), nn.ReLU()
        )

        # 分类头
        self.semantic_to_classifier = nn.Linear(semantic_dim, num_class)

    def forward(self, x, y, z):
        """
        x: (B, 1, H, W)  → 卷积网络（TE）
        y: (B, T, W)      → 时域 SA（PE）
        z: (B, W, T)      → 频域 SA（PE）
        """
        # ── TE ───────────────────────────────────────────────
        e1 = self.encoder_d1(x)
        e3 = self.encoder_d3(x)
        e5 = self.encoder_d5(x)
        enc = torch.cat([e1, e3, e5], dim=1)                # (B, 384, h, w)
        enc_pool = F.adaptive_avg_pool2d(enc, (1,1))
        enc_flat = enc_pool.view(enc_pool.size(0), -1)       # (B, 384)
        x_sem    = self.encoder_to_semantic(enc_flat)        # (B, sem_dim)

        # expand_x 用于语义增强（test_stage_1 中的 test_X_expand）
        expand_x = enc_flat                                  # (B, 384)

        # ── PE ───────────────────────────────────────────────
        y = self.encoding_to_sa1(y)                          # (B, T, 64)
        z = self.encoding_to_sa2(z)                          # (B, W, 64)
        y = y + position_coding(y).to(self.device)
        z = z + position_coding(z).to(self.device)
        y = self.SA_1(y)                                     # (B, T, 64)
        z = self.SA_2(z)                                     # (B, W, 64)
        y = y.view(y.size(0), -1)                            # (B, T*64)
        z = z.view(z.size(0), -1)                            # (B, W*64)
        y_sem = self.sa1_to_semantic(y)                      # (B, sem_dim)
        z_sem = self.sa2_to_semantic(z)                      # (B, sem_dim)

        # ── 融合 ─────────────────────────────────────────────
        semantic = torch.cat([x_sem, y_sem, z_sem], dim=1)  # (B, sem_dim*3)
        semantic = self.total_semantic(semantic)             # (B, sem_dim)

        predict  = self.semantic_to_classifier(semantic)
        return predict, semantic, x, y, z, expand_x


# ═══════════════════════════════════════════════════════════════
# outlier_check（完全照抄）
# ═══════════════════════════════════════════════════════════════
def outlier_check(distance_list):
    distance     = np.flip(np.sort(distance_list))
    distance_std = np.std(np.hstack([distance, -distance]))
    threshold    = distance[0]
    for index in range(distance.shape[0]):
        threshold = distance[index]
        if distance[index] <= 3 * distance_std:
            break
    return threshold


# ═══════════════════════════════════════════════════════════════
# 绘图工具
# ═══════════════════════════════════════════════════════════════
DARK_BG = "#0f1117"; DARK_AX = "#1a1d27"
PAL = ["#58a6ff","#f78166","#3fb950","#d29922","#bc8cff","#ff4444","#ffa040"]

def _style(ax):
    ax.set_facecolor(DARK_AX); ax.tick_params(colors="#c9d1d9")
    ax.xaxis.label.set_color("#c9d1d9"); ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.12, color="#30363d")

def save_loss(loss_log, path):
    loss_log = np.array(loss_log)
    ep = loss_log[:, 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG); _style(ax)
    ax.plot(ep, loss_log[:,1], "#58a6ff", lw=2, label="Total")
    ax.plot(ep, loss_log[:,2], "#f78166", lw=2, label="Center")
    ax.plot(ep, loss_log[:,3], "#3fb950", lw=2, label="CE")
    ax.plot(ep, loss_log[:,4], "#d29922", lw=2, label="Cluster")
    ax.set_title("Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

def save_cm(y_true, y_pred, names, path, title="Confusion Matrix"):
    lbls = list(range(len(names)))
    cm      = confusion_matrix(y_true, y_pred, labels=lbls)
    cm_norm = cm.astype(float) / (cm.sum(1, keepdims=True) + 1e-9)
    annot   = np.array([[f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]})"
                         for j in range(len(names))] for i in range(len(names))])
    sz = max(7, len(names))
    fig, ax = plt.subplots(figsize=(sz, sz-1))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax,
                linewidths=0.5, vmin=0, vmax=1, annot_kws={"fontsize":8})
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

def save_tsne(feats, labels, names, path, title="t-SNE"):
    perp = min(30, len(feats)-1)
    print(f"  t-SNE 降维 (n={len(feats)}, perplexity={perp})...")
    xy = TSNE(2, perplexity=perp, max_iter=1000,
              random_state=SEED, init="pca", learning_rate="auto").fit_transform(feats)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG); _style(ax)
    for idx, name in enumerate(names):
        mask = labels == idx
        if mask.sum() == 0: continue
        ax.scatter(xy[mask,0], xy[mask,1], c=PAL[idx % len(PAL)],
                   label=name, s=22, alpha=0.75, linewidths=0,
                   marker="^" if idx >= NUM_KNOWN else "o")
    ax.legend(title="Signal", facecolor="#21262d", labelcolor="#c9d1d9",
              title_fontsize=9, fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=13, pad=10, color="#e6edf3")
    ax.set_xlabel("t-SNE Dim 1", color="#8b949e")
    ax.set_ylabel("t-SNE Dim 2", color="#8b949e")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG); plt.close()


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {DEVICE}  已知类: {KNOWN_CLASSES}  未知类: {UNKNOWN_CLASSES}")

    # ── 训练集（仅已知类）────────────────────────────────────
    train_full = SignalImageDataset(TRAIN_DIR, KNOWN_CLASSES, unknown_classes=[])
    counts = np.bincount([s[1] for s in train_full.samples], minlength=NUM_KNOWN)
    print("训练集各类:", dict(zip(KNOWN_CLASSES, counts.tolist())))

    n_val   = int(len(train_full) * VAL_SPLIT)
    n_train = len(train_full) - n_val
    train_sub, val_sub = random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED))

    pin = (DEVICE.type == "cuda")
    train_loader = DataLoader(train_sub, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_sub,   BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # ── 测试集（含未知类）────────────────────────────────────
    test_ds     = SignalImageDataset(TEST_DIR, KNOWN_CLASSES, UNKNOWN_CLASSES)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=NUM_WORKERS)
    test_labels = np.array([s[1] for s in test_ds.samples])
    print(f"测试集: 已知 {(test_labels<NUM_KNOWN).sum()}  "
          f"未知 {(test_labels==NUM_KNOWN).sum()}")

    # ── 网络 & 损失（完全照抄原代码）────────────────────────
    Net = NET(in_channels=1,
              input_size=[IMG_SIZE, IMG_SIZE],
              semantic_dim=SEMANTIC_DIM,
              num_class=NUM_KNOWN,
              device=DEVICE).to(DEVICE)

    ce_loss   = nn.CrossEntropyLoss()
    cont_loss = ContLoss(num_classes=NUM_KNOWN, device=DEVICE,
                         feat_dims=SEMANTIC_DIM, margin=MARGIN).to(DEVICE)

    # 原代码用 Adam（不是 AdamW）
    optimizer_net    = optim.Adam(Net.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-6)
    optimizer_center = optim.Adam(cont_loss.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-6)
    # 原代码 StepLR(step_size=10, gamma=0.98)
    lr_sch_net    = optim.lr_scheduler.StepLR(optimizer_net,    step_size=10, gamma=0.98)
    lr_sch_center = optim.lr_scheduler.StepLR(optimizer_center, step_size=10, gamma=0.98)

    # ── 训练循环（对应原代码 train 函数）────────────────────
    loss_log      = []
    indicator_log = []
    best_tur      = 0.0
    best_state    = copy.deepcopy(Net.state_dict())

    print(f"\n开始训练（{NUM_EPOCHS} epochs，每 {EVAL_INTERVAL} epochs 评估）\n" + "="*60)

    for epoch in range(NUM_EPOCHS + 1):
        t0 = time.time()
        Net.train()
        L_ce = L_cen = L_clu = L_tot = 0.0

        for x_b, y_b, z_b, label in train_loader:
            x_b, y_b, z_b, label = (x_b.to(DEVICE), y_b.to(DEVICE),
                                     z_b.to(DEVICE), label.to(DEVICE))
            predict, semantic, _, _, _, _ = Net(x_b, y_b, z_b)
            loss_cen, loss_clu = cont_loss(semantic, label)
            loss_ce            = ce_loss(predict, label)
            loss_total = ETA1*loss_cen + ETA2*loss_clu + ETA3*loss_ce

            optimizer_net.zero_grad(); optimizer_center.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(Net.parameters(), 5.0)
            optimizer_net.step(); optimizer_center.step()

            L_ce  += (ETA3 * loss_ce).item()
            L_cen += (ETA1 * loss_cen).item()
            L_clu += (ETA2 * loss_clu).item()
            L_tot += loss_total.item()

        lr_sch_net.step(); lr_sch_center.step()

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"Total:{L_tot:.4f}  CE:{L_ce:.4f}  "
              f"Center:{L_cen:.4f}  Cluster:{L_clu:.4f}  "
              f"({time.time()-t0:.1f}s)")
        loss_log.append([epoch, L_tot, L_cen, L_ce, L_clu])

        # ── 评估（每 EVAL_INTERVAL 轮，对应原代码）──────────
        if epoch % EVAL_INTERVAL == 0 and epoch != 0:
            print("─── 开始评估 ───")
            Net.eval()
            # 1) 提取训练集语义，计算类中心和协方差
            train_eval_loader = DataLoader(train_sub, batch_size=1,
                                           shuffle=False, num_workers=NUM_WORKERS)
            train_X = torch.zeros(len(train_sub), SEMANTIC_DIM)
            train_Y = torch.zeros(len(train_sub))
            with torch.no_grad():
                for i, (xb, yb, zb, lbl) in enumerate(train_eval_loader):
                    xb, yb, zb = xb.to(DEVICE), yb.to(DEVICE), zb.to(DEVICE)
                    _, train_X[i], _, _, _, _ = Net(xb, yb, zb)
                    train_Y[i] = lbl

            # 2) 按类计算 pinv 协方差、中心、3σ阈值（完全照抄）
            theta        = torch.zeros(NUM_KNOWN)
            dist_matrix  = np.zeros((NUM_KNOWN, SEMANTIC_DIM, SEMANTIC_DIM))
            class_centers= torch.zeros((NUM_KNOWN, SEMANTIC_DIM))
            for clas in range(NUM_KNOWN):
                samples       = train_X[train_Y == clas].cpu().numpy()
                cov_mat       = np.cov(samples, rowvar=False, bias=True)
                dist_matrix[clas]  = np.linalg.pinv(cov_mat)
                class_centers[clas]= torch.mean(train_X[train_Y == clas], dim=0)
                dx = (train_X[train_Y == clas] -
                      class_centers[clas].expand(samples.shape[0], SEMANTIC_DIM)).cpu().numpy()
                dist_list = np.sqrt(
                    np.matmul(np.matmul(dx, dist_matrix[clas]), dx.T).diagonal())
                theta[clas] = outlier_check(dist_list)

            # 3) 提取测试集语义
            test_X      = torch.zeros(len(test_ds), SEMANTIC_DIM)
            test_X_exp  = torch.zeros(len(test_ds), 128*3)
            test_Y_raw  = torch.zeros(len(test_ds))
            with torch.no_grad():
                for i, (xb, yb, zb, lbl) in enumerate(test_loader):
                    xb, yb, zb = xb.to(DEVICE), yb.to(DEVICE), zb.to(DEVICE)
                    _, test_X[i], _, _, _, test_X_exp[i] = Net(xb, yb, zb)
                    test_Y_raw[i] = lbl

            # 4) Stage-1 判决（完全照抄）
            d_ct = np.zeros((len(test_ds), NUM_KNOWN))
            for xi in range(NUM_KNOWN):
                for xj in range(len(test_ds)):
                    dx = (test_X[xj] - class_centers[xi]).cpu().numpy()
                    d_ct[xj, xi] = np.sqrt(
                        np.matmul(np.matmul(dx, dist_matrix[xi]), dx.T))
            Theta      = theta.expand(len(test_ds), NUM_KNOWN).numpy()
            x_ct       = d_ct - Theta
            label_hat  = np.zeros(len(test_ds))
            for xi in range(len(test_ds)):
                if np.min(x_ct[xi]) > 0:
                    label_hat[xi] = -1         # Unknown
                else:
                    label_hat[xi] = np.argmin(x_ct[xi])

            # 5) 归一化 test_Y（未知类→-1）
            test_Y_norm = test_Y_raw.cpu().numpy().copy()
            for xi in range(len(test_Y_norm)):
                if test_Y_norm[xi] >= NUM_KNOWN:
                    test_Y_norm[xi] = -1

            # 6) 指标计算（完全照抄 metrics_stage_1）
            num_samples = label_hat.shape[0]
            ones        = np.ones(num_samples)
            TK = np.sum(np.logical_and(label_hat != -ones, test_Y_norm != -ones))
            TU = np.sum(np.logical_and(test_Y_norm == -ones, label_hat == -ones))
            FK = np.sum(np.logical_and(test_Y_norm == -ones, label_hat != -ones))
            FU = np.sum(np.logical_and(test_Y_norm != -ones, label_hat == -ones))
            TKR = TK / (np.sum(test_Y_norm != -ones) + 1e-9)
            TUR = TU / (np.sum(test_Y_norm == -ones) + 1e-9)
            b   = np.sum(label_hat != -ones)
            CK  = np.sum(test_Y_norm[test_Y_norm != -ones] ==
                         label_hat[test_Y_norm != -ones])
            KP  = CK / b if b > 0 else 0

            print(f"  TKR={TKR:.4f}  TUR={TUR:.4f}  KP={KP:.4f}")

            # 保存最佳模型（以 TUR 为准，因为开集识别更重要）
            score = TKR * 0.4 + TUR * 0.6
            if score > best_tur:
                best_tur   = score
                best_state = copy.deepcopy(Net.state_dict())
                torch.save({
                    "model":         best_state,
                    "theta":         theta.numpy(),
                    "dist_matrix":   dist_matrix,
                    "class_centers": class_centers.numpy(),
                    "test_X":        test_X.numpy(),
                    "test_X_exp":    test_X_exp.numpy(),
                    "test_Y":        test_Y_raw.numpy(),
                    "label_hat":     label_hat,
                    "known_classes": KNOWN_CLASSES,
                }, MODEL_SAVE)
                print(f"  ✓ 模型已保存 (score={score:.4f})")
            indicator_log.append([epoch, round(TKR,4), round(TUR,4), round(KP,4)])

    # ── 保存 Loss 曲线 ────────────────────────────────────────
    save_loss(loss_log, os.path.join(OUTPUT_DIR, "01_loss_curve.png"))
    print("\nLoss 曲线已保存")

    # ════════════════════════════════════════════════════════
    # 最终评估（用最佳模型）
    # ════════════════════════════════════════════════════════
    print("\n加载最佳模型进行最终评估...")
    ckpt = torch.load(MODEL_SAVE, map_location=DEVICE, weights_only=False)
    Net.load_state_dict(ckpt["model"])
    Net.eval()

    theta_np     = ckpt["theta"]
    dist_matrix  = ckpt["dist_matrix"]
    class_centers= torch.FloatTensor(ckpt["class_centers"])
    test_X       = torch.FloatTensor(ckpt["test_X"])
    test_X_exp   = torch.FloatTensor(ckpt["test_X_exp"])
    test_Y_raw   = ckpt["test_Y"]
    label_hat    = ckpt["label_hat"]

    test_Y_norm = test_Y_raw.copy()
    for xi in range(len(test_Y_norm)):
        if test_Y_norm[xi] >= NUM_KNOWN:
            test_Y_norm[xi] = -1

    # ── 最终 Stage-1 指标 & 混淆矩阵 ────────────────────────
    all_names   = KNOWN_CLASSES + ["Unknown"]
    # 把 label_hat 中的 -1 转为 NUM_KNOWN
    preds_cm    = label_hat.copy().astype(int)
    preds_cm[preds_cm == -1] = NUM_KNOWN
    gt_cm       = test_Y_raw.copy().astype(int)
    for xi in range(len(gt_cm)):
        if gt_cm[xi] >= NUM_KNOWN: gt_cm[xi] = NUM_KNOWN

    print("\n" + "="*60)
    print("最终分类报告（Stage-1）：")
    print(classification_report(gt_cm, preds_cm,
                                labels=list(range(NUM_KNOWN+1)),
                                target_names=all_names, zero_division=0))

    save_cm(gt_cm, preds_cm, all_names,
            os.path.join(OUTPUT_DIR, "02_confusion_matrix_stage1.png"),
            "S3R Stage-1 Confusion Matrix")
    print("混淆矩阵已保存")

    # ── Stage-2：未知类聚类（完全照抄 test_stage_2.py）──────
    print("\n── Stage-2: 未知类聚类 ──")
    unk_mask    = label_hat == -1
    unk_X       = np.hstack([test_X.numpy(), test_X_exp.numpy()])[unk_mask]  # 增强语义
    unk_Y       = test_Y_raw[unk_mask]
    num_unknown = len(UNKNOWN_CLASSES)

    if len(unk_X) > 1:
        # Case-A 判断（照抄）
        cov_u  = np.cov(unk_X, rowvar=False, bias=True)
        mat_u  = np.linalg.pinv(cov_u)
        cen_u  = unk_X.mean(0)
        dx_u   = unk_X - cen_u
        dl_u   = np.sqrt(np.matmul(np.matmul(dx_u, mat_u), dx_u.T).diagonal())
        theta_u1 = outlier_check(dl_u)

        if theta_u1 <= np.max(theta_np):
            print("  Case-A: u=1，所有未知样本归为同一类")
            u = 1
        else:
            # Case-B：用 DB 指数选 u（照抄）
            scalar  = MinMaxScaler()
            unk_X_s = scalar.fit_transform(unk_X)
            DB, SC  = [], []
            for ui in range(2, min(15, len(unk_X))):
                km     = KMeans(n_clusters=ui, init='k-means++',
                                random_state=51, n_init=10).fit(unk_X_s)
                pl     = km.labels_
                db     = metrics.davies_bouldin_score(unk_X_s, pl)
                sc     = metrics.silhouette_score(unk_X_s, pl)
                print(f"    u={ui}: SC={sc:.4f}  DB={db:.4f}")
                DB.append(db); SC.append(sc)
            u = int(np.argmin(DB)) + 2
            print(f"  选定 u={u}（DB最小）")

            km_final   = KMeans(n_clusters=u, init='k-means++',
                                random_state=51, n_init=10).fit(unk_X_s)
            pred_label = km_final.labels_

            # 统计混淆（照抄）
            conf = np.zeros((u, num_unknown))
            for xi in range(len(unk_X_s)):
                conf[int(pred_label[xi])][
                    int(unk_Y[xi]) - NUM_KNOWN] += 1
            print("  未知类聚类混淆矩阵（行=簇，列=真实未知类）:")
            print("  列顺序:", UNKNOWN_CLASSES)
            print(conf)

            dominate = np.zeros(num_unknown)
            for row in range(u):
                for col in range(num_unknown):
                    if (conf[row,col] >= np.sum(conf[row])*0.5 and
                            np.argmax(conf[:,col]) == row):
                        dominate[col] = conf[row,col]

            unk_acc = []
            for clas in range(num_unknown):
                a = np.sum(unk_Y == (clas + NUM_KNOWN))
                unk_acc.append(dominate[clas] / a if a > 0 else 0)
            up = np.sum(dominate) / len(unk_X_s)
            print(f"  各未知类精度: {[f'{a:.4f}' for a in unk_acc]}")
            print(f"  mean unknown_acc: {np.mean(unk_acc):.4f}  UP: {up:.4f}")

    # ── t-SNE ─────────────────────────────────────────────────
    print("\n绘制 t-SNE...")
    fine_names  = KNOWN_CLASSES + UNKNOWN_CLASSES
    fine_labels = np.array([fine_names.index(n) for n in test_ds.fine_names])

    # 增强语义用于 t-SNE（照抄原代码思路）
    aug_feats = np.hstack([test_X.numpy(), test_X_exp.numpy()])
    save_tsne(aug_feats, fine_labels, fine_names,
              os.path.join(OUTPUT_DIR, "03_tsne_all.png"),
              "t-SNE (Augmented Semantics): Known(●) Unknown(▲)")

    km2 = fine_labels < NUM_KNOWN
    save_tsne(test_X.numpy()[km2], fine_labels[km2], KNOWN_CLASSES,
              os.path.join(OUTPUT_DIR, "04_tsne_known.png"),
              "t-SNE: Known Classes Only")

    print("\n✅ 完成！输出目录:", OUTPUT_DIR)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")