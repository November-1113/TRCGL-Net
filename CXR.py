import os
import warnings
import logging
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

import timm 
from timm.utils import ModelEmaV2 

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
# from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit 


warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
RANK = int(os.environ.get("RANK", -1))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", -1))

# DATA PATHS
SYNTH_IMAGE_DIR = "/home/shaotong/CMDC/RoentGen-v2-main/roentgenv2/inference_code/outputs"
DATA_DIR = '/home/shaotong/CMDC/originaldata'
IMAGE_BASE_DIR = '/home/shaotong/CMDC/originaldata/files'
REAL_NPY_DIR = "/lhh/shared_workspace/dataset2"
SYNTH_NPY_DIR = "/home/shaotong/CMDC/CXR_LT-main/Task1_Long_tailed/clean_preprocessed_synth_npy"
LABELS_CSV = '/home/shaotong/CMDC/CXR-LT-2026-main/CXR-LT-2026-main/CXRLT-2026-TRAINING DATA/CXRLT-2026-TRAINING DATA/train.csv'
CORRUPTED_CSV = '/home/shaotong/CMDC/CXR-LT-2026-main/CXR-LT-2026-main/CXRLT-2026-TRAINING DATA/CXRLT-2026-TRAINING DATA/corrupted_images.csv'
REMOVAL_CSV = '/home/shaotong/CMDC/CXR-LT-2026-main/CXR-LT-2026-main/CXRLT-2026-TRAINING DATA/CXRLT-2026-TRAINING DATA/Removal.csv'

# PRETRAIN / RESUME
PRETRAINED_CHECKPOINT = None 
RESUME_CHECKPOINT = None

# SPLIT
VAL_SPLIT = 0.1
RANDOM_SEED = 42
SPLIT_SEED = 42

# TRAINING
IMAGE_SIZE = 224
BATCH_SIZE = 32       # per GPU
VAL_BATCH_SIZE = 8
NUM_WORKERS = 4

# 预热头层
WARMUP_HEAD_EPOCHS = 10
WARMUP_HEAD_LR = 1e-3

# 联合微调
FINETUNE_EPOCHS = 50
BACKBONE_LR = 1.5e-5
HEAD_LR = 1.5e-4

MAX_GRAD_NORM = 1.0
EMA_DECAY = 0.9999

CHECKPOINT_DIR = "/home/shaotong/CMDC/CXR_LT-main/Task1_Long_tailed/roentgen_cxr_ECA_label_output1"

AMP_ENABLED = True

# ----------------------------
# DB Loss hyperparams
# ----------------------------
# Effective number beta (close to 1.0)
DB_BETA = 0.9999

# Rebalance strength: larger -> stronger tail boost
DB_REBALANCE_ALPHA = 0.5  # try 0.3~1.0

# Negative-tolerant regularization (reduce neg dominance)
DB_NEG_SCALE = 1        # keep 1.0 default, try 0.5 if too many negs
DB_NEG_MARGIN = 0.0       # optional margin, keep 0.0 usually

# Logit margin for tail classes (optional but helpful)
DB_MARGIN_SCALE = 1.0     # try 0.5~2.0
DB_MAX_MARGIN = 0.2       # cap margin

# ----------------------------
# CAS (Class-Aware Sampling)
# ----------------------------
# Repeat factor threshold like detectron2
CAS_T = 0.01  # class with freq < T gets repeated more (try 0.01 or 0.005)
CAS_MAX_REPEAT = 10  # cap repeats to avoid exploding epochs

EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 1e-4     

F1_THRESHOLD = 0.2
# ----------------------------
# Label Co-occurrence / Label GCN
# ----------------------------
LABEL_GCN_ENABLED = True
LABEL_GCN_HIDDEN = 256
LABEL_GCN_ALPHA = 0.05
LABEL_GCN_DROPOUT = 0.10

COOC_SMOOTH = 1.0
COOC_THRESHOLD = 0.02
COOC_SELF_LOOP = 1.0
# =============================================================================
# LOGGING
# =============================================================================



def setup_logger():
    if RANK == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - [Rank0] - %(message)s")
        return logging.getLogger("train")
    else:
        logger = logging.getLogger("train")
        logger.addHandler(logging.NullHandler())
        return logger

logger = setup_logger()


# =============================================================================
# DDP UTILS
# =============================================================================
def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def set_seed(seed):
    seed = seed + max(RANK, 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reduce_tensor(tensor):
    rt = tensor.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= WORLD_SIZE
    return rt

def gather_tensor(tensor):
    outs = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(outs, tensor)
    return torch.cat(outs, dim=0)

def get_splits(seed):
    train_df, val_df, class_names, cls_counts, freq = load_and_split_data(42)
    return val_df


# =============================================================================
# DB LOSS (Distribution-Balanced Loss for Multi-Label Long-Tail)
# =============================================================================
class DistributionBalancedLoss(nn.Module):
    """
    Practical implementation of Distribution-Balanced Loss (DB Loss) for multi-label long-tail.
    Components:
      1) Re-balanced weighting via effective number (class frequency)
      2) Negative-tolerant regularization (reduce neg dominance)
      3) Logit margin for tail classes

    This is a "production" style implementation tuned for stability.
    """

    def __init__(
        self,
        cls_counts,
        beta=0.9999,
        rebalance_alpha=0.5,
        neg_scale=1.0,
        neg_margin=0.0,
        margin_scale=1.0,
        max_margin=0.2,
        eps=1e-8,
    ):
        super().__init__()
        cls_counts = np.array(cls_counts, dtype=np.float32)
        cls_counts = np.clip(cls_counts, 1.0, None)
        self.num_classes = len(cls_counts)

        # class frequency (prior)
        freq = cls_counts / cls_counts.sum()
        self.register_buffer("freq", torch.tensor(freq, dtype=torch.float32))

        # effective number weights
        # w_c = (1 - beta) / (1 - beta^{n_c})
        eff_num = (1.0 - beta) / (1.0 - np.power(beta, cls_counts))
        eff_num = eff_num / eff_num.mean()  # normalize
        self.register_buffer("eff_w", torch.tensor(eff_num, dtype=torch.float32))

        self.beta = beta
        self.rebalance_alpha = rebalance_alpha
        self.neg_scale = neg_scale
        self.neg_margin = neg_margin
        self.margin_scale = margin_scale
        self.max_margin = max_margin
        self.eps = eps

        # margin per class: more margin for head classes, less for tail
        # Here we set margin inversely proportional to effective weight.
        # Tail (large eff_w) => smaller margin; Head => larger margin.
        # Then subtract margin on positives to encourage tail positives.
        inv = 1.0 / (self.eff_w + 1e-6)
        inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-6)
        margin = self.max_margin * inv * self.margin_scale
        self.register_buffer("margin", margin.float())


# class LDAMDRWLoss(nn.Module):
#     def __init__(self, class_counts, max_margin=0.5, scale=30.0, drw_epoch=10, beta=0.9999, synth_alpha=0.3):
#         super().__init__()
#         self.scale = scale
#         self.drw_epoch = drw_epoch
#         self.current_epoch = 0
#         self.synth_alpha = synth_alpha 

#         class_counts = torch.tensor(class_counts, dtype=torch.float32)
   
#         margins = 1.0 / torch.sqrt(torch.sqrt(class_counts))
#         margins = margins * (max_margin / margins.max())
#         self.register_buffer('margins', margins)
        
        
#         effective_num = 1.0 - torch.pow(beta, class_counts)
#         weights = (1.0 - beta) / effective_num
#         weights = weights / weights.sum() * len(class_counts)
#         self.register_buffer('cb_weights', weights)
#         self.register_buffer('uniform_weights', torch.ones_like(weights))

#     def update_epoch(self, epoch):
#         self.current_epoch = epoch

#     def forward(self, logits, targets, synthetic_flags=None, synth_weight=None):
       
#         ldam_logits = logits - targets * self.margins.unsqueeze(0)
        

#         if self.current_epoch < self.drw_epoch:
#             cls_weights = self.uniform_weights
#         else:
#             cls_weights = self.cb_weights

       
#         bce_loss = nn.functional.binary_cross_entropy_with_logits(
#             self.scale * ldam_logits, targets, reduction='none'
#         )

  
#         sample_cls_weight = (targets * cls_weights).sum(dim=1) / (targets.sum(dim=1) + 1e-6)

        
#         s_alpha = synth_weight if synth_weight is not None else self.synth_alpha
        
#         if synthetic_flags is not None:
    
#             sample_weights = torch.where(
#                 synthetic_flags.bool(),
#                 1.0 + s_alpha * sample_cls_weight,
#                 1.0 + sample_cls_weight
#             )
#         else:
#             sample_weights = 1.0 + sample_cls_weight

      
#         loss = (bce_loss.sum(dim=1) * sample_weights).mean()
#         return loss
    # def forward(self, logits, targets):
    #     """
    #     logits: (B, C)
    #     targets: (B, C) float {0,1}
    #     """
    #     # Apply per-class margin to positives (helps tail AP)
    #     # For positive label: logits' = logits - margin_c
    #     logits_adj = logits - targets * self.margin  # subtract margin only for positives

    #     # base BCE with logits
    #     bce = torch.nn.functional.binary_cross_entropy_with_logits(
    #         logits_adj, targets, reduction="none"
    #     )  # (B,C)

    #     # -------- Re-balanced weighting --------
    #     # weight positives more for tail classes, based on effective number
    #     # and also re-balance by sample-level label composition (optional)
    #     # Here practical: weight = eff_w^alpha
    #     w = torch.pow(self.eff_w, self.rebalance_alpha)  # (C,)
    #     bce = bce * w.unsqueeze(0)  # broadcast (B,C)

    #     # -------- Negative-tolerant regularization --------
    #     # Reduce penalty from negatives to avoid drowning rare positives
    #     # Scale negative part
    #     if self.neg_scale != 1.0 or self.neg_margin != 0.0:
    #         # separate pos / neg
    #         pos_mask = targets
    #         neg_mask = 1.0 - targets

    #         # For negatives, optionally add margin to logits to make negatives "easier"
    #         # logits_neg = logits + neg_margin
    #         logits_neg = logits_adj + self.neg_margin

    #         neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    #             logits_neg, targets, reduction="none"
    #         ) * neg_mask

    #         pos_loss = bce * pos_mask  # already weighted by eff_w

    #         # apply neg scale
    #         neg_loss = neg_loss * self.neg_scale

    #         loss = pos_loss + neg_loss
    #     else:
    #         loss = bce

    #     # mean over all elements
    #     return loss.mean()
    def forward(self, logits, targets, synthetic_flags=None, synth_weight=0.5):
       
       
        logits_adj = logits - targets * self.margin 

     
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_adj, targets, reduction="none"
        ) 


        w = torch.pow(self.eff_w, self.rebalance_alpha) 
        bce = bce * w.unsqueeze(0) 

    
        if self.neg_scale != 1.0 or self.neg_margin != 0.0:
            pos_mask = targets
            neg_mask = 1.0 - targets
            logits_neg = logits_adj + self.neg_margin

            neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits_neg, targets, reduction="none"
            ) * neg_mask
            pos_loss = bce * pos_mask 
            neg_loss = neg_loss * self.neg_scale
            sample_wise_loss = (pos_loss + neg_loss).sum(dim=1) 
        else:
            sample_wise_loss = bce.sum(dim=1) # (B,)

      
        if synthetic_flags is not None:
          
            sample_weights = torch.where(synthetic_flags.bool(), 
                                         torch.tensor(synth_weight).to(logits.device), 
                                         torch.tensor(1.0).to(logits.device))
            sample_wise_loss = sample_wise_loss * sample_weights

        return sample_wise_loss.mean()

# =============================================================================
# CAS Sampler (DDP-friendly)
# =============================================================================
class RepeatFactorDistributedSampler(DistributedSampler):
    """
    DDP-friendly "repeat factor" sampler (Detectron2-style) for multi-label.

    For each sample i, compute repeat factor r_i based on the rarest class it contains.
      r(c) = max(1, sqrt(T / freq(c)))
      r_i = max_{c in labels_i} r(c)
    Then repeat sample i r_i times (cap).

    We implement this by building an expanded index list per epoch (deterministic).
    """

    def __init__(
        self,
        dataset,
        labels_np,
        class_freq,
        threshold=0.01,
        max_repeat=10,
        shuffle=True,
        seed=0
    ):
        self.dataset = dataset
        self.labels_np = labels_np.astype(np.float32)  # (N,C)
        self.class_freq = class_freq.astype(np.float32)  # (C,)
        self.threshold = threshold
        self.max_repeat = max_repeat
        self.shuffle = shuffle
        self.seed = seed

        super().__init__(dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=shuffle, seed=seed)

        # precompute per-class repeat factor
        rf = np.sqrt(self.threshold / np.clip(self.class_freq, 1e-12, None))
        rf = np.maximum(1.0, rf)

        # compute sample repeat = max rf among its positive classes
        # if sample has no positives -> repeat = 1
        sample_rf = []
        for i in range(self.labels_np.shape[0]):
            pos = np.where(self.labels_np[i] > 0.5)[0]
            if len(pos) == 0:
                sample_rf.append(1.0)
            else:
                sample_rf.append(np.max(rf[pos]))
        sample_rf = np.clip(sample_rf, 1.0, float(self.max_repeat))

        self.sample_repeat = np.array(sample_rf, dtype=np.float32)

        if RANK == 0:
            logger.info(
                f"[CAS] threshold={threshold}, max_repeat={max_repeat} | "
                f"mean_repeat={self.sample_repeat.mean():.3f} | max_repeat={self.sample_repeat.max():.1f}"
            )

        # Build expanded indices (base, will be re-shuffled per epoch)
        self.base_indices = np.arange(len(dataset))

    def _build_epoch_indices(self, epoch):
        # deterministic for all ranks: must use same expansion order
        rng = np.random.RandomState(self.seed + epoch)
        indices = self.base_indices.copy()

        if self.shuffle:
            rng.shuffle(indices)

        expanded = []
        for idx in indices:
            r = int(np.round(self.sample_repeat[idx]))
            expanded.extend([idx] * r)

        expanded = np.array(expanded, dtype=np.int64)
        return expanded

    def __iter__(self):
        # Build expanded indices for this epoch
        expanded = self._build_epoch_indices(self.epoch)

        # Make total size divisible by world size
        total_size = int(np.ceil(len(expanded) / self.num_replicas)) * self.num_replicas
        if len(expanded) < total_size:
            # pad by wrapping
            pad = expanded[: total_size - len(expanded)]
            expanded = np.concatenate([expanded, pad], axis=0)

        # Subsample for this rank
        expanded_rank = expanded[self.rank: total_size: self.num_replicas]
        return iter(expanded_rank.tolist())

    def __len__(self):
        # approximate length per rank
        expanded = self._build_epoch_indices(self.epoch)
        total_size = int(np.ceil(len(expanded) / self.num_replicas)) * self.num_replicas
        return total_size // self.num_replicas


# =============================================================================
# DATA / AUGMENT
# =============================================================================
def get_transforms(is_train=False, img_size=224):
    transform_list = []
    # if is_train:
    #     return A.Compose([
    #         A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
    #         A.HorizontalFlip(p=0.5),
    #         A.ShiftScaleRotate(
    #             shift_limit=0.03, scale_limit=0.05, rotate_limit=7,
    #             p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0
    #         ),
    #         A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    #         A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.2),
    #         A.GaussNoise(var_limit=(1.0, 5.0), p=0.15),
    #         A.CoarseDropout(max_holes=6, max_height=24, max_width=24, p=0.15),
    #         A.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225],
    #                     max_pixel_value=255.0),
    #         ToTensorV2(),
    #     ])
    # else:
    #     return A.Compose([
    #         A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
    #         A.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225],
    #                     max_pixel_value=255.0),
    #         ToTensorV2(),
    #     ])
    # transform_list.append(A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR))
    if is_train:
     
        transform_list.extend([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03, scale_limit=0.05, rotate_limit=7,
                p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.2),
            A.GaussNoise(var_limit=(1.0, 5.0), p=0.15),
            A.CoarseDropout(max_holes=6, max_height=24, max_width=24, p=0.15),
        ])
    transform_list.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return A.Compose(transform_list)

# class PadChestDataset(Dataset):
#     def __init__(self, df, class_names, image_base_dir, augment=False, img_size=512):
#         self.df = df.reset_index(drop=True)
#         self.class_names = class_names
#         self.image_base_dir = image_base_dir
#         self.transform = get_transforms(is_train=augment, img_size=img_size)
#         self.img_size = img_size

#         self.labels = self.df[class_names].values.astype(np.float32)
#         self.image_paths = [os.path.join(image_base_dir, fname) for fname in self.df["ImageID"].values]

#     def __len__(self):
#         return len(self.df)
# class PadChestDataset(Dataset):
#     def __init__(self, df, class_names, image_base_dir,synth_image_dir, augment=False, img_size=224):
#         self.df = df.reset_index(drop=True)
#         self.class_names = class_names
#         self.image_base_dir = image_base_dir
#         self.img_size = img_size

   
#         self.synth_dir = synth_image_dir

#         self.is_synthetic = self.df["synthetic"].values.astype(bool)

#         self.transform_real = get_transforms(is_train=augment, img_size=img_size)
#         self.transform_synth = get_transforms(is_train=False, img_size=img_size) 

#         self.labels = self.df[class_names].values.astype(np.float32)
    
#         self.image_paths = []
#         for i, fname in enumerate(self.df["ImageID"].values):
#             if self.is_synthetic[i]:
#                 path = os.path.join(self.synth_dir, fname)
#             else:
#                 path = os.path.join(self.image_base_dir, fname)
#             self.image_paths.append(path)
class PadChestDataset(Dataset):
    def __init__(self, df, class_names, image_base_dir, synth_image_dir, augment=False, img_size=224):
        self.df = df.reset_index(drop=True)
        self.class_names = class_names
        self.image_base_dir = REAL_NPY_DIR
        self.synth_dir = SYNTH_NPY_DIR
        self.img_size = img_size
        self.augment = augment

        self.is_synthetic = self.df["synthetic"].values.astype(bool)
        self.labels = self.df[self.class_names].values.astype(np.float32)
        self.image_ids = self.df["ImageID"].values

 
        self.transform_train = get_transforms(is_train=True, img_size=img_size)
        self.transform_val = get_transforms(is_train=False, img_size=img_size)

        self.image_paths = [
            os.path.join(self.synth_dir if self.is_synthetic[i] else self.image_base_dir, fname)
            for i, fname in enumerate(self.df["ImageID"].values)
        ]
    def __len__(self):
        return len(self.df)

    @staticmethod
    def _percentile_to_uint8(img: np.ndarray):
        if img.dtype == np.uint8: return img
        flat = img.ravel()
        stride = max(1, flat.size // 2000)
        sampled = flat[::stride]
        p_low, p_high = np.percentile(sampled, (0.5, 99.5))
        if p_high > p_low:
            out = np.clip(img, p_low, p_high)
            out = ((out - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
        else:
            mx = img.max()
            out = ((img / mx) * 255.0).astype(np.uint8) if mx > 0 else img.astype(np.uint8)
        return out
    def _load_and_preprocess_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Image not found")
        if len(image.shape) == 2:
            image = self._percentile_to_uint8(image)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.dtype != np.uint8:
                image = self._percentile_to_uint8(image)
        return image

    # def __getitem__(self, idx):
    #     image_path = self.image_paths[idx]
    #     try:
    #         img = self._load_and_preprocess_image(image_path)
    #     except Exception:
    #         img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    #     aug = self.transform(image=img)
    #     image_tensor = aug["image"]
    #     label_tensor = torch.from_numpy(self.labels[idx])
    #     return image_tensor, label_tensor
    # def __getitem__(self, idx):
    #     img_path = self.image_paths[idx]
    #     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    #     if img is None:
    #         print(f"Image not found: {img_path}") 
    #         img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        
    #     if self.is_synthetic[idx]:
    #         img_tensor = self.transform_synth(image=img)["image"]
    #     else:
    #         img_tensor = self.transform_real(image=img)["image"]

    #     label_tensor = torch.from_numpy(self.labels[idx])
    #     synthetic_flag = self.is_synthetic[idx]
    #     return img_tensor, label_tensor, synthetic_flag
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        is_synth = self.is_synthetic[idx]

        try:
            if not is_synth:
                
                npy_path = os.path.join(self.image_base_dir, f"{img_id}.npy")
                img = np.load(npy_path)
            else:
                base_name = os.path.splitext(img_id)[0]
                npy_path = os.path.join(self.synth_dir, f"{base_name}.npy")
                img = np.load(npy_path)
                # img_path = os.path.join(self.synth_image_dir, img_id)
            
        #         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        #         if img is None: raise ValueError("File not found")
                
        #         if len(img.shape) == 2:
        #             img = self._percentile_to_uint8(img)
        #             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #         else:
        #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #             if img.dtype != np.uint8:
        #                 img = self._percentile_to_uint8(img)
                
        #         img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        except Exception:
        
            new_idx = np.random.randint(0, len(self.df))
            return self.__getitem__(new_idx)

   
        if self.augment:
            img_tensor = self.transform_train(image=img)["image"]
        else:
            img_tensor = self.transform_val(image=img)["image"]

        return img_tensor, torch.from_numpy(self.labels[idx]), is_synth
def fast_collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    synthetic_flags = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return images, labels, synthetic_flags

def load_removal_list():
    removal = set()
    if os.path.exists(CORRUPTED_CSV):
        removal.update(pd.read_csv(CORRUPTED_CSV)["ImageID"].tolist())
    if os.path.exists(REMOVAL_CSV):
        removal.update(pd.read_csv(REMOVAL_CSV)["ImageID"].tolist())
    if RANK == 0:
        logger.info(f"Total removed images: {len(removal)}")
    return removal

def calculate_class_counts(df, class_names):
    counts = []
    for c in class_names:
        counts.append(max(1, int(df[c].sum())))
    return counts

# def load_and_split_data():
#     df = pd.read_csv(LABELS_CSV)
#     removal = load_removal_list()
#     if removal:
#         df = df[~df["ImageID"].isin(removal)].reset_index(drop=True)

#     metadata_cols = ["ImageID", "StudyDate_DICOM", "PatientID"]
#     class_names = [c for c in df.columns if c not in metadata_cols]

#     splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
#     train_idx, val_idx = next(splitter.split(df, groups=df["PatientID"]))
#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     val_df = df.iloc[val_idx].reset_index(drop=True)

#     if RANK == 0:
#         logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Classes: {len(class_names)}")

#     cls_counts = calculate_class_counts(train_df, class_names)
#     freq = np.array(cls_counts, dtype=np.float32)
#     freq = freq / freq.sum()

#     return train_df, val_df, class_names, cls_counts, freq


from sklearn.model_selection import GroupShuffleSplit
import numpy as np

def load_and_split_data(split_seed):

    def filter_missing_files(df, npy_dir, synth_dir):
        valid_mask = []
        for _, row in df.iterrows():
          if row['synthetic']:
              path = os.path.join(synth_dir, row['ImageID'])
          else:
            path = os.path.join(npy_dir, f"{row['ImageID']}.npy")
          valid_mask.append(os.path.exists(path))
    
        return df[valid_mask].reset_index(drop=True)
    
    df_real = pd.read_csv(LABELS_CSV)
    df_real["synthetic"] = False
    
    removal = load_removal_list()
    if removal:
        df_real = df_real[~df_real["ImageID"].isin(removal)].reset_index(drop=True)

  
    metadata_cols = ["ImageID", "StudyDate_DICOM", "PatientID", "synthetic"]
    class_names = [c for c in df_real.columns if c not in metadata_cols]

    # splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=split_seed)
    train_real_idx, val_idx = next(splitter.split(df_real, groups=df_real["PatientID"].astype(str)))
    
    df_real_train = df_real.iloc[train_real_idx].reset_index(drop=True)
    val_df = df_real.iloc[val_idx].reset_index(drop=True) 


    df_synth = pd.read_csv("/home/shaotong/CMDC/CXR_LT-main/Task1_Long_tailed/synthetic_labels_tagged_cleaned.csv")    
    df_synth["synthetic"] = True
    if removal:
        df_synth = df_synth[~df_synth["ImageID"].isin(removal)].reset_index(drop=True)


    train_df = pd.concat([df_real_train, df_synth], ignore_index=True)

    if RANK == 0:
        logger.info(f"Train Total: {len(train_df)} (Real: {len(df_real_train)}, Synth: {len(df_synth)})")
        logger.info(f"Val (Real Only): {len(val_df)}")

    cls_counts = calculate_class_counts(train_df, class_names)
    freq = np.array(cls_counts, dtype=np.float32)
    freq = freq / freq.sum()

    return train_df, val_df, class_names, cls_counts, freq

# =============================================================================
# LABEL CO-OCCURRENCE GRAPH
# =============================================================================
def normalize_adj(adj: np.ndarray):
    adj = adj.astype(np.float32)
    deg = adj.sum(axis=1)
    deg = np.clip(deg, 1e-6, None)
    d_inv_sqrt = np.power(deg, -0.5)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    return adj_norm.astype(np.float32)


def build_label_adjacency(train_df, class_names,
                          smooth=1.0,
                          threshold=0.02,
                          self_loop=1.0):

    Y = train_df[class_names].values.astype(np.float32)
    Y = (Y > 0.5).astype(np.float32)   
    num_classes = Y.shape[1]

    cooc = Y.T @ Y   # (C, C)

  
    cls_count = Y.sum(axis=0) + smooth


    cond = cooc / cls_count[:, None]

 
    cond = 0.5 * (cond + cond.T)

 
    np.fill_diagonal(cond, 1.0)

  
    if threshold > 0:
        cond[cond < threshold] = 0.0
        np.fill_diagonal(cond, 1.0)


    adj = cond + np.eye(num_classes, dtype=np.float32) * self_loop

  
    adj = normalize_adj(adj)

    if RANK == 0:
        density = float((adj > 0).mean())
        logger.info(f"[LabelGraph] shape={adj.shape}, density={density:.4f}")

    return adj


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """
        x:   (C, Din)
        adj: (C, C)
        """
        x = self.fc(x)
        x = torch.matmul(adj, x)
        return x


class LabelGCNRefiner(nn.Module):
    def __init__(self, image_dim, num_classes, adj_matrix,
                 hidden_dim=256, alpha=0.3, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

        self.register_buffer("adj", torch.tensor(adj_matrix, dtype=torch.float32))

 
        self.label_emb = nn.Parameter(torch.randn(num_classes, hidden_dim) * 0.02)


        self.gc1 = GraphConvolution(hidden_dim, hidden_dim, bias=False)
        self.gc2 = GraphConvolution(hidden_dim, image_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

      
        self.image_proj = nn.Linear(image_dim, image_dim, bias=False)

  
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, image_feat, base_logits):
        """
        image_feat: (B, D)
        base_logits: (B, C)
        """
        label_proto = self.gc1(self.label_emb, self.adj)
        label_proto = F.relu(label_proto, inplace=True)
        label_proto = self.dropout(label_proto)
        label_proto = self.gc2(label_proto, self.adj)   # (C, D)

        label_proto = F.normalize(label_proto, dim=1)
        image_feat = F.normalize(self.image_proj(image_feat), dim=1)

        gcn_logits = torch.matmul(image_feat, label_proto.t()) + self.bias

        return base_logits + self.alpha * gcn_logits

# def load_and_split_data():

#     df_real = pd.read_csv(LABELS_CSV)
#     df_real["synthetic"] = False


#     df_synth = pd.read_csv("/home/shaotong/CMDC/CXR_LT-main/Task1_Long_tailed/synthetic_labels_tagged.csv")

  
#     df = pd.concat([df_real, df_synth], ignore_index=True)


#     removal = load_removal_list()
#     if removal:
#         df = df[~df["ImageID"].isin(removal)].reset_index(drop=True)

   
#     metadata_cols = ["ImageID", "StudyDate_DICOM", "PatientID", "synthetic"]
#     class_names = [c for c in df.columns if c not in metadata_cols]


#     splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
#     # train_idx, val_idx = next(splitter.split(df, groups=df["PatientID"]))
  
#     train_idx, val_idx = next(splitter.split(df, groups=df["PatientID"].astype(str)))
#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     val_df = df.iloc[val_idx].reset_index(drop=True)

#     cls_counts = calculate_class_counts(train_df, class_names)
#     freq = np.array(cls_counts, dtype=np.float32)
#     freq = freq / freq.sum()

#     return train_df, val_df, class_names, cls_counts, freq

# =============================================================================
# MODEL: ConvNeXtV2 + CSRA
# =============================================================================

class CSRA(nn.Module):
    def __init__(self, input_dim, num_classes, lam=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.classifier = nn.Linear(input_dim, num_classes)
        self.conv_att = nn.Conv2d(input_dim, num_classes, kernel_size=1, bias=False)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, input_dim // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 16, input_dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, return_feat=False):
        b, c, h, w = x.size()

        # GAP branch
        gap_feat = torch.mean(x, dim=(2, 3))
        logit_gap = self.classifier(gap_feat)

        # light channel gating
        gate = self.channel_gate(x)

        # residual gating
        x_refine = x + 0.05 * (x * gate)

 
        graph_feat = torch.mean(x_refine, dim=(2, 3))   # (B, c)

        # CSRA attention
        att_map = self.conv_att(x_refine).view(b, self.num_classes, h * w)
        att_score = self.softmax(att_map)

        x_flat = x_refine.view(b, c, h * w)
        csra_feat = torch.bmm(att_score, x_flat.permute(0, 2, 1))

        w_cls = self.classifier.weight
        logit_csra = torch.sum(csra_feat * w_cls.unsqueeze(0), dim=2) + self.classifier.bias

        logits = logit_gap + self.lam * logit_csra

        if return_feat:
            return logits, graph_feat
        return logits

from safetensors.torch import load_file


class ConvNeXtV2Classifier(nn.Module):
    def __init__(self, num_classes, adj_matrix=None):
        super().__init__()

        self.backbone = timm.create_model(
            "convnextv2_base",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.2,
            global_pool=""
        )

        weights_path = "/home/shaotong/CMDC/CXR_LT-main/Task1_Long_tailed/convnextv2_base_mimic-cxr_padchest_csra_dbcas.safetensors"

        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone."):
                    new_k = k.replace("backbone.", "")
                    new_state_dict[new_k] = v

            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            if RANK == 0:
                print(f"Medical backbone weights loaded: {msg}")

        nf = self.backbone.num_features
        self.bn = nn.BatchNorm2d(nf)

        self.head = CSRA(input_dim=nf, num_classes=num_classes, lam=0.1)

        if LABEL_GCN_ENABLED and adj_matrix is not None:
            self.label_gcn = LabelGCNRefiner(
                image_dim=nf,
                num_classes=num_classes,
                adj_matrix=adj_matrix,
                hidden_dim=LABEL_GCN_HIDDEN,
                alpha=LABEL_GCN_ALPHA,
                dropout=LABEL_GCN_DROPOUT
            )
        else:
            self.label_gcn = None

    def forward(self, x):
        x = self.backbone(x)
        x = self.bn(x)

        logits, image_feat = self.head(x, return_feat=True)

        if self.label_gcn is not None:
            logits = self.label_gcn(image_feat, logits)

        return logits

def load_pretrained_and_surgery(checkpoint_path, num_classes):
    if RANK == 0:
        logger.info("Loading MIMIC-CXR pretrained weights...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = ConvNeXtV2Classifier(num_classes=num_classes)
    model_sd = model.state_dict()

    filtered = {}
    for k, v in sd.items():
        if "classifier" in k or "head" in k:
            continue
        if k in model_sd and v.shape == model_sd[k].shape:
            filtered[k] = v

    model.load_state_dict(filtered, strict=False)

    nn.init.xavier_uniform_(model.head.classifier.weight)
    nn.init.constant_(model.head.classifier.bias, 0)
    nn.init.xavier_uniform_(model.head.conv_att.weight)

    return model

def load_checkpoint_for_resume(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    if RANK == 0:
        logger.info(f"Resuming from: {checkpoint_path} | epoch={ckpt.get('epoch', 'N/A')}")

    # Migration old arch -> new CSRA
    new_sd = {}
    is_migration = False
    if "classifier.0.weight" in sd and "head.classifier.weight" not in sd:
        if RANK == 0:
            logger.warning("Detected OLD ARCH -> migrating to CSRA")
        is_migration = True
        for k, v in sd.items():
            if "backbone" in k:
                new_sd[k] = v
            elif "classifier.0" in k:
                new_sd[k.replace("classifier.0", "bn")] = v
    else:
        new_sd = sd

    model.load_state_dict(new_sd, strict=not is_migration)

    if is_migration:
        nn.init.xavier_uniform_(model.head.classifier.weight)
        nn.init.constant_(model.head.classifier.bias, 0)
        nn.init.xavier_uniform_(model.head.conv_att.weight)
        return None

    return ckpt

def freeze_backbone(model):
    for n, p in model.named_parameters():
        if "backbone" in n:
            p.requires_grad = False
        else:
            p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# =============================================================================
# SCHEDULER
# =============================================================================
def cosine_with_warmup_lr_lambda(epoch_idx, total_epochs, warmup_epochs):
    if epoch_idx < warmup_epochs:
        return float(epoch_idx + 1) / float(max(1, warmup_epochs))
    t = (epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return max(0.01, 0.5 * (1.0 + np.cos(np.pi * t)))


# =============================================================================
# TRAIN / EVAL
# =============================================================================
def train_epoch(model, model_ema, loader, criterion, optimizer, scaler, epoch,synth_weight=0.5):
    model.train()
    if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)

    avg_loss = 0.0
    steps = 0

    iterator = tqdm(loader, desc=f"Train Ep {epoch}", disable=(RANK != 0))
    for images, labels, synthetic_flags in iterator:
        images = images.cuda(LOCAL_RANK, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.cuda(LOCAL_RANK, non_blocking=True)
        synthetic_flags = synthetic_flags.cuda(LOCAL_RANK, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits.float(), labels.float(), synthetic_flags=synthetic_flags, synth_weight=synth_weight)
        if torch.isnan(loss) or torch.isinf(loss):
            if RANK == 0:
                logger.warning("NaN/Inf loss. Skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        reduced = reduce_tensor(loss.detach())
        avg_loss += reduced.item()
        steps += 1

        if RANK == 0:
            iterator.set_postfix(loss=f"{reduced.item():.4f}")

    return avg_loss / max(1, steps)

# @torch.no_grad()
# def evaluate(model, loader, criterion):
#     model.eval()
#     avg_loss = 0.0
#     steps = 0
#     local_preds, local_labels = [], []

#     for images, labels, _ in tqdm(loader, desc="Val", disable=(RANK != 0)):
#         images = images.cuda(LOCAL_RANK, non_blocking=True).to(memory_format=torch.channels_last)
#         labels = labels.cuda(LOCAL_RANK, non_blocking=True)

#         with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=torch.float16):
#             logits = model(images)
#             loss = criterion(logits.float(), labels.float())

#         reduced = reduce_tensor(loss.detach())
#         avg_loss += reduced.item()
#         steps += 1

#         local_preds.append(torch.sigmoid(logits.float()).detach())
#         local_labels.append(labels.detach())

#     local_preds = torch.cat(local_preds).cuda(LOCAL_RANK)
#     local_labels = torch.cat(local_labels).cuda(LOCAL_RANK)

#     global_preds = gather_tensor(local_preds)
#     global_labels = gather_tensor(local_labels)

#     preds_np = global_preds.cpu().numpy()
#     labels_np = global_labels.cpu().numpy()

#     metrics = {"loss": avg_loss / max(1, steps), "mAP": 0.0}

#     if RANK == 0:
#         true_len = len(loader.dataset)
#         preds_np = preds_np[:true_len]
#         labels_np = labels_np[:true_len]

#         ap_scores = []
#         for i in range(labels_np.shape[1]):
#             if labels_np[:, i].sum() > 0:
#                 ap_scores.append(average_precision_score(labels_np[:, i], preds_np[:, i]))
#             else:
#                 ap_scores.append(0.0)

#         metrics["mAP"] = float(np.mean(ap_scores)) if ap_scores else 0.0
#         metrics["mAUC"] = float(np.mean(auc_scores)) if auc_scores else 0.0

    # return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, class_names=None, f1_threshold=F1_THRESHOLD):
    model.eval()
    avg_loss = 0.0
    steps = 0
    local_preds, local_labels = [], []

    for images, labels, synthetic_flags in tqdm(loader, desc="Val", disable=(RANK != 0)):
        images = images.cuda(LOCAL_RANK, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.cuda(LOCAL_RANK, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=AMP_ENABLED, dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits.float(), labels.float(), synthetic_flags=None)

        reduced = reduce_tensor(loss.detach())
        avg_loss += reduced.item()
        steps += 1

        local_preds.append(torch.sigmoid(logits.float()).detach())
        local_labels.append(labels.detach())

    local_preds = torch.cat(local_preds).cuda(LOCAL_RANK)
    local_labels = torch.cat(local_labels).cuda(LOCAL_RANK)

    global_preds = gather_tensor(local_preds)
    global_labels = gather_tensor(local_labels)

    metrics = {
        "loss": avg_loss / max(1, steps),
        "mAP": 0.0,
        "mAUC": 0.0,
        "macro_F1": 0.0,
        "per_class": {}
    }

    if RANK == 0:
        true_len = len(loader.dataset)
        preds_np = global_preds.cpu().numpy()[:true_len]
        labels_np = global_labels.cpu().numpy()[:true_len]
        preds_bin = (preds_np >= f1_threshold).astype(np.int32)

        num_classes = labels_np.shape[1]
        ap_scores = []
        auc_scores = []
        f1_scores = []
        per_class_metrics = {}

        for i in range(num_classes):
            y_true = labels_np[:, i].astype(np.int32)
            y_score = preds_np[:, i]
            y_pred = preds_bin[:, i]

            if y_true.sum() > 0:
                cur_ap = average_precision_score(y_true, y_score)
                if len(np.unique(y_true)) > 1:
                    cur_auc = roc_auc_score(y_true, y_score)
                else:
                    cur_auc = 0.5
            else:
                cur_ap = 0.0
                cur_auc = 0.5

            cur_f1 = f1_score(y_true, y_pred, zero_division=0)

            ap_scores.append(cur_ap)
            auc_scores.append(cur_auc)
            f1_scores.append(cur_f1)

            class_name = class_names[i] if class_names is not None else f"class_{i}"
            per_class_metrics[class_name] = {
                "AP": float(cur_ap),
                "AUC": float(cur_auc),
                "F1": float(cur_f1),
                "support": int(y_true.sum()),
            }

        metrics["mAP"] = float(np.mean(ap_scores)) if ap_scores else 0.0
        metrics["mAUC"] = float(np.mean(auc_scores)) if auc_scores else 0.0
        metrics["macro_F1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
        metrics["per_class"] = per_class_metrics

    return metrics
def load_best_model(ckpt_path, class_names):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = ConvNeXtV2Classifier(
        num_classes=len(class_names),
        adj_matrix=None
    )

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.cuda().eval()

    return model

@torch.no_grad()
def eval_single_split(model, val_df, class_names):
    val_ds = PadChestDataset(
        val_df,
        class_names,
        REAL_NPY_DIR,
        SYNTH_NPY_DIR,
        augment=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=fast_collate_fn
    )

    criterion = DistributionBalancedLoss(
        cls_counts=np.ones(len(class_names))
    ).cuda()

    metrics = evaluate(model, val_loader, criterion, class_names=class_names)

    return metrics
# =============================================================================
# MAIN
# =============================================================================
def main(split_seed=SPLIT_SEED):
    setup_ddp()
    set_seed(RANDOM_SEED)

    
    if RANK == 0:
        logger.info(f"DB Loss + CAS | GPUs={WORLD_SIZE} | Batch/GPU={BATCH_SIZE} | Global={BATCH_SIZE * WORLD_SIZE}")

    # Load data
    train_df, val_df, class_names, cls_counts, class_freq = load_and_split_data(SPLIT_SEED)
   # 构建标签共现图
    label_adj = build_label_adjacency(
       train_df=train_df,
       class_names=class_names,
       smooth=COOC_SMOOTH,
       threshold=COOC_THRESHOLD,
       self_loop=COOC_SELF_LOOP
     )
    train_ds = PadChestDataset(train_df, class_names, REAL_NPY_DIR, SYNTH_NPY_DIR, augment=True, img_size=IMAGE_SIZE)
    val_ds = PadChestDataset(val_df, class_names, REAL_NPY_DIR, SYNTH_NPY_DIR, augment=False, img_size=IMAGE_SIZE)

    # CAS Sampler (train only)
    labels_np = train_df[class_names].values.astype(np.float32)
    cas_sampler = RepeatFactorDistributedSampler(
        dataset=train_ds,
        labels_np=labels_np,
        class_freq=class_freq,
        threshold=CAS_T,
        max_repeat=CAS_MAX_REPEAT,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_sampler = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=cas_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=fast_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=fast_collate_fn,
        drop_last=False,
    )

    # Model
    model = ConvNeXtV2Classifier(
    num_classes=len(class_names),
    adj_matrix=label_adj
    ).cuda(LOCAL_RANK)
    model = model.to(memory_format=torch.channels_last)

    # Resume / pretrained
    resume_ckpt = None
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        resume_ckpt = load_checkpoint_for_resume(model, RESUME_CHECKPOINT)
        if RANK == 0 and resume_ckpt is None:
            logger.info("Migration happened -> reset optimizer/scheduler/scaler.")
    else:
        if RANK == 0:
            logger.info("Starting from MIMIC-CXR pretrained.")
        # model = ConvNeXtV2Classifier(num_classes=len(class_names)).cuda(LOCAL_RANK)
        # model = model.to(memory_format=torch.channels_last)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
   
    criterion = DistributionBalancedLoss(
        cls_counts=cls_counts,
        beta=DB_BETA,
        rebalance_alpha=DB_REBALANCE_ALPHA,
        neg_scale=DB_NEG_SCALE,
        neg_margin=DB_NEG_MARGIN,
        margin_scale=DB_MARGIN_SCALE,
        max_margin=DB_MAX_MARGIN,
    ).cuda(LOCAL_RANK)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # State
    if resume_ckpt:
       best_mAP = resume_ckpt.get("best_mAP", resume_ckpt.get("mAP", 0.0))
       global_epoch = resume_ckpt.get("epoch", 0)
       start_finetune_epoch = resume_ckpt.get("finetune_epoch", 0)
       best_metrics_at_best_map = resume_ckpt.get("best_metrics_at_best_map", None)
    else:
       best_mAP = 0.0
       global_epoch = 0
       start_finetune_epoch = 0
       best_metrics_at_best_map = None

    early_stop_counter = 0
    # PHASE 1: head warmup (if fresh or migration)
    if resume_ckpt is None:
        if RANK == 0:
            logger.info("PHASE 1: Warmup head-only")

        freeze_backbone(model.module)

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=WARMUP_HEAD_LR, weight_decay=0.01, eps=1e-5)
        scheduler = LambdaLR(optimizer, lambda e: 1.0)

        for e in range(1, WARMUP_HEAD_EPOCHS + 1):
            global_epoch += 1
            tr_loss = train_epoch(model, None, train_loader, criterion, optimizer, scaler, global_epoch, synth_weight=0.0)
            metrics = evaluate(model.module, val_loader, criterion, class_names=class_names)
            scheduler.step()

            if RANK == 0:
                # logger.info(f"[WARMUP] Ep {e} | TrainLoss={tr_loss:.4f} | ValLoss={metrics['loss']:.4f} | mAP={metrics['mAP']:.4f}")
                logger.info(
    f"[WARMUP] Ep {e} | TrainLoss={tr_loss:.4f} | "
    f"ValLoss={metrics['loss']:.4f} | mAP={metrics['mAP']:.4f} | "
    f"mAUC={metrics['mAUC']:.4f} | macro-F1={metrics['macro_F1']:.4f}"
)
                if metrics["mAP"] > best_mAP:
                    best_mAP = metrics["mAP"]

    else:
        if RANK == 0:
            logger.info("Skipping warmup (normal resume).")

    # PHASE 2: full finetune
    if RANK == 0:
        logger.info("PHASE 2: Full fine-tuning")

    model_unwrapped = model.module
    unfreeze_all(model_unwrapped)

    model_ema = ModelEmaV2(model_unwrapped, decay=EMA_DECAY, device=torch.device(f"cuda:{LOCAL_RANK}"))
    model = DDP(model_unwrapped, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False)

    backbone_params, head_params = [], []
    for n, p in model.module.named_parameters():
        if "backbone" in n:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR, "weight_decay": 0.01},
            {"params": head_params, "lr": HEAD_LR, "weight_decay": 0.05},
        ],
        eps=1e-5
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_with_warmup_lr_lambda(e, FINETUNE_EPOCHS, warmup_epochs=5)
    )

    # Load states if resume_ckpt (not migration)
    if resume_ckpt:
        if "optimizer_state_dict" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            except Exception:
                if RANK == 0:
                    logger.warning("Failed to load optimizer state -> restart optimizer.")
        if "scheduler_state_dict" in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            except Exception:
                pass
        if "scaler_state_dict" in resume_ckpt:
            try:
                scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
            except Exception:
                pass

    start_epoch = start_finetune_epoch + 1 if resume_ckpt else 1
    if RANK == 0 and resume_ckpt:
        logger.info(f"Resuming finetune from epoch {start_epoch}")

    for epoch in range(start_epoch, FINETUNE_EPOCHS + 1):
        global_epoch += 1
        # if isinstance(criterion, LDAMDRWLoss):
        #     criterion.update_epoch(epoch)
        tr_loss = train_epoch(model, model_ema, train_loader, criterion, optimizer, scaler, global_epoch, synth_weight=0.5)
        torch.cuda.empty_cache()

        metrics = evaluate(model_ema.module, val_loader, criterion, class_names=class_names)
        scheduler.step()

        if RANK == 0:
            # logger.info(f"[FINETUNE] Ep {epoch} | Train: {loss:.4f} | Val: {metrics['loss']:.4f} | mAP: {metrics['mAP']:.4f}")
            logger.info(
    f"[FINETUNE] Ep {epoch} | TrainLoss={tr_loss:.4f} | "
    f"ValLoss={metrics['loss']:.4f} | mAP={metrics['mAP']:.4f} | "
    f"mAUC={metrics['mAUC']:.4f} | macro-F1={metrics['macro_F1']:.4f}"
)
            
            if metrics['mAP'] > best_mAP:
                best_mAP = metrics['mAP']
                early_stop_counter = 0

                best_metrics_at_best_map = {
        'epoch': global_epoch,
        'finetune_epoch': epoch,
        'loss': float(metrics['loss']),
        'mAP': float(metrics['mAP']),
        'mAUC': float(metrics['mAUC']),
        'macro_F1': float(metrics['macro_F1']),
        'per_class': metrics['per_class'],
    }

                checkpoint = {
        'epoch': global_epoch,
        'finetune_epoch': epoch,
        'model_state_dict': model_ema.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_mAP': best_mAP,
        'mAP': float(metrics['mAP']),
        'mAUC': float(metrics['mAUC']),
        'macro_F1': float(metrics['macro_F1']),
        'class_names': class_names,
        'best_metrics_at_best_map': best_metrics_at_best_map,
        'per_class_metrics_at_best_map': metrics['per_class'],
    }

                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_padchest_finetune.pth"))
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{global_epoch}.pth"))

                logger.info(f" >>> Saved Best (mAP: {best_mAP:.4f})")
                logger.info(" >>> Per-class metrics at best mAP:")
                for cls_name, cls_metric in metrics['per_class'].items():
                   logger.info(
            f"     {cls_name}: AP={cls_metric['AP']:.4f} | "
            f"AUC={cls_metric['AUC']:.4f} | F1={cls_metric['F1']:.4f} | "
            f"support={cls_metric['support']}"
        )
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_padchest_finetune.pth"))
                # Also save a backup with epoch number
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{global_epoch}.pth"))
                logger.info(f" >>> Saved Best (mAP: {best_mAP:.4f})")
            else:
                early_stop_counter += 1
                logger.info(f"EarlyStopping Counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")

                if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                   logger.info(">>> Early stopping triggered. Stopping training.")
                   break


    cleanup_ddp()
    return {
    "best_mAP": best_mAP,
    "best_mAUC": best_metrics_at_best_map["mAUC"] if best_metrics_at_best_map else 0.0,
    "best_macroF1": best_metrics_at_best_map["macro_F1"] if best_metrics_at_best_map else 0.0,
    "checkpoint_path": os.path.join(CHECKPOINT_DIR, "best_padchest_finetune.pth"),
    "class_names": class_names,
    }


if __name__ == "__main__":

    result = main()

    print("\n===== FINAL RESULT =====")
    print(f"Best checkpoint: {result['checkpoint_path']}")
    print(f"Best mAP: {result['best_mAP']:.4f}")
    print(f"Best mAUC: {result['best_mAUC']:.4f}")
    print(f"Best macro-F1: {result['best_macroF1']:.4f}")
