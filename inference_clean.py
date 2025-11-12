import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from multiverseg.models.sp_mvs import MultiverSeg
from scribbleprompt.models.unet import ScribblePromptUNet
import universeg
import torch

# ======== 基本設定 ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== 自訂繪圖輔助函式 ========
def show_images(images, titles=None, ncols=2, width=5):
    """
    使用 matplotlib 顯示多張灰階影像。
    """
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(width * ncols, width * nrows))
    for i, img in enumerate(images):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img.cpu().squeeze(), cmap='gray')
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ======== 載入模型 ========
import universeg
import scribbleprompt
import multiverseg
from multiverseg.models.sp_mvs import MultiverSeg
from scribbleprompt.models.unet import ScribblePromptUNet

# ======== 自訂 BUSI Dataset 類別 ========
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

class BUSIDataset(Dataset):
    """
    BUSI Ultrasound Dataset (Final robust version)
    結構：
    Dataset_BUSI_with_GT/
        ├── benign/
        │   ├── images/
        │   └── mask/
        ├── malignant/
        │   ├── images/
        │   └── mask/
        └── normal/
            ├── images/
            └── mask/
    支援 mask 命名規則：xxx.png 對應 xxx_mask.png
    """
    def __init__(self, root, split='support', test_size=0.3, transform=None):
        from sklearn.model_selection import train_test_split
        from PIL import Image
        import numpy as np

        self.root = Path(root)
        self.transform = transform
        img_paths, mask_paths = [], []
        class_counts = {}

        for cls_folder in ['benign', 'malignant', 'normal']:
            image_dir = self.root / cls_folder / "images"
            mask_dir = self.root / cls_folder / "mask"

            if not image_dir.exists() or not mask_dir.exists():
                print(f"[WARN] Skip {cls_folder}, missing images/ or mask/")
                continue

            class_count = 0
            for img_file in image_dir.glob("*.*"):
                if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                    continue
                stem = img_file.stem  # e.g. "malignant (1)"
                # 嘗試對應 _mask 檔名
                mask_file = mask_dir / f"{stem}_mask{img_file.suffix}"
                if not mask_file.exists():
                    # 若沒有 _mask 後綴，嘗試同名
                    mask_file = mask_dir / f"{stem}{img_file.suffix}"
                if mask_file.exists():
                    img_paths.append(img_file)
                    mask_paths.append(mask_file)
                    class_count += 1
            class_counts[cls_folder] = class_count

        total = sum(class_counts.values())
        print(f"[INFO] Found {total} image-mask pairs in {root}")
        for cls, cnt in class_counts.items():
            print(f"   {cls}: {cnt} pairs")

        if total == 0:
            raise RuntimeError(f"No image-mask pairs found under {root}")

        # 顯示第一筆樣本確認
        print(f"[SAMPLE] First image: {img_paths[0]}")
        print(f"[SAMPLE] First mask : {mask_paths[0]}")

        # 分割支援/測試集
        train_idx, test_idx = train_test_split(
            range(len(img_paths)), test_size=test_size, random_state=42
        )
        idxs = train_idx if split == 'support' else test_idx
        self.img_paths = [img_paths[i] for i in idxs]
        self.mask_paths = [mask_paths[i] for i in idxs]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.img_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
    
        # --- 新增：統一大小 ---
        target_size = (256, 256)  # 或改成 (512, 512)
        img = img.resize(target_size)
        mask = mask.resize(target_size)
    
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0.5).astype(np.float32)
    
        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        return img, mask




# ======== 載入支援資料集與測試資料 ========
root = r"C:\limu\limu007\limu007\Dataset_BUSI_with_GT"
d_support = BUSIDataset(root=root, split='support', test_size=0.3)
d_test = BUSIDataset(root=root, split='test', test_size=0.3)

# 取樣支援資料集的部分影像
import itertools
support_images, support_labels = zip(*itertools.islice(d_support, 10))
support_images = torch.stack(support_images).to(device)
support_labels = torch.stack(support_labels).to(device)
print("支援集大小:", support_images.shape, support_labels.shape)

show_images(
    [support_images[i] for i in range(2)] +
    [support_labels[i] for i in range(2)],
    titles=["Support Image 1", "Support Image 2", "Label 1", "Label 2"],
    ncols=4
)



# ======== 初始化模型（完整預訓練版本，防錯版） ========
import torch
from multiverseg.models.sp_mvs import MultiverSeg, MultiverSegNet
from scribbleprompt.models.unet import ScribblePromptUNet
from universeg import universeg
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = r"C:\anaconda\envs\multiverseg\Lib\site-packages\checkpoints"

mvs_weight = os.path.join(checkpoint_dir, "MultiverSeg_v1_nf256_res128.pt")
sp_weight  = os.path.join(checkpoint_dir, "ScribblePrompt_unet_v1_nf192_res128.pt")
uvs_weight = os.path.join(checkpoint_dir, "universeg_v1_nf64_ss64_STA.pt")

print("\n[STEP] 檢查預訓練權重狀態：")
print(f" - MultiverSeg_v1_nf256_res128.pt: {'✅' if os.path.exists(mvs_weight) else '❌'}")
print(f" - ScribblePrompt_unet_v1_nf192_res128.pt: {'✅' if os.path.exists(sp_weight) else '❌'}")
print(f" - UniverSeg_v1_nf64_ss64_STA.pt: {'✅' if os.path.exists(uvs_weight) else '❌'}\n")

# ======== MultiverSeg 主體 ========
try:
    print(f"[INFO] 嘗試載入 MultiverSeg 權重：{mvs_weight}")
    model = MultiverSeg(version="v1").to(device)
    state = torch.load(mvs_weight, map_location=device)

    if "model" in state:
        model.multiverseg.load_state_dict(state["model"])
        print("[INFO] MultiverSeg: loaded submodule from 'model' key.")
    else:
        model.load_state_dict(state)
        print("[INFO] MultiverSeg: loaded full checkpoint.")
except Exception as e:
    print(f"[WARN] MultiverSeg 載入失敗 ({e})，改用空模型結構。")
    model = MultiverSeg.__new__(MultiverSeg)
    MultiverSeg.__init__(model, version="v1")
    model.to(device)

model.eval()
print("[INFO] ✅ MultiverSeg successfully initialized on", device)

# ======== ScribblePrompt 模型 ========
try:
    sp = ScribblePromptUNet(version='v1', device=device)
    print("[INFO] ✅ ScribblePrompt weights loaded successfully.")
except AssertionError:
    print("[WARN] ScribblePrompt 權重缺失，改用隨機初始化。")
    sp = ScribblePromptUNet(version='v1', device=device)
    sp.build_model(pretrained=False)

# ======== UniverSeg 模型 ========
if os.path.exists(uvs_weight):
    print(f"[INFO] Found UniverSeg checkpoint: {uvs_weight}")
    try:
        uvs = universeg(pretrained=False)
        state_uvs = torch.load(uvs_weight, map_location=device)
        missing, unexpected = uvs.load_state_dict(state_uvs, strict=False)
        print(f"[INFO] UniverSeg: loaded nf64_ss64_STA weights ({len(missing)} missing, {len(unexpected)} unexpected).")
        print("[INFO] ✅ Using official paper version (nf64_ss64_STA).")
    except Exception as e:
        print(f"[WARN] UniverSeg 載入失敗：{e}")
        uvs = universeg(pretrained=False)
else:
    print("[WARN] UniverSeg 權重不存在，改用隨機初始化。")
    uvs = universeg(pretrained=False)

uvs.to(device)
print("\n[✅] 所有模型已載入完成，準備進行 Inference。")






# ======== 選擇測試影像 ========
idx = np.random.permutation(len(d_test))[0]
image, label = d_test[idx]
image, label = image.to(device), label.to(device)
show_images([image, label], titles=['Test Image', 'Label'], ncols=2)

# ======== In-Context Segmentation ========
yhat = model.predict(image[None], support_images[None], support_labels[None], return_logits=False).to('cpu')
yhat_uvs = uvs(image[None], support_images[None], support_labels[None]).to('cpu').detach()
yhat_uvs = torch.sigmoid(yhat_uvs).squeeze()

show_images(
    [image, label, yhat_uvs > 0.5, yhat > 0.5],
    titles=['Image', 'Label', 'UniverSeg', 'MultiverSeg'],
    ncols=4
)



# ======== 定義互動提示生成器 ========
import importlib

def _import_obj(dotted: str):
    mod, name = dotted.rsplit('.', 1)
    return getattr(importlib.import_module(mod), name)

def eval_config(cfg):
    if isinstance(cfg, dict):
        if '_class' in cfg:
            cls = _import_obj(cfg['_class'])
            kwargs = {k: eval_config(v) for k, v in cfg.items() if k not in {'_class', '_fn'}}
            return cls(**kwargs)
        if '_fn' in cfg:
            return _import_obj(cfg['_fn'])
        return {k: eval_config(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return type(cfg)(eval_config(v) for v in cfg)
    return cfg

# ======== ScribblePrompt 組態 ========
random_warm_start = {
    "_class": "scribbleprompt.interactions.prompt_generator.FlexiblePromptEmbed",
    "click_embed": {"_fn": "scribbleprompt.interactions.embed.click_onehot"},
    "init_pos_click_generators": [{"_class": "scribbleprompt.interactions.clicks.RandomClick", "train": False}],
    "init_neg_click_generators": [{"_class": "scribbleprompt.interactions.clicks.RandomClick", "train": False}],
    "correction_click_generators": [{"_class": "scribbleprompt.interactions.clicks.ComponentCenterClick", "train": False}],
    "init_pos_click": 3,
    "init_neg_click": 3,
    "correction_clicks": 1,
    "prob_bbox": 0.0,
    "prob_click": 1.0,
    "from_logits": True,
}

prompt_generator = eval_config(random_warm_start)

# ======== 生成互動提示 ========
prompts = prompt_generator(image[None], label[None])
clicks = {k: prompts.get(k) for k in ['point_coords', 'point_labels']}

# ======== Interactive Segmentation ========
yhat = model.predict(image[None], **clicks, return_logits=False).to('cpu')
yhat_sp = sp.predict(image[None], **clicks).to('cpu')

show_images([image, label, yhat_sp > 0.5, yhat > 0.5],
            titles=['Image', 'Label', 'ScribblePrompt', 'MultiverSeg'], ncols=4)

# ======== Interactive In-Context Segmentation ========
yhat = model.predict(image[None], support_images[None], support_labels[None], **clicks, return_logits=False).to('cpu')
yhat_sp = sp.predict(image[None], **clicks).to('cpu')

show_images([image, label, yhat_sp > 0.5, yhat_uvs > 0.5, yhat > 0.5],
            titles=['Image', 'Label', 'ScribblePrompt', 'UniverSeg', 'MultiverSeg'], ncols=5)

# ======== 多輪互動修正 ========
yhat_logits = model.predict(image[None], support_images[None], support_labels[None], return_logits=True).to(device)

corrections = prompt_generator.subsequent_prompt(
    mask_pred=yhat_logits.to(device),
    binary_mask_pred=(yhat_logits.to(device) > 0).int(),
    prev_input=prompts
)
correction_clicks = {k: corrections.get(k) for k in ['point_coords', 'point_labels', 'mask_input']}

yhat2 = model.predict(image[None], support_images[None], support_labels[None], **correction_clicks, return_logits=False).to('cpu')

show_images([image, label, yhat > 0.0, yhat2 > 0.5],
            titles=['Image', 'Label', 'MultiverSeg (step 0)', 'MultiverSeg (step 1)'], ncols=4)

# ======== 評估指標 ========
y_pred = (yhat > 0.5).int().flatten()
y_true = label.int().flatten()
eps = 1e-8
TP = ((y_pred == 1) & (y_true == 1)).sum().item()
TN = ((y_pred == 0) & (y_true == 0)).sum().item()
FP = ((y_pred == 1) & (y_true == 0)).sum().item()
FN = ((y_pred == 0) & (y_true == 1)).sum().item()

accuracy  = (TP + TN) / (TP + TN + FP + FN + eps)
precision = TP / (TP + FP + eps)
recall    = TP / (TP + FN + eps)
iou       = TP / (TP + FP + FN + eps)
dice      = 2 * TP / (2 * TP + FP + FN + eps)

print("\n=== Segmentation Evaluation ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"IoU      : {iou:.4f}")
print(f"Dice     : {dice:.4f}")
