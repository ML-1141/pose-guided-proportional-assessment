


from PIL import Image
from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d

import _util.keypoints_v0 as ukey
import os
import csv

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input_dir')   # ⭐ 要處理的資料夾
parser.add_argument('fn_model')    # 模型 checkpoint
parser.add_argument('output_dir')  # ⭐ 輸出資料夾
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)


######################## SEGMENTER ########################

from _train.character_bg_seg.models.alaska import Model as CharacterBGSegmenter
model_segmenter = CharacterBGSegmenter.load_from_checkpoint(
    './_train/character_bg_seg/runs/eyeless_alaska_vulcan0000/checkpoints/'
    'epoch=0096-val_f1=0.9508-val_loss=0.0483.ckpt'
)

def abbox(img, thresh=0.5, allow_empty=False):
    # get bbox from alpha image, at threshold
    img = I(img).np()
    assert len(img) in [1,4], 'image must be mode L or RGBA'
    a = img[-1] > thresh
    xlim = np.any(a, axis=1).nonzero()[0]
    ylim = np.any(a, axis=0).nonzero()[0]
    if len(xlim)==0 and allow_empty: xlim = np.asarray([0, a.shape[0]])
    if len(ylim)==0 and allow_empty: ylim = np.asarray([0, a.shape[1]])
    axmin,axmax = max(int(xlim.min()-1),0), min(int(xlim.max()+1),a.shape[0])
    aymin,aymax = max(int(ylim.min()-1),0), min(int(ylim.max()+1),a.shape[1])
    return [(axmin,aymin), (axmax-axmin,aymax-aymin)]

def infer_segmentation(self, images, bbox_thresh=0.5, return_more=True):
    anss = []
    _size = self.hparams.largs.bg_seg.size
    self.eval()
    for img in images:
        oimg = img
        # img = a2bg(resize_min(img, _size).convert('RGBA'),1).convert('RGB')
        img = I(img).resize_min(_size).convert('RGBA').alpha_bg(1).convert('RGB').pil()
        timg = TF.to_tensor(img)[None].to(self.device)
        with torch.no_grad():
            out = self(timg)
        ans = TF.to_pil_image(out['softmax'][0,1].float().cpu()).resize(oimg.size[::-1])
        ans = {'segmentation': I(ans)}
        ans['bbox'] = abbox(ans['segmentation'], thresh=bbox_thresh, allow_empty=True)
        anss.append(ans)
    return anss


######################## POSE ESTIMATOR ########################

if 'feat_concat' in args.fn_model:
    from _train.character_pose_estim.models.passup import Model as CharacterPoseEstimator
elif 'feat_match' in args.fn_model:
    from _train.character_pose_estim.models.fermat import Model as CharacterPoseEstimator
else:
    assert 0, 'must use one of the provided pose estimation models'
model_pose = CharacterPoseEstimator.load_from_checkpoint(args.fn_model, strict=False)

def infer_pose(self, segmenter, images, smoothing=0.1, pad_factor=1):
    self.eval()
    try:
        largs = self.hparams.largs.adds_keypoints
    except:
        largs = self.hparams.largs.danbooru_coco
    _s = largs.size
    _p = _s * largs.padding
    anss = []
    segs = infer_segmentation(segmenter, images)
    for img,seg in zip(images,segs):
        # segment
        oimg = img
        ans = {
            'segmentation_output': seg,
        }
        bbox = seg['bbox']
        cb = u2d.cropbox_sequence([
            # crop to bbox, resize to square, pad sides
            [bbox[0], bbox[1], bbox[1]],
            resize_square_dry(bbox[1], _s),
            [-_p*pad_factor/2, _s+_p*pad_factor, _s],
        ])
        icb = u2d.cropbox_inverse(oimg.size, *cb)
        img = u2d.cropbox(img, *cb)
        img = img.convert('RGBA').alpha(0).convert('RGB')
        ans['bbox'] = bbox
        ans['cropbox'] = cb
        ans['cropbox_inverse'] = icb
        ans['input_image'] = img
        
        # pose estim
        timg = img.tensor()[None].to(self.device)
        with torch.no_grad():
            out = self(timg, smoothing=smoothing, return_more=True)
        ans['out'] = out
        
        # post-process keypoints
        kps = out['keypoints'][0].cpu().numpy()
        kps = u2d.cropbox_points(kps, *icb)
        ans['keypoints'] = kps
        
        anss.append(ans)
    return anss

######################## MODEL FORWARD ########################

def _visualize(image=None, bbox=None, keypoints=None):
    """
    建一張和原圖一樣大小的透明底圖，只在上面畫骨架（不畫 bbox）。
    """
    # 取得原圖尺寸
    base = I(image)
    w, h = base.pil().size

    # 建立全透明畫布（RGBA，alpha=0）
    canvas = I(Image.new('RGBA', (w, h), (0, 0, 0, 0)))
    v = canvas

    # 只畫骨架 & 關節
    if keypoints is not None:
        # 若是 dict，就依照 coco_keypoints 順序轉成 array
        if isinstance(keypoints, dict):
            keypoints = np.asarray([keypoints[k] for k in ukey.coco_keypoints])

        # 畫骨架線（線條加粗：w=10）
        for (a, b), c in zip(ukey.coco_parts, ukey.coco_part_colors):
            v = v.line(keypoints[a], keypoints[b], w=14, c=c)

        # 畫關節點（點加大：s=8）
        keypoints = keypoints[:len(ukey.coco_keypoints)]
        for kp in keypoints:
            v = v.dot(kp, s=5, c='r')

    return v

import csv

# ⭐ 確保輸出資料夾存在
os.makedirs(output_dir, exist_ok=True)

# 支援的副檔名
valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

# ⭐ 準備輸出 keypoints 的 CSV 檔
csv_path = output_dir / 'keypoints.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)

    # ---- 建立欄位名稱 ----
    header = ['filename']
    for k in ukey.coco_keypoints:
        header.append(f'{k}_x')
        header.append(f'{k}_y')

    # ⭐ 另外加上肩膀中點 & 骨盆中點
    header += ['shoulder_mid_x', 'shoulder_mid_y',
               'pelvis_mid_x', 'pelvis_mid_y']
    writer.writerow(header)

    # ⭐ 用實際名稱找 index（肩膀 & 骨盆）
    idx_l_sh = ukey.coco_keypoints.index('shoulder_left')
    idx_r_sh = ukey.coco_keypoints.index('shoulder_right')
    idx_l_hip = ukey.coco_keypoints.index('hip_left')
    idx_r_hip = ukey.coco_keypoints.index('hip_right')

    # ⭐ 所有左右成對關節的名稱
    left_right_name_pairs = [
        ('eye_left',      'eye_right'),
        ('ear_left',      'ear_right'),
        ('shoulder_left', 'shoulder_right'),
        ('elbow_left',    'elbow_right'),
        ('wrist_left',    'wrist_right'),
        ('hip_left',      'hip_right'),
        ('knee_left',     'knee_right'),
        ('ankle_left',    'ankle_right'),
    ]

    # 轉成 index pair，之後用來交換座標
    lr_idx_pairs = []
    for lname, rname in left_right_name_pairs:
        if lname in ukey.coco_keypoints and rname in ukey.coco_keypoints:
            li = ukey.coco_keypoints.index(lname)
            ri = ukey.coco_keypoints.index(rname)
            lr_idx_pairs.append((li, ri))

    # 走訪 input_dir 裡所有檔案
    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in valid_exts:
            continue

        print(f'\n=== processing {img_path} ===')

        # 讀圖
        img = I(str(img_path))
        w, h = img.pil().size  # ⭐ 圖片寬高，h 會用在座標變換裡

        # 跑 pose 推論
        ans = infer_pose(model_pose, model_segmenter, [img,])

        bbox = ans[0]['bbox']
        print(f'bounding box\n\ttop-left: {bbox[0]}\n\tsize: {bbox[1]}')
        print()

        # 取出 keypoints（只拿前面 coco 定義的 17 個點）
        all_kps = ans[0]['keypoints']
        kps = all_kps[:len(ukey.coco_keypoints)]  # shape (17, 2)

        print('keypoints (original, before transform)')
        for k, (x, y) in zip(ukey.coco_keypoints, kps):
            print((f'\t({x:.2f}, {y:.2f})'), k)
        print()

        # =========================
        # ⭐ 座標變換 + 左右交換
        # =========================

        # 1) 先做座標系變換：
        #    x' = y
        #    y' = h - x
        kps_transformed = kps.copy()
        x = kps[:, 0].copy()
        y = kps[:, 1].copy()
        kps_transformed[:, 0] = y         # 新 x'
        kps_transformed[:, 1] = h - x     # 新 y'

        # 2) 再做左右關節交換：
        #    left/right 成對的 index，把整個 (x', y') pair 互換
        kps_out = kps_transformed.copy()
        for li, ri in lr_idx_pairs:
            tmp = kps_out[li].copy()
            kps_out[li] = kps_out[ri]
            kps_out[ri] = tmp

        kps_out[:, 1] = h - kps_out[:, 1]

        print('keypoints (after transform + L/R swap)')
        for k, (x2, y2) in zip(ukey.coco_keypoints, kps_out):
            print((f'\t({x2:.2f}, {y2:.2f})'), k)
        print()

        # ⭐ 用變換 & 交換後的座標來算中點
        shoulder_mid = (kps_out[idx_l_sh] + kps_out[idx_r_sh]) / 2.0
        pelvis_mid   = (kps_out[idx_l_hip] + kps_out[idx_r_hip]) / 2.0

        # ⭐ 把這張圖的 keypoints + 中點寫進 CSV
        row = [img_path.name]
        for (x2, y2) in kps_out:
            row.append(f'{x2:.4f}')
            row.append(f'{y2:.4f}')

        # 加上中點
        row.append(f'{shoulder_mid[0]:.4f}')
        row.append(f'{shoulder_mid[1]:.4f}')
        row.append(f'{pelvis_mid[0]:.4f}')
        row.append(f'{pelvis_mid[1]:.4f}')

        writer.writerow(row)

        # ⭐ 決定輸出檔名：沿用原檔名 + ".png"
        out_name = img_path.stem + '.png'
        out_path = output_dir / out_name

        # ⭐ 視覺化目前仍用原本 all_kps（沒做座標變換）
        #    如果你希望圖也照新座標變換，可以把 all_kps 換成對應版本
        _visualize(img, keypoints=all_kps).save(out_path)
        print(f'output saved to {out_path}')
