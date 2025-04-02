import os
import cv2
import glob
import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from einops import rearrange
from ultralytics import YOLO
from natsort import natsorted
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

def remove_duplicates(mask_seq, mse_threshold=0.01, overlap_ratio_threshold=0.50):
    # Here, mask_seq has shape (num_masks, Time, H, W)
    new_mask_seq = []
    for m_i, mask in enumerate(mask_seq):
        # For each mask sequence, determine which time frames are active (nonzero)
        candidate_valid = (mask.view(mask.size(0), -1).sum(dim=1) > 0)
        candidate_sum = mask.sum().item()  # overall "completeness"
        candidate_active_count = candidate_valid.sum().item()

        is_unique = True
        for a_i, added in enumerate(new_mask_seq):
            added_valid = (added.view(added.size(0), -1).sum(dim=1) > 0)
            added_sum = added.sum().item()

            # Determine common active frames
            common_valid = candidate_valid & added_valid
            common_count = common_valid.sum().item()

            # Calculate the overlap ratio: how many of the candidate's active frames overlap?
            overlap_ratio = common_count / candidate_active_count if candidate_active_count > 0 else 0

            # Only compare if a large portion of the candidate is overlapping with the added sequence.
            if overlap_ratio >= overlap_ratio_threshold:
                
                candidate_common = mask[common_valid].float()
                added_common = added[common_valid].float()
                loss = F.mse_loss(candidate_common, added_common)
                if loss < mse_threshold:
                    # They are considered similar.
                    # Replace the added sequence if the candidate is more "complete"
                    if candidate_sum > added_sum:
                        new_mask_seq[a_i] = mask
                    is_unique = False
                    break

        if is_unique:
            new_mask_seq.append(mask)

    return new_mask_seq

def is_small_object(size, canvas, threshold=0.005):
    full_area = canvas[0] * canvas[1]
    obj_area = size[0] * size[1]
    portion = obj_area / full_area
    if portion < threshold:
        return True
    else:
        return False

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint).to('cuda:0')
model_yolo = YOLO('yolo11x-seg.pt') 
keep_classes = [2, 3, 5, 6, 7]
video_root = '/home/eliyas/datasets/KITTI/splitted'
# video_root = './temp'
videos = natsorted(os.listdir(video_root))
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    video_count = 0
    for video_dir in tqdm.tqdm(videos):

        video_dir = os.path.join(video_root, video_dir)
        frames = sorted(glob.glob(os.path.join(video_dir, '*.png')))
        if len(frames) != 10:
            continue
        state = predictor.init_state(video_path=video_dir)
        print(f'Processing {video_dir} with {len(frames)} frames')
        yolo_ret = model_yolo(frames, classes=keep_classes)
        object_count = 0
        object_identified = False
        for y_i, ret in enumerate(yolo_ret):
            if len(ret.boxes.cls) > 0:
                
                xyxys = yolo_ret[y_i].boxes.xyxy
                xywhs = yolo_ret[y_i].boxes.xywh

                for b_i in range(len(xywhs)):
                    xywh = xywhs[b_i]
                    coor = xyxys[b_i]
                    wh = xywh[2:]
                    if not is_small_object(wh, ret.orig_shape):
                        object_identified = True
                        frame_idx, obj_id, _ = predictor.add_new_points_or_box(state,
                                                                  obj_id=object_count,
                                                                  frame_idx=y_i,
                                                                  box=coor)
                        object_count += 1
        if not object_identified:
            continue
        yolo_ret = None
        xyxys = None
        xywhs = None
        xywh = None
        coor = None
        wh = None

        # propagate the prompts to get masklets throughout the video
        mask_seq = []
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0):
            all_masks = []
            for m_i, mask in enumerate(masks):
                mask = mask.squeeze().cpu()
                mask = torch.where(mask > 0, 1, 0)
                all_masks.append(mask)
            all_masks = torch.stack(all_masks)
            mask_seq.append(all_masks)
        all_masks = None
        masks = None
        mask = None
             
        mask_seq = torch.stack(mask_seq)
        
        mask_seq = rearrange(mask_seq, 't c h w -> c t h w')
        torch.save(mask_seq, f'mask_seq.pt')
        new_mask_seq = remove_duplicates(mask_seq)
        
                    
        mask_seq = torch.stack(new_mask_seq)
        new_mask_seq = []
        for mask in mask_seq:
            sum_mask = torch.sum(mask).item()
            new_mask_seq.append((mask, sum_mask))
        new_mask_seq = sorted(new_mask_seq, key=lambda x: x[1], reverse=True)[:4]
        
        print(f'Original: {mask_seq.shape}, New: {len(new_mask_seq)}')
        save_dir = video_dir.replace('splitted', 'segmented')
        mask_seq = torch.stack([x[0] for x in new_mask_seq])   
        mask_seq = rearrange(mask_seq, 'c t h w -> t c h w')
        for t in range(mask_seq.shape[0]):
            image = cv2.imread(frames[t])
            masks = mask_seq[t]
            for m in range(mask_seq.shape[1]):
                mask = masks[m].cpu().numpy()
                temp = image.copy()
                
                # apply mask
                temp[mask == 0] = 0
                os.makedirs(f'{save_dir}/s{m}', exist_ok=True)
                cv2.imwrite(f'{save_dir}/s{m}/{t}.png', temp)
                temp = None
            
            all_others = masks.sum(0)
            all_others = torch.where(all_others > 0, 0, 1)
            all_others = all_others.cpu().numpy()
            image[all_others == 0] = 0
            os.makedirs(f'{save_dir}/background', exist_ok=True)
            cv2.imwrite(f'{save_dir}/background/{t}.png', image)
        
        mask_seq = None
        new_mask_seq = None
        masks = None
        mask = None
        image = None
        frames = None
        torch.cuda.empty_cache()
        
        video_count += 1
        
        # if video_count == 10:
        #     break