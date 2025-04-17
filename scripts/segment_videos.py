import os
import cv2
import glob
import tqdm
import json
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from PIL import Image
from einops import rearrange
from ultralytics import YOLO
from natsort import natsorted
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from transformers import AutoProcessor, CLIPSegForImageSegmentation


def load_class_dict(json_file):
    with open(json_file, 'r') as f:
        class_dict = json.load(f)
    return class_dict['class']


def get_class_indices(classes):
    # Invert the dictionary for easier lookup
    
    class_dict = load_class_dict('./assets/yolo_classes.json')
    class_to_index = {v: k for k, v in class_dict.items()}
    classes = classes.split(',')
    indices = []
    for cls in classes:
        if cls in class_to_index:
            indices.append(int(class_to_index[cls]))

    return indices
        

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


def load_yolo_model(yolo_checkpoint_dir, device='cpu'):
    # Load YOLO model
    model_yolo = YOLO(yolo_checkpoint_dir)
    model_yolo.to(device)
    return model_yolo


def load_clipseg_model(clipseg_checkpoint_dir, device='cpu'):
    # Load CLIPSeg model
    processor = AutoProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
    model = CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')
    model.to(device)
    return processor, model


def get_promt_masks_with_yolo(seg_model, 
                              sam_model,
                              sam_init_state, 
                              images, 
                              classes,
                              min_object_size=0.005,
                              detect_first_frame_only=True):
    if detect_first_frame_only:
        yolo_frames = [images[0]]
    else:
        yolo_frames = images
    object_identified = False
    yolo_ret = seg_model(yolo_frames, classes=classes)
    object_count = 0
    for y_i, ret in enumerate(yolo_ret):
        if len(ret.boxes.cls) > 0:
            
            xyxys = yolo_ret[y_i].boxes.xyxy
            xywhs = yolo_ret[y_i].boxes.xywh
            
            for b_i in range(len(xywhs)):
                xywh = xywhs[b_i]
                coor = xyxys[b_i]
                wh = xywh[2:]
                if not is_small_object(wh, ret.orig_shape, threshold=min_object_size):
                    object_identified = True
                    _, _, _ = sam_model.add_new_points_or_box(sam_init_state,
                                                              obj_id=object_count,
                                                              frame_idx=y_i,
                                                              box=coor)
                    object_count += 1
    yolo_ret = None
    xyxys = None
    xywhs = None
    xywh = None
    coor = None
    wh = None
    return sam_model, object_identified


def get_promt_masks_with_clipseg(seg_model,
                                 seg_model_processor,
                                 sam_model,
                                 sam_init_state,
                                 images,
                                 classes,
                                 detect_first_frame_only=True):
    if detect_first_frame_only:
        clipseg_frames = [images[0]]
    else:
        clipseg_frames = images
    object_identified = False 
    
    for i, image in enumerate(clipseg_frames):
        input_image = Image.open(image)
        inputs = seg_model_processor(
            text=classes, 
            images=[input_image] * len(classes), return_tensors="pt", padding=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        outputs = seg_model(**inputs)
        input_image = np.array(input_image)
        logits = outputs.logits.float()
        for j, logit in enumerate(logits):
            logit = logit.detach().cpu().numpy()
            logit = cv2.resize(logit, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.where(logit > 0, 1, 0)
            _, _, _ = sam_model.add_new_mask(sam_init_state, frame_idx=i, obj_id=j, mask=mask)
            object_identified = True

        
    return sam_model, object_identified


def segment_video(root_dir, 
                  save_dir,
                  sam_checkpoint_dir,
                  sam_model_cfg_dir,
                  seg_model_checkpoint_dir,
                  use_clipseg=False,
                  image_type='png',
                  device='cpu',
                  is_demo=False,
                  classes=None,
                  detect_first_frame_only=True,
                  min_object_size=0.005,
                  overlap_ratio_threshold=0.50,
                  mse_threshold=0.01):
    
    # Load the model
    checkpoint = sam_checkpoint_dir
    model_cfg = sam_model_cfg_dir
    sam_predictor = build_sam2_video_predictor(model_cfg, checkpoint).to(device)
    
    if not use_clipseg:
        '''
        # Full information about classes in YOLO can be found in here (not verified):
        # (https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8)
        '''
        seg_model = load_yolo_model(seg_model_checkpoint_dir, device=device)
    else:
        '''
        Huggingface model: https://huggingface.co/docs/transformers/en/model_doc/clipseg
        '''
        processor, seg_model = load_clipseg_model(seg_model_checkpoint_dir, device=device)
    
    print('Models Loaded Successfully')
    
    # Load the video
    video_root = root_dir
    assert os.path.exists(video_root), f'Video root {video_root} does not exist'
    if is_demo:
        save_dir = './output'
    assert os.path.exists(save_dir), f'Save directory {save_dir} does not exist'
    videos = natsorted(glob.glob(video_root + '/*/'))
    assert len(videos) > 0, f'No videos found in {video_root}'
    
    
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        video_count = 0
        for video_dir in tqdm.tqdm(videos):

            frames = sorted(glob.glob(video_dir + f'/*.{image_type}'))
            assert len(frames) > 0, f'No frames found in {video_dir} with type {image_type}'
            
            state = sam_predictor.init_state(video_path=video_dir)
            if not use_clipseg:
                sam_predictor, object_identified = get_promt_masks_with_yolo(seg_model,
                                                                      sam_predictor,
                                                                      state,
                                                                      frames,
                                                                      classes=classes,
                                                                      min_object_size=min_object_size,
                                                                      detect_first_frame_only=detect_first_frame_only)
            else:
                sam_predictor, object_identified = get_promt_masks_with_clipseg(seg_model,
                                                                      processor,
                                                                      sam_predictor,
                                                                      state,
                                                                      frames,
                                                                      classes=classes,
                                                                      detect_first_frame_only=detect_first_frame_only)
            
            if not object_identified:
                continue

            # propagate the prompts to get masklets throughout the video
            mask_seq = []
            for frame_idx, _, masks in sam_predictor.propagate_in_video(state, start_frame_idx=0):
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

            mask_seq = rearrange(torch.stack(mask_seq), 't c h w -> c t h w')
            mask_seq = torch.stack(
                remove_duplicates(mask_seq, mse_threshold=mse_threshold, overlap_ratio_threshold=overlap_ratio_threshold)
            )


            new_mask_seq = []
            for mask in mask_seq:
                sum_mask = torch.sum(mask).item()
                new_mask_seq.append((mask, sum_mask))
            new_mask_seq = sorted(new_mask_seq, key=lambda x: x[1], reverse=True)[:4]
            
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
                    os.makedirs(f'{save_dir}/object{m}', exist_ok=True)
                    cv2.imwrite(f'{save_dir}/object{m}/{t}.png', temp)
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
            
            if device.find('cuda') != -1:
                torch.cuda.empty_cache()

            video_count += 1

            if is_demo:
                break



def main():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--video_dir', type=str, default='./dummy_video_dataset')
    parser.add_argument('--save_dir', type=str, default='./output')
    
    # model configs
    parser.add_argument('--sam_checkpoint', type=str, default='./sam2/checkpoints/sam2.1_hiera_large.pt')
    parser.add_argument('--sam_model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml')
    parser.add_argument('--use_clipseg', action='store_true')
    parser.add_argument('--yolo_checkpoint', type=str, default='./checkpoints/yolo11x-seg.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--classes', type=str, default='person')
    
    # other configs
    parser.add_argument('--is_demo', action='store_true')
    parser.add_argument('--image_type', type=str, default='jpg')
    parser.add_argument('--detect_first_frame_only', action='store_true')
    
    # object detection configs
    parser.add_argument('--min_object_size', type=float, default=0.005)
    parser.add_argument('--overlap_ratio_threshold', type=float, default=0.50)
    parser.add_argument('--mse_threshold', type=float, default=0.01)
    
    
    args = parser.parse_args()

    video_dir = args.video_dir
    use_clipseg = args.use_clipseg
    save_dir = args.save_dir
    sam_checkpoint = args.sam_checkpoint
    sam_model_cfg = args.sam_model_cfg
    yolo_checkpoint = args.yolo_checkpoint
    device = args.device
    is_demo = args.is_demo
    image_type = args.image_type
    detect_first_frame_only = args.detect_first_frame_only
    min_object_size = args.min_object_size
    overlap_ratio_threshold = args.overlap_ratio_threshold
    mse_threshold = args.mse_threshold
    
    classes = args.classes
    if not use_clipseg:
        classes = get_class_indices(classes)
    else:
        classes = classes.split(',')
        classes = [x.strip() for x in classes]
    print(f'Classes: {classes}')
    # segment the video
    segment_video(video_dir, 
                  save_dir,
                  sam_checkpoint,
                  sam_model_cfg,
                  yolo_checkpoint,
                  use_clipseg=use_clipseg,
                  image_type=image_type,
                  device=device,
                  is_demo=is_demo,
                  classes=classes,
                  detect_first_frame_only=detect_first_frame_only,
                  min_object_size=min_object_size,
                  overlap_ratio_threshold=overlap_ratio_threshold,
                  mse_threshold=mse_threshold)    
    
    print('Segmentation Completed Successfully!')
    
    
if __name__ == '__main__':
    main()

