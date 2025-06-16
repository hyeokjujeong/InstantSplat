import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import argparse

import torch
import shutil
from PIL import Image
import os

from time import time
from utils.sfm_utils import save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks, init_filestructure, get_sorted_image_files, split_train_test, load_images, compute_co_vis_masks, load_images_single_channel
from utils.camera_utils import generate_interpolated_path
from utils.mask_utils import get_corresponding_mask_paths, get_object_masks
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


from PIL import Image
import os
def extract_num(path):
    # 파일명만 추출 (예: 0012.jpg)
    filename = os.path.basename(path)
    # 확장자 제거 후 숫자로 변환 (예: 0012)
    num = int(os.path.splitext(filename)[0])
    return num

def get_valid_matches(fmoutput, fmodel, idx, device):
    desc1, desc2 = fmoutput['pred1']['desc'][idx].squeeze(0).detach(), fmoutput['pred2']['desc'][idx].squeeze(0).detach()
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13)

    H0, W0 = fmoutput['view1']['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = fmoutput['view2']['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    return matches_im0, matches_im1




def get_correspondance_mat(mask0, mask1, matches_im0, matches_im1, threshold=0.01):
    print('get_corr_mat start')
    resized_mask0 = (mask0).T
    resized_mask1 = (mask1).T

    print(resized_mask0)


    mask0_size = torch.zeros((torch.max(resized_mask0)+1)).long()
    for i in range(torch.max(resized_mask0).item()+1):
        mask0_size[i] = (resized_mask0==i).sum().item()
    mask1_size = torch.zeros((torch.max(resized_mask1)+1)).long()
    for i in range(torch.max(resized_mask1).item()+1):
        mask1_size[i] = (resized_mask1==i).sum().item()
    correspondances = torch.zeros((torch.max(resized_mask0)+1, torch.max(resized_mask1)+1))
    corr_tf = -torch.ones((torch.max(resized_mask0)+1, torch.max(resized_mask1)+1))
    xs0, ys0 = matches_im0[:,0], matches_im0[:,1]
    im0_mask_idx = resized_mask0[xs0, ys0]
    xs1, ys1 = matches_im1[:,0], matches_im1[:,1]
    im1_mask_idx = resized_mask1[xs1, ys1]
    print(im0_mask_idx)
    for i in range(len(im0_mask_idx)):
        correspondances[im0_mask_idx[i], im1_mask_idx[i]]+=1
        #corr_tf[im0_mask_idx[i], im1_mask_idx[i]] = 1
    for i in range(torch.max(resized_mask0).item()+1):
        for j in range(torch.max(resized_mask1).item()+1):
            min_size = min(mask0_size[i], mask1_size[j])
            if correspondances[i, j]/min_size > threshold:
                corr_tf[i, j] = 1
    zero_one_corr = correspondances.argmax(dim=1)
    one_zero_corr = correspondances.argmax(dim=0)
    temp_corr = []
    for i in range(len(zero_one_corr)):
        if corr_tf[i,zero_one_corr[i]]==1:
            temp_corr.append([i, zero_one_corr[i].item()])
    for i in range(len(one_zero_corr)):
        if corr_tf[one_zero_corr[i], i]==1:
            if [one_zero_corr[i], i] not in temp_corr:
                temp_corr.append([one_zero_corr[i].item(), i])
    for i in range(torch.max(resized_mask0).item()+1):
        marker = False
        for temp in temp_corr:
            if temp[0] == i:
                marker = True
                break
        if marker == False:
            temp_corr.append([i, -1])

    for i in range(torch.max(resized_mask1).item()+1):
        marker = False
        for temp in temp_corr:
            if temp[1] == i:
                marker = True
                break
        if marker == False:
            temp_corr.append([-1, i])
    print(f'Mask Correspondances: {temp_corr}')
    return temp_corr


def init_obj_dict(train_img_files, first_mask):
    obj_dict = {}
    obj_dict['0'] = []
    init_sam2_mask = first_mask #np.load(train_img_dir+'/init_mask.npy')
    init_sam1_mask = np.array(Image.open(train_img_files[0]).convert('L'))
    for i in range( np.max(init_sam1_mask).item()):
        temp_mask = (init_sam1_mask==i)
        cap_mask = np.logical_and(init_sam2_mask, temp_mask).sum()
        temp_mask_sum = temp_mask.sum()
        if cap_mask/temp_mask_sum > 0.5:
            obj_dict['0'].append(i)
    return obj_dict
def update_single_object(obj_dict, temp_corr, n):
    import copy
    np1 = n+1
    obj_dict[str(np1)] = []
    for temp in temp_corr:
        if temp[0] in obj_dict[str(n)] and temp[1]!=-1:
            obj_dict[str(np1)].append(temp[1])
    obj_dict[str(np1)] = list(set(obj_dict[str(np1)]))
    return obj_dict


def extract_instance_mask(mask, label, x_size, y_size):
    #mask = cv2.imread(mask_path)#, cv2.IMREAD_GRAYSCALE
    
    #if mask is None:
    #    raise ValueError(f"Failed to read mask image from {mask_path}")
    target_size = (x_size, y_size)
    
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (resized_mask == label).astype(np.uint8)


def iou_score(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def find_best_iou_for_pred(pred_mask, gt_masks):
    best_iou = 0
    best_gt_idx = -1
    for i, gt_mask in enumerate(gt_masks):
        iou = iou_score(pred_mask, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = i
    return best_iou, best_gt_idx

def get_first_mask(mask_dir, gt_dir, px, py, thr=0.5):
    gt_mask = np.array(Image.open(gt_dir).convert('RGB'))
    ys, xs, cs = gt_mask.shape
    sam_mask_temp = np.array(Image.open(mask_dir).convert('L'))
    sam_mask = cv2.resize(sam_mask_temp, (xs,ys), interpolation=cv2.INTER_NEAREST)
    final_mask = np.zeros(sam_mask.shape, dtype=np.uint8)
    gt_mask_perm = gt_mask
    print(gt_mask_perm.shape)
    gt_color = gt_mask_perm[py, px]
    print(gt_color.shape)
    gt_mask_binary = np.all(gt_mask_perm == gt_color, axis=-1)
    init_obj = {}
    init_obj['0']=[]
    for i in range(np.max(sam_mask).item()):
        temp_mask = (sam_mask==i)
        cap_mask = np.logical_and(gt_mask_binary, temp_mask).sum()
        temp_mask_sum = temp_mask.sum()
        if cap_mask/temp_mask_sum > thr:
            final_mask[temp_mask] = 1
            init_obj['0'].append(i)
    
    return final_mask, gt_color, init_obj


def main(data_dir, video_dir,n_view, px, py, threshold):
    test_dir = video_dir + '/test'
    final_video_dir = video_dir+'/train'
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    #sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    #model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained('./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth').to(device)


    
    image_size  = 640
    for test_img in test_files:
        src_path = os.path.join(test_dir, test_img)
        dst_path = os.path.join(final_video_dir, test_img)
        shutil.copyfile(src_path, dst_path)

        
        image_dir = final_video_dir
        image_files, image_suffix = get_sorted_image_files(image_dir)


        frame_names = [p for p in os.listdir(final_video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    
        frame_names_sorted = sorted(frame_names, key=extract_num)
        gt_dir = data_dir+'/vis_sem_instance/'
        mask_dir = data_dir+'/raw_sam_mask'
        mask_names = [mask_dir+'/train_rgb_'+os.path.splitext(p)[0]+'.png' for p in frame_names_sorted]
        gt_names = [gt_dir+'/train_vis_sem_instance_'+os.path.splitext(p)[0]+'.png' for p in frame_names_sorted]
        first_mask , obj_color, obj_dict = get_first_mask(mask_names[0], gt_names[0], px, py)

        # when geometry init, only use train images
        #image_files = train_img_files
        images, org_imgs_shape = load_images(image_files, size=image_size)

        start_time = time()
        print(f'>> Making pairs...')
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        print(f'>> Inference...')
        output = inference(pairs, model, device, batch_size=1, verbose=True)
        
        frame_names_sorted = sorted(frame_names, key=extract_num)
        mask_names = [mask_dir+'/train_rgb_'+os.path.splitext(p)[0]+'.png' for p in frame_names_sorted]
        gt_names = [gt_dir+'train_vis_sem_instance_'+os.path.splitext(p)[0]+'.png' for p in frame_names_sorted]

        #train_mask_files = get_corresponding_mask_paths(image_files)
        masks, org_masks_shape = load_images_single_channel(mask_names, image_size)
        mask_list = []
        for i in range(len(masks)):
            mask_list.append((255*masks[i]['img']).squeeze(0).squeeze(0).long())

        while i!= len(image_files)-1:
            for j in range(len(pairs)):
                if pairs[j][0]["idx"]==i and pairs[j][1]["idx"]==i+1:
                    matches_im0, matches_im1 = get_valid_matches(output, model, j, device)
                    temp_corr = get_correspondance_mat(masks_list[i], masks_list[i+1], matches_im0, matches_im1, threshold)
                    obj_dict = update_single_object(obj_dict, temp_corr, i)

        print(obj_dict)
        gt_mask_dir = os.path.join(data_dir, 'vis_sem_instance')
        number_part = test_img.split('.')[0]
        number_part = int(number_part)
        if number_part > 1000:
            number_part -=1000
        mask_single = os.path.join(gt_mask_dir, f'train_vis_sem_instance_{number_part:04d}.png')
        mask = Image.open(mask_single).convert('RGB')
        mask_np = np.array(mask)
        unique_colors = np.unique(mask_np.reshape(-1, 3), axis=0)
        num_unique_colors = len(unique_colors)
        color_to_label = {tuple(color): idx for idx, color in enumerate(unique_colors)}
        new_mask = np.zeros(mask_np.shape[:2], dtype = np.uint8)

        for color, label in color_to_label.items():
            if np.all(color == obj_color):
                mask = np.all(mask_np == color, axis=-1)
                new_mask[mask] = 1
        pred_mask = np.zeros(mask_np.shape[:2], dtype=np.uint8)
        for i in range(len(mask_lsit)):
            if i in obj_dict[f'{len(image_files)}']:
                masking_pred = (mask_list[-1] == i)
                pred_mask[masking_pred] = 1
        
        best_iou = iou_score(pred_mask, new_mask)
        print(f'Best IOU for {number_part}.png: {best_iou}')

        os.remove(dst_path)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='Directory containing images')
    parser.add_argument('--scene', '-s', type=str, required=True, help='Scene name')
    parser.add_argument('--n_view', '-n', type=int, required=True, help='number of training images')
    parser.add_argument('--px', type=int, required=True, help='x pixel coordinate of mask in start image')
    parser.add_argument('--py', type=int, required=True, help='y pixel coordinate of mask in start image')
    parser.add_argument('--thr', type=float, required=True, help='Threshold for pixel matching')
    
    args = parser.parse_args()

    data_dir = args.data_dir + '/' + args.scene
    video_dir = data_dir + f'/images/{args.n_view}_input/'
    px = args.px
    py = args.py
    threshold = args.thr
    main(data_dir, video_dir, args.n_view, px, py, threshold)

