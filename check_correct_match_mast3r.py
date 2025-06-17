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


def extract_instance_mask(mask, label, x_size, y_size):
    #mask = cv2.imread(mask_path)#, cv2.IMREAD_GRAYSCALE
    
    #if mask is None:
    #    raise ValueError(f"Failed to read mask image from {mask_path}")
    target_size = (x_size, y_size)
    
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (resized_mask == label).astype(np.uint8)




def get_gt_masks(frame_list, px, py):
    list_mask = []
    cnt=0
    for fra in frame_list:
        #gt_inst_path = os.path.join(gt_mask_dir, fra)
        tmp_mask = Image.open(fra).convert('RGB')
        tmp_mask_np = np.array(tmp_mask)
        unique_colors = np.unique(tmp_mask_np.reshape(-1, 3), axis=0)
        if cnt==0:
            gt_color = tmp_mask_np[py, px]
        gt_mask_binary = np.all(tmp_mask_np==gt_color, axis=-1)
        list_mask.append(gt_mask_binary)

        cnt+=1
    return list_mask

def get_accuracy(mask0, mask1, matches_im0, matches_im1):
    x0 = matches_im0[:,0]
    y0 = matches_im0[:,1]
    x1 = matches_im1[:,0]
    y1 = matches_im1[:,1]

    total_0 = mask0[y0, x0]
    total_1 = mask1[y1, x1]
    total_both = total_0 & total_1

    accuracy = total_both.sum().item()/total_0.sum().item()



def main(data_dir, video_dir,n_view, px, py):
    test_dir = video_dir + '/test'
    final_video_dir = video_dir+'/train'
    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    #sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    #model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AsymmetricMASt3R.from_pretrained('./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth').to(device)


    
    image_size  = 640
    count_idx = 0
    avg_acc=0
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
        frame_names_revised= []
        for frame_name in frame_names_sorted:
            basename = os.path.basename(frame_name)       # '0000.jpg'
            name_only = os.path.splitext(basename)[0]     # '0000'
            temp_frame = int(name_only) 
            if temp_frame>1000:
                temp_frame -=1000
            revised = f'{temp_frame:04d}'
            frame_names_revised.append(revised)
        mask_names = [mask_dir+'/train_rgb_'+p+'.png' for p in frame_names_revised]
        gt_names = [gt_dir+'/train_vis_sem_instance_'+p+'.png' for p in frame_names_revised]

        # when geometry init, only use train images
        #image_files = train_img_files
        images, org_imgs_shape = load_images(image_files, size=image_size)

        start_time = time()
        print(f'>> Making pairs...')
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        print(f'>> Inference...')
        output = inference(pairs, model, device, batch_size=1, verbose=True)

       
        mask_list = get_gt_masks(gt_names, px, py)
        
        acc_list = []
        i=0
        while i!= len(image_files)-1:
            for j in range(len(pairs)):
                if pairs[j][0]["idx"]==i and pairs[j][1]["idx"]==i+1:
                    matches_im0, matches_im1 = get_valid_matches(output, model, j, device)
                    acc = get_accuracy(mask_list[i], mask_list[i+1], matches_im0, matches_im1)
                    acc_list.append(acc)
                    break
            
            i+=1

        instance_acc = np.mean(acc_list).item()
        
        
        
        txt_path = data_dir+'/acc_results.txt'
        with open(txt_path, 'a') as f:
            f.write(f"{n_view} instance: {instance_acc}\n")
        avg_acc+=instance_acc


        os.remove(dst_path)
    
    with open(txt_path, 'a') as f:
        f.write(f"{n_view}_input average: {avg_acc/(count_idx)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and save results.')
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='Directory containing images')
    parser.add_argument('--scene', '-s', type=str, required=True, help='Scene name')
    parser.add_argument('--n_view', '-n', type=int, required=True, help='number of training images')
    parser.add_argument('--px', type=int, required=True, help='x pixel coordinate of mask in start image')
    parser.add_argument('--py', type=int, required=True, help='y pixel coordinate of mask in start image')
    #parser.add_argument('--thr', type=float, required=True, help='Threshold for pixel matching')
    
    args = parser.parse_args()

    data_dir = args.data_dir + '/' + args.scene
    video_dir = data_dir + f'/images/{args.n_view}_input/'
    px = args.px
    py = args.py
    #threshold = args.thr
    main(data_dir, video_dir, args.n_view, px, py)

