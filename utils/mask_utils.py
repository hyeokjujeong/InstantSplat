import re
from pathlib import Path
from typing import List, Tuple
from mast3r.fast_nn import fast_reciprocal_NNs
import torch
import os

def get_sorted_image_files_rgb(image_dir: str) -> Tuple[List[str], str]:
    """
    Get sorted RGB image files from the given directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        Tuple[List[str], str]: A tuple containing:
            - List of sorted RGB image file paths
            - The file suffix used (e.g., '.png')
    """
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_path = Path(image_dir)

    def extract_number(filename: Path) -> int:
        match = re.search(r'\d+', filename.stem)
        return int(match.group()) if match else float('inf')

    image_files = [
        f for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_extensions and f.name.startswith("train_")
    ]

    sorted_files = sorted(image_files, key=extract_number)
    sorted_file_paths = [str(f) for f in sorted_files]
    suffixes = [f.suffix for f in sorted_files]

    return sorted_file_paths, suffixes[0] if suffixes else ''


def get_corresponding_mask_paths(rgb_paths):
    """
    Args:
        rgb_paths (List[str]): RGB 이미지 경로 리스트 (예: scene_name/images/image.jpg)
    
    Returns:
        List[str]: SAM mask 경로 리스트 (예: scene_name/raw_sam_mask/image.png)
    """
    mask_paths = []
    for rgb_path in rgb_paths:
        # 'images' → 'raw_sam_mask' 로 폴더 이름 교체
        mask_path = rgb_path.replace('/images/', '/raw_sam_mask/')
        
        # 확장자 변경 (.jpg or .jpeg → .png)
        base, ext = os.path.splitext(mask_path)
        mask_path = base + '.png'
        
        mask_paths.append(mask_path)
    
    return mask_paths


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

    resized_mask0 = (255*mask0).squeeze(0).squeeze(0).long().T
    resized_mask1 = (255*mask1).squeeze(0).squeeze(0).long().T
    mask0_size = torch.zeros((torch.max(resized_mask0)+1)).long()
    for i in range(torch.max(resized_mask0)+1):
        mask0_size[i] = (resized_mask0==i).sum().item()
    mask1_size = torch.zeros((torch.max(resized_mask1)+1)).long()
    for i in range(torch.max(resized_mask1)+1):
        mask1_size[i] = (resized_mask1==i).sum().item()
    correspondances = torch.zeros((torch.max(resized_mask0)+1, torch.max(resized_mask1)+1))
    corr_tf = -torch.ones((torch.max(resized_mask0)+1, torch.max(resized_mask1)+1))
    xs0, ys0 = matches_im0[:,0], matches_im0[:,1]
    im0_mask_idx = resized_mask0[xs0, ys0]
    xs1, ys1 = matches_im1[:,0], matches_im1[:,1]
    im1_mask_idx = resized_mask1[xs1, ys1]
    for i in range(len(im0_mask_idx)):
        correspondances[im0_mask_idx[i], im1_mask_idx[i]]+=1
        #corr_tf[im0_mask_idx[i], im1_mask_idx[i]] = 1
    for i in range(torch.max(resized_mask0)+1):
        for j in range(torch.max(resized_mask1)+1):
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
    for i in range(torch.max(resized_mask0)+1):
        marker = False
        for temp in range(temp_corr):
            if temp[0] == i:
                marker = True
                break
        if marker == False:
            temp_corr.append([i, -1])
    
    for i in range(torch.max(resized_mask1)+1):
        marker = False
        for temp in range(temp_corr):
            if temp[1] == i:
                marker = True
                break
        if marker == False:
            temp_corr.append([-1, i])
    print(f'Mask Correspondances: {temp_corr}')
    return temp_corr

def update_obj_list(obj_list, temp_corr, n):
    import copy

    np1 = n + 1
    new_entries = []

    # 초기 프레임 처리
    if n == 0 and not obj_list:
        obj_list_0 = []
        init_max = 0
        for temp in temp_corr:
            if temp[0] > init_max:
                init_max = temp[0]
        for i in range(init_max + 1):
            temp_dict = {}
            temp_dict[str(n)] = [i]
            obj_list_0.append(temp_dict)
        obj_list = obj_list_0
    elif n == 0 or not obj_list:
        raise AssertionError("Images except for idx 0 should have initialized object list")

    # correspondence에 따라 obj_list 업데이트
    for m_n, m_np1 in temp_corr:
        if m_n != -1:
            if m_np1 == -1:
                for obj in obj_list:
                    if str(n) in obj and m_n in obj[str(n)]:
                        obj[str(np1)] = []
                        break
                continue
            else:
                for obj in obj_list:
                    if str(n) in obj and m_n in obj[str(n)]:
                        if str(np1) not in obj:
                            obj[str(np1)] = []
                        if m_np1 not in obj[str(np1)]:
                            obj[str(np1)].append(m_np1)
                        break
        else:
            temp_dict = {str(i): [] for i in range(np1)}
            temp_dict[str(np1)] = [m_np1]
            new_entries.append(temp_dict)

    obj_list.extend(new_entries)

    # 병합 루프 (중복 제거 및 병합)
    merged = True
    while merged:
        print('loop')
        merged = False
        new_obj_list = []
        used = [False] * len(obj_list)

        for i in range(len(obj_list)):
            if used[i]:
                continue
            base = copy.deepcopy(obj_list[i])
            used[i] = True

            for j in range(i + 1, len(obj_list)):
                if used[j]:
                    continue
                other = obj_list[j]
                overlap = False

                # 병합 기준: np1에서의 object id가 겹치는 경우
                for img_key in [str(np1)]:
                    if img_key in base and img_key in other:
                        if set(base[img_key]) & set(other[img_key]):
                            overlap = True
                            break

                if overlap:
                    changed = False
                    for key in other:
                        if key in base:
                            before = set(base[key])
                            base[key].extend(other[key])
                            base[key] = list(set(base[key]))
                            if set(base[key]) != before:
                                changed = True
                        else:
                            base[key] = other[key][:]
                            changed = True

                    if changed:
                        used[j] = True
                        merged = True

            new_obj_list.append(base)

        obj_list = new_obj_list

    return obj_list


def get_object_masks(masks_list, fmoutput, fmodel, pairs, device, threshold=0.01):
  
    i=0
    obj_list = []
    while i!= len(masks_list)-1:
        print('get_obj_masks')
        for j in range(len(pairs)):
            if pairs[j][0]["idx"]==i and pairs[j][1]["idx"]==i+1:
            #mask0_pth , mask1_pth = mask_pths[i], mask_pths[i+1]
                matches_im0, matches_im1 = get_valid_matches(fmoutput, fmodel, i, device)
                temp_corr = get_correspondance_mat(masks_list[i], masks_list[i+1], matches_im0, matches_im1, threshold)
                print(temp_corr)
                obj_list = update_obj_list(obj_list, temp_corr, i)
                i+=1
                break
    return obj_list