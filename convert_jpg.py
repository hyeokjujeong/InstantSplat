import os
from PIL import Image

# 대상 디렉토리 경로
dir_path = '/content/drive/MyDrive/datasets/replica'  # 예: './images'
scene_list = ["office_0", "office_1", "office_2", "office_3", "office_4", "room_0", "room_1", "room_2"]
view_list = [1, 2, 5]
folder_list = ['train', 'test']

# 디렉토리 내 파일 순회

for scene in scene_list:
    for view in view_list:
        for fold in folder_list:
            temp_dir = dir_path+f'/{scene}/images/'+f'{view}_input/'+f'{fold}'
            for filename in os.listdir(temp_dir):
                if filename.endswith('.png') and filename.startswith('train_rgb_'):
                    png_path = os.path.join(temp_dir, filename)
                    base_name = filename.replace('train_rgb_', '').replace('.png', '')
                    jpg_filename = f'{base_name}.jpg'
                    jpg_path = os.path.join(temp_dir, jpg_filename)

                    with Image.open(png_path) as img:
                        rgb_img = img.convert('RGB')  # PNG에는 알파 채널이 있을 수 있음
                        rgb_img.save(jpg_path, 'JPEG')

                    os.remove(png_path)

