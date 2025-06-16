#!/bin/bash

# 배열 정의 (공백 없이 =)
scenes=("office_0" "office_1" "office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
views=(1 2 5)
thrsh=(0.025 0.005 0.0075 0.01 0.0125 0.0015 0.0175)
pxs=(210 210 210 200 200 200 300 300 300 50 100 100 100 100 100 200 200 200 350 350 350 400 400 400)
pys=(350 350 350 200 200 200 460 460 460 400 460 460 460 460 460 400 400 400 220 220 220 460 350 350)

# scene index loop
for i in {0..7}; do
    # view index loop
    for j in {0..2}; do
        # threshold value loop
        for t in "${thrsh[@]}"; do
            tmp=$((3 * i + j))  # 인덱스 계산

            # 파라미터 출력 (디버그용)
            echo "Scene: ${scenes[i]}, View: ${views[j]}, Thr: $t, px: ${pxs[tmp]}, py: ${pys[tmp]}"

            # 실행
            python mast3r_mask.py \
                -d /content/drive/MyDrive/datasets/replica \
                -s "${scenes[i]}" \
                -n "${views[j]}" \
                --px "${pxs[tmp]}" \
                --py "${pys[tmp]}" \
                --thr "$t"
        done
    done
done
