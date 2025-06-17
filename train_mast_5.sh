#!/bin/bash

scenes=("office_0" "office_1" "office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
views=(1 2 5)
thrsh=(0.0005 0.001 0.002 0.004 0.006 0.008)
pxs=(210 210 210 200 200 200 300 300 300 50 100 100 100 100 100 200 200 200 350 350 350 400 400 400)
pys=(350 350 350 200 200 200 460 460 460 400 460 460 460 460 460 400 400 400 220 220 220 460 350 350)

for i in {0..7}; do
    # 특정 scene 제외
    if [ "${scenes[i]}" = "office_1" ]; then
        continue
    fi

    for j in {0..2}; do
        # 특정 view 제외 (예: view 1 또는 2 제외)
        if [ "${views[j]}" = 1 ] || [ "${views[j]}" = 2 ]; then
            continue
        fi

        for t in "${thrsh[@]}"; do
            tmp=$((3 * i + j))

            echo "Scene: ${scenes[i]}, View: ${views[j]}, Thr: $t, px: ${pxs[tmp]}, py: ${pys[tmp]}"

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
