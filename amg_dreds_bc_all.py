# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os

from tqdm import tqdm


def find_max_iou_mask(mask_data_list, gt_mask):
    max_iou = 0
    max_iou_mask = None
    masks = [mask_data["segmentation"] * 255 for mask_data in mask_data_list]
    for mask in masks:
        iou = np.sum(np.logical_and(mask, gt_mask)) / np.sum(np.logical_or(mask, gt_mask))
        if iou > max_iou:
            max_iou = iou
            max_iou_mask = mask
    return max_iou_mask, max_iou


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)

    splits = ["train", "val", "test"]
    if args.debug:
        splits = ["val"]

    all_filenames_json = os.path.join(args.root_dir, args.bc_pairs_json)
    with open(all_filenames_json, "r") as f:
        data_dict = json.load(f)

    for split in splits:
        cur_split_all_filenames = data_dict[split]
        cur_job_pairs = cur_split_all_filenames[args.part_idx :: args.part_num]

        if args.sub_job_num > 0:
            cur_job_pairs = cur_job_pairs[args.sub_job_idx :: args.sub_job_num]

        if args.debug:
            cur_job_pairs = cur_job_pairs[:10]

        desc_str = f"Job {args.job_idx} part [{args.part_idx}/{args.part_num}] Processing {split}"
        if args.sub_job_num > 0:
            desc_str += f" sub job [{args.sub_job_idx}/{args.sub_job_num}]"

        for pair in tqdm(cur_job_pairs, desc=desc_str):
            bc_color_name = pair["bc_filename"]  # bc stands for blended controlnet output
            if args.rand:
                bc_color_name = bc_color_name.replace("seed12345", "seed-1")
            mask_name = pair["mask_filename"]

            bc_color_path = os.path.join(args.root_dir, bc_color_name)
            mask_path = os.path.join(args.root_dir, mask_name)
            out_mask_path = bc_color_path.replace("_bc_", "_sam_")
            out_iou_path = out_mask_path.replace(".png", ".txt").replace("_sam_", "_iou_")

            if os.path.exists(out_mask_path) and os.path.getsize(out_mask_path) > 0:
                continue

            dir_name = os.path.dirname(out_mask_path)
            os.makedirs(dir_name, exist_ok=True)
            iou_dir_name = os.path.dirname(out_iou_path)
            os.makedirs(iou_dir_name, exist_ok=True)

            image = cv2.imread(bc_color_path)
            if image is None:
                print(f"Part [{args.part_idx}/{args.part_num}] Could not load '{bc_color_path}' as an image, skipping...")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = generator.generate(image)
            gt_mask = cv2.imread(mask_path)
            gt_mask = gt_mask[:, :, 0]

            max_iou_mask, max_iou = find_max_iou_mask(masks, gt_mask)

            max_iou_mask = np.tile(max_iou_mask[:, :, None], (1, 1, 3))
            cv2.imwrite(out_mask_path, max_iou_mask)

            with open(out_iou_path, "w") as f:
                f.write(f"{max_iou:.6f}\n")
            # print(f"Saved mask to '{out_mask_path}' with iou = {max_iou}")

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="../data/DREDS/DREDS-CatKnown")
    parser.add_argument("--bc_pairs_json", type=str, default="dreds_bottle_bc_pairs.json")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rand", action="store_true")

    # ['default', 'vit_h', 'vit_l', 'vit_b']
    parser.add_argument("--model_type", type=str, default="vit_b")
    parser.add_argument("--checkpoint", type=str, default="pretrained_models/sam_vit_b_01ec64.pth")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    parser.add_argument("--convert-to-rle", action="store_true")

    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--job_num", type=int, default=1)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=8)
    parser.add_argument("--part_idx", type=int, default=0)
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--sub_job_idx", type=int, default=-1)
    parser.add_argument("--sub_job_num", type=int, default=-1)

    args = parser.parse_args()

    assert args.job_idx < args.job_num
    assert args.gpu_idx < args.gpu_num
    args.part_num = args.job_num * args.gpu_num
    args.part_idx = args.job_idx * args.gpu_num + args.gpu_idx

    main(args)
