# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os

from typing import Any, Dict, List


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--root_dir",
    type=str,
    default="../data/NOCS",
)

parser.add_argument(
    "--bc_pairs_json",
    type=str,
    default="nocs_bottle_bc_pairs_valid.json",
)

parser.add_argument(
    "--debug",
    action="store_true",
)

parser.add_argument(
    "--model-type",
    type=str,
    default="default",
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default="pretrained_models/sam_vit_h_4b8939.pth",
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=("Save masks as COCO RLEs in a single json instead of as a folder of PNGs. " "Requires pycocotools."),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def find_max_iou_mask(mask_data_list, gt_mask):
    max_iou = 0
    max_iou_mask = None
    masks = [mask_data["segmentation"] * 255 for mask_data in mask_data_list]
    for mask in masks:
        iou = np.sum(np.logical_and(mask, gt_mask)) / np.sum(np.logical_or(mask, gt_mask))
        if iou > max_iou:
            max_iou = iou
            max_iou_mask = mask
    return max_iou_mask


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    splits = ["train", "val", "real_train", "real_test"]
    if args.debug:
        splits = ["val"]

    all_filenames_json = os.path.join(args.root_dir, args.bc_pairs_json)
    with open(all_filenames_json, "r") as f:
        data_dict = json.load(f)

    for split in splits:
        cur_split_all_filenames = data_dict[split]
        for filenames_pair in cur_split_all_filenames:
            bc_color_name = filenames_pair["output_filename"]  # bc stands for blended controlnet output
            mask_name = filenames_pair["mask_filename"]
            # depth_name = filenames_pair["depth_filename"]
            bc_color_path = os.path.join(args.root_dir, bc_color_name)
            mask_path = os.path.join(args.root_dir, mask_name)

            out_mask_path = mask_path.replace("mask", "sam_mask")
            image = cv2.imread(bc_color_path)
            if image is None:
                print(f"Could not load '{bc_color_path}' as an image, skipping...")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = generator.generate(image)
            gt_mask = cv2.imread(mask_path)

            max_iou_mask = find_max_iou_mask(masks, gt_mask)
            cv2.imwrite(out_mask_path, max_iou_mask)

    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
