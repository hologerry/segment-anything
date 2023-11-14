import numpy as np
import cv2

from segment_anything import SamPredictor, sam_model_registry

import argparse
import json
import os

from tqdm import tqdm


ori_w = 256
ori_h = 256


def main(args) -> None:
    print(f"Loading {args.model_type} model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    predictor = SamPredictor(sam)

    results_folder = "/home/yuegao/Bot/free_bot_evaluation_raft_ir_results/eval_raft_mvs_2layers_bot_render_x_ir_depth_crop_more_bs6_bg01_scratch_63000_real_test_water_video_ir_depth_2layers_crop"
    scene_names = sorted(os.listdir(results_folder))

    for scene_name in tqdm(scene_names):
        scene_path = os.path.join(results_folder, scene_name)
        file_names = os.listdir(scene_path)
        file_names = [f for f in file_names if "vis.png" in f]
        file_names = sorted(file_names)
        for file_name in file_names:
            image_path = os.path.join(results_folder, scene_name, file_name)
            image = cv2.imread(image_path)
            image = image[:, 768:, :]


            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            predictor.set_image(image)

            input_box = np.array([0, 0, 0 + 256, 0 + 256])
            mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True,
            )
            num_masks = mask.shape[0]
            for i in range(num_masks):
                out_mask_path = image_path.replace(".png", f"_diffmask_{i}.png")
                mask_i = mask[i, ...].astype(np.uint8) * 255
                mask_i = np.tile(mask_i[:, :, None], (1, 1, 3))
                cv2.imwrite(out_mask_path, mask_i)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="../data/RealTestWaterVideo")
    parser.add_argument("--sub_dir", type=str, default="captured_1016_end")
    parser.add_argument("--debug", action="store_true")

    # ['default', 'vit_h', 'vit_l', 'vit_b']
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--checkpoint", type=str, default="pretrained_models/sam_vit_h_4b8939.pth")

    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    args = parser.parse_args()

    main(args)
