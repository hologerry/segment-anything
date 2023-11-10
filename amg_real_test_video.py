import numpy as np
import cv2

from segment_anything import SamPredictor, sam_model_registry

import argparse
import json
import os

from tqdm import tqdm


ori_w = 640
ori_h = 360


def main(args) -> None:
    print(f"Loading {args.model_type} model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    predictor = SamPredictor(sam)

    anno_json = f"1016_anno_box.json"

    print("predicting...")
    all_filenames_json = os.path.join(args.root_dir, anno_json)
    with open(all_filenames_json, "r") as f:
        data_dict = json.load(f)

    for item in tqdm(data_dict):
        image_path = os.path.join(args.root_dir, args.sub_dir, item["image"]+".png")

        image = cv2.imread(image_path)
        image = cv2.resize(image, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
        if image is None:
            print(f"Image {image_path} as an image, skipping...")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = item["label"]
        for label in labels:
            x = float(label["x"])
            y = float(label["y"])
            w = float(label["width"])
            h = float(label["height"])

            x = int(x / 100.0 * ori_w)
            y = int(y / 100.0 * ori_h)
            w = int(w / 100.0 * ori_w)
            h = int(h / 100.0 * ori_h)

            input_box = np.array([x, y, x + w, y + h])

            predictor.set_image(image)

            mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            # print(f"mask {mask.shape} min {mask.min()} max {mask.max()}")
            label_class = label["rectanglelabels"][0].strip()

            out_mask_path = image_path.replace(".png", f"_mask.png")
            mask = mask.astype(np.uint8)[0, ...] * 255
            mask = np.tile(mask[:, :, None], (1, 1, 3))
            cv2.imwrite(out_mask_path, mask)

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
