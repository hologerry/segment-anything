from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


sam = sam_model_registry["default"](checkpoint="pretrained_models/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate()
