import mrcnn.model as modellib
import os
import numpy as np
import random
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.config import Config
from detect import ShapesConfig
from train import dataset_val
from train import dataset_train
from detect import ShapesDataset

config=ShapesConfig(Config)

class InferenceConfig(ShapesConfig):

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1



inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
#ROOT_DIR='C:\Users\DELL\PycharmProjects\MaskRCNN'
#model_path="C:/Users/DELL/PycharmProjects/MaskRCNN/mask_rcnn_coco.h5"
#model_path = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))
