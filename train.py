import mrcnn.model as modellib
import os
import numpy as np

from mrcnn.config import Config
from detect import ShapesConfig
from detect import ShapesDataset
from mrcnn.model import MaskRCNN


ROOT_DIR=os.path.abspath("../../")
MODEL_DIR=os.path.join(ROOT_DIR,"logs")
COOO_MODEL_PATH=os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")



#local path to trained weighted fiel
COCO_MODEL_PATH=os.path.join(ROOT_DIR,"mask_rcnn_coco.h5")


#create model in training mode

config=ShapesConfig()

model=modellib.MaskRCNN(mode="training",config=config,model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)





# Training dataset
dataset_train = ShapesDataset(Config)
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)