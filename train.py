"""

"train.py"

Training script.

"""

import os

import keras

from utils.parse import restore_cwd
from yolo.yolo import YOLO


# ---------------- SETUP ----------------
DATASET = "/media/ryan/Data/x-ray-datasets/sixray/images/"

DEFAULTS = {
    "model_path": "/home/ryan/models/sixray/x-net/models/first/final/trained_weights_stage_1.h5",
    "anchors_path": "/home/ryan/models/sixray/x-net/anchors/sixray_anchors.txt",
    "classes_path": "/media/ryan/Data/x-ray-datasets/sixray/classes.txt"
}

TRAIN_CONFIG = {
    "log_dir": "/home/ryan/models/sixray/x-net/models/third/stage_1/",
    "annotation_path": "/media/ryan/Data/x-ray-datasets/sixray/images/annotations.csv",
    "val_split": 0.1,
    "epochs": 50,
    "save_path": "train_weights_v1.h5"
}

FULL_TRAIN_CONFIG = {
    "optimizer": keras.optimizers.Adam(1e-4),
    "batch_size": 4
}

FINETUNE_CONFIG = {
    "optimizer": keras.optimizers.Adam(1e-5),
    "batch_size": 1
}

CALLBACKS = [
    keras.callbacks.TensorBoard(TRAIN_CONFIG["log_dir"], write_images=True),
    keras.callbacks.ModelCheckpoint(
        TRAIN_CONFIG["log_dir"] + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        save_weights_only=True, save_best_only=True, period=3),
    keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
    keras.callbacks.EarlyStopping(patience=10, verbose=1)
]


# ---------------- TRAINING ----------------
@restore_cwd
def train_xnet(dataset=None, defaults=None, train_config=None, callbacks=None, mode="finetune"):
    """Trains an X-Net/YOLO model

    :param dataset: dataset path (default: None)
    :param defaults: default dict (default: None)
    :param train_config: training config dict (default: None)
    :param callbacks: list of callbacks (default: None)
    :param mode: 'finetune' or 'full train' (default: finetune)
    :returns: trained x-net model

    """

    global DATASET, DEFAULTS, TRAIN_CONFIG, CALLBACKS

    # overwriting default configs with user-supplied configs
    DATASET = dataset if dataset is not None else DATASET
    DEFAULTS = {**DEFAULTS, **defaults} if defaults is not None else DEFAULTS
    if mode == "full train":
        TRAIN_CONFIG = {**TRAIN_CONFIG, **FULL_TRAIN_CONFIG}
    elif mode == "finetune":
        TRAIN_CONFIG = {**TRAIN_CONFIG, **FINETUNE_CONFIG}
    else:
        raise ValueError("supported modes are 'full train' and 'finetune'")
    TRAIN_CONFIG = {**TRAIN_CONFIG, **train_config} if train_config is not None else TRAIN_CONFIG
    CALLBACKS = {**CALLBACKS, **callbacks} if callbacks is not None else CALLBACKS

    # cd into dataset directory
    os.chdir(DATASET)

    # make and compile x-net
    xnet = YOLO(**DEFAULTS)

    xnet.prepare_for_training(freeze=YOLO.freeze, optimizer=TRAIN_CONFIG["optimizer"], mode=mode)

    # train x-net
    xnet.train(annotation_path=TRAIN_CONFIG["annotation_path"],
               save_path=TRAIN_CONFIG["log_dir"] + TRAIN_CONFIG["save_path"],
               epochs=TRAIN_CONFIG["epochs"],
               batch_size=TRAIN_CONFIG["batch_size"],
               val_split=TRAIN_CONFIG["val_split"],
               callbacks=CALLBACKS
    )

    return xnet


# ---------------- TESTING ----------------
if __name__ == "__main__":
    train_xnet(
        # defaults={
        #     "model_path": "/home/ryan/models/sixray/yolov3/yolo_weights_sixray.h5"
        # },
        train_config={
            "log_dir": "/home/ryan/models/sixray/x-net/models/first/stage_1_v2/",
            "batch_size": 4,
            "optimizer": keras.optimizers.Adam(learning_rate=1e-3)
        },
        mode="finetune"
    )
