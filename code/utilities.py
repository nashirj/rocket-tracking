import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Set memory growth to prevent OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print("Successfully set memory growth")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return image.reshape((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    

def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None,
                    min_score_thresh=0.8):
    """Wrapper function to visualize detections.

    Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=min_score_thresh)
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)


def convert_yolo_to_gt(yolo_paths):
    """Converts annotations output by LabelStudio in yolo format to the ground
    truth format expected in the Tensorflow object detection API notebooks for
    model fine tuning."""
    gt_boxes=[]
    for path in yolo_paths:
        row=[]
        bbox_temp=[]
        with open(path, 'rt') as fd:
            for line in fd.readlines():
                splitted = line.split()
                try:
                    xc = float(splitted[1])
                    yc = float(splitted[2])
                    w = float(splitted[3])
                    h = float(splitted[4])
                    x1 = xc - (w / 2)
                    x2 = xc + (w / 2)
                    y1 = yc - (h / 2)
                    y2 = yc + (h / 2)
                    gt_boxes.append(np.array([[y1, x1, y2, x2]], dtype=np.float32))
                except Exception as e:
                    print(e)
                    print("file is not in YOLO format!")
    return gt_boxes


def create_model_and_restore_weights_for_train(
    num_classes, pipeline_config, checkpoint_path):
    """Loads a pretrained model from a specified pipeline config, restores the
    weights from checkpoint path, and sets the num classes in the classification
    head to num_classes. Assumes the use of RetinaNet model architecture."""
    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)
    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be
    # just one (for our new rocket class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
          model_config=model_config, is_training=True)

    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
        )
    fake_model = tf.compat.v2.train.Checkpoint(
              _feature_extractor=detection_model._feature_extractor,
              _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')
    return detection_model
