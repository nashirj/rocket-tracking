import os

import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder


def load_model(model_dir, pipeline_config, ckpt_name):
    """Loads model as indiciated by pipeline_config and specified checkpoint."""
    pipeline_config_path = f"../tf_code/models/research/object_detection/configs/tf2/{pipeline_config}"

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, ckpt_name)).expect_partial()

    return detection_model


# Uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(detection_model, input_tensor):
    """Run detection on an input image.

    Args:
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.

    Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
            and `detection_scores`).
    """
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    return detection_model.postprocess(prediction_dict, shapes)

