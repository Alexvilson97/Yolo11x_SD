import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

# Set paths
pipeline_config = 'path/to/your/pipeline.config'  # Update with your pipeline config path
checkpoint_path = 'path/to/your/checkpoint'  # Update with your checkpoint path
label_map_path = 'path/to/label_map.pbtxt'  # Update with your label map path

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Load pipeline config and build model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(checkpoint_path).expect_partial()

@tf.function
def detect_objects(image_tensor):
    """Performs detection on an input image tensor."""
    preprocessed_image, shapes = detection_model.preprocess(image_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Load and preprocess images
def load_image_into_numpy_array(path):
    image = Image.open(path)
    (width, height) = image.size
    return np.array(image).reshape((height, width, 3)).astype(np.uint8)

image_paths = ['Images\City_sunny\RD\frame_540.png', 'Images\City_sunny\SD\Frame_540.png']  # Update with your image paths
images = [load_image_into_numpy_array(img_path) for img_path in image_paths]

for i, image_np in enumerate(images):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_objects(input_tensor)

    # Extract detection data
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Convert classes to integers
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Visualize results
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False
    )

    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.title(f"Detection Results for Image {i + 1}")
    plt.axis('off')
    plt.show()
