import tensorflow.contrib.tensorrt as trt
from tf_trt_models.detection import download_detection_model, build_detection_graph

frozen_graph, input_names, output_names = build_detection_graph(
    config_path='training/pipeline.config',
    checkpoint='training/model.ckpt-28607'
    #score_threshold=0.3,
    #batch_size=1
)


trt_graph_def = trt.create_inference_graph(
    #input_graph_def='person_inference_graph/person_inference_graph.pb',
    input_graph_def=frozen_graph,
    outputs = output_names,
    #outputs=['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections'],
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

pb_path = 'ssd_mobilenet_v1_coco_people/{}_trt.pb'.format('ssd_mobilenet_v1_coco_people')
with open(pb_path, 'wb') as pf:
        pf.write(trt_graph_def.SerializeToString())