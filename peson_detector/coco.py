## This script contains notes and python commands for custon training a pre-existion object detector using tensorflow (i.e. transfer learning).


# Folder Structure
Object-Detection
-data/
    --person_test.csv
    --person_train.csv
    --people_label_map.pbtxt
-images/
    --test/
        ---testingimages1.jpg
        ---testingimages2.jpg
        -- ...
    --train/
        ---trainingimages1.jpg
        ---trainingimages2.jpg
        -- ...
-training
-ssd_mobilenet_v1_coco_2018_01_28
    ---model.ckpt
    --...
-ssd_mobilenet_v1_coco_people.config


## Commands for creating a Tensorflow record for the validation (test) dataset. Assumes the 'CocoDatasetModForTrainAndVal.ipynb' script has been ran
python3 generate_tfrecord.py
--csv_input=data/person_test.csv
--output_path=data/test.record
--image_dir=images/test

## Commands for creating a Tensorflow record for the training dataset. Assumes the 'CocoDatasetModForTrainAndVal.ipynb' script has been ran
python3 generate_tfrecord.py
--csv_input=data/person_train.csv
--output_path=data/train.record
--image_dir=images/train

# Command for training the model. Assumes .config .pbtxt files have been setup accordingly
python3 train.py
--logtostderr
--train_dir=training/
--pipeline_config_path=ssd_mobilenet_v1_coco_people.config

# Command for exporting the frozen weights (.ckpt file) from the last training step
python3 export_inference_graph.py 
--input_type image_tensor 
--pipeline_config_path=ssd_mobilenet_v1_coco_people.config 
--trained_checkpoint_prefix training/model.ckpt-28607 
--output_directory person_inference_graph/


# required to convert dataset to tf record
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim





####################################################################################################################
## OLD command that didn't work for creating TensorFlow record for training and validation dataset
python3 third_party/models/research/object_detection/dataset_tools/create_coco_tf_record.py
    --train_image_dir='coco/people_train2017/' 
    --val_image_dir='coco/people_val2017/' 
    --test_image_dir='coco/test2017/' 
    --train_annotations_file='person_train.json' 
    --val_annotations_file='person_val.json' 
    --testdev_annotations_file='coco/annotations/image_info_test-dev2017.json' 
    --output_dir='coco/output/' 


