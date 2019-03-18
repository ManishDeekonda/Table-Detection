#First to create Dataset use LabelImg-master folder
    -run python file as 'python3 labelImg.py' in the terminal.
    -you will get a window where you can draw the boxes and save respected XML files.

#change the terminal dir to LUSIP4 location then run the files

#to get the output_inference_graph folder
    -run the export_inference_graph.py file with some requied attributes after training
    -python Project/export_inference_graph.py \
        --input_type image_tensor \
        --pipeline_config_path Project/training_table/faster_rcnn_resnet.config \
        --trained_checkpoint_prefix Project/training_table/model.ckpt-STEP_NUMBER \
        --output_directory output_inference_graph
    -note:STEP_NUMBER = step number where the training stopped
    
#now change/recheck the required path fields in the table_detection_runner.py

    -PATH_TO_LABELS
    -PATH_TO_TEST_IMAGES_DIR
    -MODEL_NAME      ('path to output_inference_graph')
    -PATH_TO_CKPT    ('path to frozen_inference_graph.pb' which is in above folder)
 
 
#now run the table_detection_runner.py after saving some images in test_images in my_dataset/test_images
youll get an image with the table/s detected (also saves in output folder)

