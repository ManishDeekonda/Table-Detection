#tfrecord file creation
    -run create_tfrecord.py to get the train.record and val.record
    -NOTE:check the file locations in the create_tfrecord.py before running
    >>python Project/create_tfrecord.py --data_dir=my_dataset/ 
    --label_map_path=Project/train_file/table_label_map.pbtxt/ 
    --output_dir=new_tfrecord/

#convert poroto files to name_pb2.py files as follows
    >>protoc Project/protos/*.proto --python_out=.

#after creating the dataset and tfrecord file run the train.py as follows
#in tensorflow environment (activate tensorflow)
    >>python Project/train.py --logtostderr --train_dir=Project/training_table/
    --pipeline_config_path=Project/training_table/faster_rcnn_resnet.config/
    -NOTE: the current dir should be in 'LUSIP4/'

#after some seconds the training starts and wait for some iterations to occur 
test the model by creating the output_inference_graph folder for every 1000 iterations if the 
covergence is happening then stop the training. 

#you can also evaluate the model by using eval.py
    >>python Project/eval.py --logtostderr --checkpoint_dir=training_table/ 
    --eval_dir=Project/eval_table/ --pipeline_config_path=train_file/table_label_map.pbtxt/


