#to run protos first install/verify if installed protobuf
    -conda install -c anaconda protobuf #to install
#then run the folloing command to convert .proto file to required python file as 'name_pb2.py'
>>protoc Project/protos/*.proto --python_out=.
