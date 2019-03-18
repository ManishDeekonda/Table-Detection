import os
import numpy as np

def generate_txt_files(path_to_toy_data):
    files = os.listdir(path_to_toy_data)
    #train = open("train.txt", "w")
    train_val = open("trainval.txt", "w")
    #val = open("val.txt", "w")
    size_train_val_set = int(len(files))
    train_val_files = np.random.choice(files, size=size_train_val_set, replace=False)
    for f in train_val_files:
        train_val.write(f.replace(".png", "") + " " + str(1) + "\n")
        files.remove(f)
    train_val.close()
generate_txt_files("my_dataset/raw")

