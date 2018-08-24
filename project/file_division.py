import os


counter = 0
tag = "abnormal"
training = "d/"

global_path = "../../../../../../Downloads/datasets/MITDATASET/training-" + training

tag_type = "RECORDS-" + tag

lines = [line.rstrip('\n')+ ".wav" for line in open(global_path + tag_type)]
print("Training: " + training)
print("Amount of tag lines for type: " + tag)
print(len(lines))
files_names = sorted(os.listdir(global_path))[1:]

for file_name in files_names:
    if file_name in lines:
        counter += 1
        os.rename(global_path + file_name, global_path + tag + "/" + file_name)
print("Amount of files moved for type: " + tag)
print(counter)
