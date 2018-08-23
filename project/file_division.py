import os

global_path = "../../../../../../Downloads/datasets/MITDATASET/training-b/RECORDS-normal"

f = open(global_path, 'r')
print(f.read())
lista = [name for name in f]

lines = [line.rstrip('\n') for line in open(global_path)]

print(lines)
