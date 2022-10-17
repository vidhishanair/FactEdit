import argparse
import json
from os import listdir

from os.path import isfile, join

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, default=None, help='Data path')
parser.add_argument('--split', type=str, default="train", help='Data Split')

args = parser.parse_args()

files_dir = join(args.data_dir, args.split)
wp = open(join(args.data_dir, args.split+'.json'), 'w')

onlyjsonfiles = [f for f in listdir(files_dir) if isfile(join(files_dir, f)) and '.json' in f]
error_type_dict = {}
count = 0
file_count = 0
for file in onlyjsonfiles:
    file_count += 1
    print(join(files_dir, file))
    fp = open(join(files_dir, file))
    for line in fp:
        count += 1
        wp.write(line.strip("\n")+"\n")
        data = json.loads(line)
        if 'error_type' in list(data.keys()):
            if data['error_type'] not in error_type_dict:
                error_type_dict[data['error_type']] = 0
            error_type_dict[data['error_type']] += 1
    fp.close()
wp.close()

print("Total file count: "+str(file_count))
if 'error_type' in list(data.keys()):
    print("Error Types: ")
    print(error_type_dict)
    print("Total count: "+str(count))
