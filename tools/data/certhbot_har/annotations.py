import os
import argparse
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='Create annotations for custom dataset CERTHBOT_HAR')
    parser.add_argument('src_dir', type=str, help='source data directory')
    parser.add_argument('out_dir', type=str, help='output data directory')
    parser.add_argument(
        '--format',
        type=str,
        default='videos',
        choices=['videos', 'rawframes'],
        help='format of raw data (either videos or rawframes)')
    parser.add_argument(
        '--test_split_percentage',
        type=float,
        default=0.2,
        help='percentage of test set over the dataset')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    path = args.src_dir + "certhbot_har/" + args.format
    out_path = args.out_dir + "certhbot_har/"
    
    train_path = out_path + 'certhbot_har_train_split1_' + args.format + '.txt'
    val_path = out_path + 'certhbot_har_val_split1_' + args.format + '.txt'
    file_train = open(train_path, 'w+')
    file_val = open(val_path, 'w+')

    classes = dict()

    if args.format == 'videos':
        for class_id, action in enumerate(sorted(os.listdir(path))):
            classes[action] = class_id
            
            class_examples = []
            class_path = path + '/' + action
            for example in os.listdir(class_path):
                full_example_path = class_path + '/' + example
                class_examples.append(full_example_path)
            random.shuffle(class_examples)

            examples_size = len(class_examples)
            val_size = int(np.ceil(examples_size*args.test_split_percentage))

            val_list = class_examples[:val_size]
            train_list = class_examples[val_size:]

            for t in train_list:
                file_train.write(t + ' ' + str(class_id) + '\n')
            for t in val_list:
                file_val.write(t + ' ' + str(class_id) + '\n')
    else:
        for class_id, action in enumerate(sorted(os.listdir(path))):
            classes[action] = class_id
            
            class_examples = []
            class_path = path + '/' + action
            for example in os.listdir(class_path):
                full_example_path = class_path + '/' + example
                num_frames = len(os.listdir(full_example_path)) - 1
                class_examples.append([full_example_path, num_frames])
            random.shuffle(class_examples)

            examples_size = len(class_examples)
            val_size = int(np.ceil(examples_size*args.test_split_percentage))

            val_list = class_examples[:val_size]
            train_list = class_examples[val_size:]

            for t in train_list:
                file_train.write(t[0] + ' ' + str(t[1]) + ' ' + str(class_id) + '\n')
            for t in val_list:
                file_val.write(t[0] + ' ' + str(t[1]) + ' ' + str(class_id) + '\n')        

if __name__ == '__main__':
    main()