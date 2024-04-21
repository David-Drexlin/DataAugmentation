import pandas as pd
import os
import h5py
from PIL import Image
import numpy as np
import random

def is_tumor(annotation_classes):
    annotations = eval(annotation_classes)
    return not (annotations.get("1") == 0.0 and annotations.get("2") == 0.0)

def process_directories(subdirs, data_file, label_file, csv_filename, root_dir):
    global_index = 0
    is_first = True
    with h5py.File(data_file, 'w') as data_h5file, h5py.File(label_file, 'w') as label_h5file:
        for subdir in subdirs:
            full_subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(full_subdir_path) and not subdir in ['metadata', 'thumbnails']:
                data_group = data_h5file.create_group(subdir)
                for img_name in os.listdir(full_subdir_path):
                    img_path = os.path.join(full_subdir_path, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                        image = Image.open(img_path).convert('RGB')
                        image = np.array(image)
                        data_group.create_dataset(img_name, data=image)
            
                metadata_path = os.path.join(root_dir, subdir, 'metadata', 'df.csv')
                if os.path.exists(metadata_path):
                    df = pd.read_csv(metadata_path)
                    df['local_index'] = df.index
                    df['global_index'] = [f"{subdir}_{idx}" for idx in range(global_index, global_index + len(df))]
                    global_index += len(df)
                    with open(csv_filename, mode='a') as file:
                        df.to_csv(file, index=False, header=is_first)
                    is_first = False
                    label_group = label_h5file.create_group(subdir)
                    for index, row in df.iterrows():
                        image_name = row['path']
                        annotation_classes = row['annotation_classes']
                        label = is_tumor(annotation_classes)
                        label_group.create_dataset(image_name, data=label)

def main():
    root_dir = '/home/space/datasets/camelyon16/patches/20x'
    #print(os.listdir('/home/daviddrexlin/Master'))
    #print(os.listdir('/home'))
    #print(os.listdir('home/space'))
    #print(os.listdir('/home/space/datasets/camelyon16'))
    #print(os.listdir(root_dir))
    #print(1/0)

    train_data_file = './data/train_images.h5'
    train_label_h5_file = './data/train_labels.h5'
    valid_data_file = './data/valid_images.h5'
    valid_label_h5_file = './data/valid_labels.h5'
    test_data_file = './data/test_images.h5'
    test_label_h5_file = './data/test_labels.h5'
    train_csv_filename = './data/master_train_meta_data.csv'
    valid_csv_filename = './data/master_valid_meta_data.csv'
    test_csv_filename = './data/master_test_meta_data.csv'

    normal_dirs = [d for d in sorted(os.listdir(root_dir)) if d.startswith('normal')]
    tumor_dirs = [d for d in sorted(os.listdir(root_dir)) if d.startswith('tumor')]
    test_dirs = [d for d in sorted(os.listdir(root_dir)) if d.startswith('test')]
    random.shuffle(normal_dirs)
    random.shuffle(tumor_dirs)

    # Split subdirectories into train, validation, and test sets
    normal_split = int(0.1 * len(normal_dirs))
    tumor_split = int(0.1 * len(tumor_dirs))
    train_subdirs = normal_dirs[normal_split:] + tumor_dirs[tumor_split:]
    valid_subdirs = normal_dirs[:normal_split] + tumor_dirs[:tumor_split]

    # process_directories(test_dirs, test_data_file, test_label_h5_file, test_csv_filename, root_dir)
    process_directories(train_subdirs, train_data_file, train_label_h5_file, train_csv_filename, root_dir)
    process_directories(valid_subdirs, valid_data_file, valid_label_h5_file, valid_csv_filename, root_dir)
    #process_directories(test_dirs, test_data_file, test_label_h5_file, test_csv_filename, root_dir)

if __name__ == "__main__":
    main()
