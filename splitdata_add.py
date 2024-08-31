import os
import shutil
import random


dataset_path = r"E:\\Project\\STPS\\combined_folder"  
labels_path = r"E:\\Project\\STPS\\add_label"   


output_base = r"E:\Project\STPS\split_dataset_add"
train_image_dir = os.path.join(output_base, 'train', 'images')
val_image_dir = os.path.join(output_base, 'val', 'images')
test_image_dir = os.path.join(output_base, 'test', 'images')
train_label_dir = os.path.join(output_base, 'train', 'labels')
val_label_dir = os.path.join(output_base, 'val', 'labels')
test_label_dir = os.path.join(output_base, 'test', 'labels')


os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)


image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and not f.endswith('.npy')]
image_label_pairs = []
for image_file in image_files:
    label_file = image_file.replace('.png', '.txt').replace('.jpg', '.txt')
    if os.path.exists(os.path.join(labels_path, label_file)):
        image_label_pairs.append((image_file, label_file))


random.shuffle(image_label_pairs)


total_images = len(image_label_pairs)
train_split = int(total_images * 0.8)
val_split = int(total_images * 0.1)
test_split = total_images - train_split - val_split


train_pairs = image_label_pairs[:train_split]
val_pairs = image_label_pairs[train_split:train_split + val_split]
test_pairs = image_label_pairs[train_split + val_split:]

def move_files(pairs_list, src_image_dir, src_label_dir, dest_image_dir, dest_label_dir):
    for image_file, label_file in pairs_list:
        src_image = os.path.join(src_image_dir, image_file)
        dest_image = os.path.join(dest_image_dir, image_file)
        shutil.copy(src_image, dest_image)

        src_label = os.path.join(src_label_dir, label_file)
        dest_label = os.path.join(dest_label_dir, label_file)
        shutil.copy(src_label, dest_label)

move_files(train_pairs, dataset_path, labels_path, train_image_dir, train_label_dir)
move_files(val_pairs, dataset_path, labels_path, val_image_dir, val_label_dir)
move_files(test_pairs, dataset_path, labels_path, test_image_dir, test_label_dir)

print(f"The dataset split is complete, with a total of {total_images} pairs of images and their corresponding tags:")
print(f"train set: {len(train_pairs)} ")
print(f"val set: {len(val_pairs)} ")
print(f"test set: {len(test_pairs)} ")
