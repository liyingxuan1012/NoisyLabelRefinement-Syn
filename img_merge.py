import os
import shutil


src_folder1 = '/scratch/ace14550vm/ImageNet100_noisy/train'
src_folder2 = '/scratch/ace14550vm/SD-xl-turbo/train'
dst_folder = 'ImageNet100_noisy/train'
os.makedirs(dst_folder, exist_ok=True)

# get category labels
subfolders = next(os.walk(src_folder1))[1]

for subfolder in subfolders:
    target_subfolder = os.path.join(dst_folder, subfolder)
    os.makedirs(target_subfolder, exist_ok=True)

    for src_folder in [src_folder1, src_folder2]:
        src_subfolder = os.path.join(src_folder, subfolder)
        for filename in os.listdir(src_subfolder):
            src_file = os.path.join(src_subfolder, filename)
            dst_file = os.path.join(target_subfolder, filename)

            if not os.path.exists(dst_file):
                shutil.copy(src_file, dst_file)
            else:
                new_filename = filename.replace('.', '_duplicate.')
                dst_file = os.path.join(target_subfolder, new_filename)
                shutil.copy(src_file, dst_file)
