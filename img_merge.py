import os
import shutil


src_folder = '/scratch/ace14550vm/SD-xl-turbo/train'
dst_folder = 'ImageNet100_noisy_filtered/train'

# get category labels
subfolders = next(os.walk(dst_folder))[1]

for subfolder in subfolders:
    target_subfolder = os.path.join(dst_folder, subfolder)

    src_subfolder = os.path.join(src_folder, subfolder)
    if os.path.exists(src_subfolder):
        for filename in os.listdir(src_subfolder):
            src_file = os.path.join(src_subfolder, filename)
            dst_file = os.path.join(target_subfolder, filename)

            if not os.path.exists(dst_file):
                shutil.copy(src_file, dst_file)
            else:
                new_filename = filename.replace('.', '_duplicate.')
                dst_file = os.path.join(target_subfolder, new_filename)
                shutil.copy(src_file, dst_file)
