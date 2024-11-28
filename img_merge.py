import os
import shutil


src_folder = '/home/sd-finetune/data_generated/clean-SD-resized'
dst_folder = 'train_CIFAR/data/noisy_PMD70_onestep'

# get category labels
subfolders = next(os.walk(dst_folder))[1]

for subfolder in subfolders:
    src_subfolder = os.path.join(src_folder, subfolder)
    target_subfolder = os.path.join(dst_folder, subfolder)

    for filename in os.listdir(src_subfolder):
        src_file = os.path.join(src_subfolder, filename)
        dst_file = os.path.join(target_subfolder, filename)

        # if not os.path.exists(dst_file):
        shutil.copy(src_file, dst_file)
        # else:
        #     continue
