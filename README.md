# How to run (CIFAR-100)
- Install required packages
  ```
  pip install -r requirements.txt
  ```
- Download datasets and put them at `train_CIFAR/data`
- Train a classifier using noisy data
  ```
  cd  train_CIFAR
  CUDA_VISIBLE_DEVICES=0 python resnet_train.py --train_dir data/CIFAR100_noisy/noisy_PMD70 --model_dir models_pretrained/cifar100_PMD70.pt --log_dir logs/cifar100_PMD70.log
  ```
- Refine noisy labels using generated images and fine-tune the classifier
  ```
  CUDA_VISIBLE_DEVICES=0 python resnet_finetune.py --dataset_dir data/CIFAR100_noisy/noisy_PMD70 --train_dir data/noisy_PMD70_onestep --pretrained_model_dir models_pretrained/cifar100_PMD70.pt --best_model_dir models/cifar100_PMD70_onestep.pt --log_dir logs/cifar100_PMD70_onestep.log
  ```

# Codes
关于代码的说明

- `train_CIFAR/img_filter_sim.py`：使用生成图片进行降噪的模块
  - `filter_images_iter0`：在每一个类别里，根据和生成图片的相似度直接删除指定数量的图片
  - `random_discard_images`：在每一个类别里，随机删除指定数量的图片
  - `relabel_and_copy_images`：不删除图片，而是使用图片的新标签
- `train_CIFAR/img_filter_clip.py`：与`train_CIFAR/img_filter_sim.py`算法相同，只是使用CLIP作为特征提取器
- `train_CIFAR/resnet_finetune.py`：主要代码
  - 根据需要选择使用`train_CIFAR/img_filter_sim.py`里的哪一个方法来进行降噪
  - 在命令里添加`--add_generated`时会给降噪后的数据集加入对应类别的生成图片
  - 降噪完成后计算噪声率并使用降噪后的数据来微调分类器
