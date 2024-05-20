import matplotlib.pyplot as plt
import numpy as np


def parse_accuracy_file(file_path):
    class_ids = []
    acc_real = []
    acc_generated = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            class_id = parts[0].split(': ')[1]
            acc_real_value = float(parts[1].split(': ')[1])
            acc_generated_value = float(parts[2].split(': ')[1])
            
            class_ids.append(class_id)
            acc_real.append(acc_real_value)
            acc_generated.append(acc_generated_value)
    return class_ids, acc_real, acc_generated

def combine_and_save_data(class_ids, acc_real_clean, acc_generated_clean, acc_real_noisy, acc_generated_noisy, output_file_path):
    with open(output_file_path, 'w') as file:
        for i, class_id in enumerate(class_ids):
            line = f"Class: {class_id}, Acc_real_clean: {acc_real_clean[i]:.4f}, Acc_real_noisy: {acc_real_noisy[i]:.4f}, Acc_generated_clean: {acc_generated_clean[i]:.4f}, Acc_generated_noisy: {acc_generated_noisy[i]:.4f}\n"
            file.write(line)

def plot_accuracies_by_group(class_ids, acc_real_clean, acc_real_noisy, acc_generated_clean, acc_generated_noisy, start_index, end_index, group_number):
    ids = class_ids[start_index:end_index]
    real_clean = acc_real_clean[start_index:end_index]
    real_noisy = acc_real_noisy[start_index:end_index]
    generated_clean = acc_generated_clean[start_index:end_index]
    generated_noisy = acc_generated_noisy[start_index:end_index]
    
    num_classes = len(ids)
    width = 0.2
    index = np.arange(num_classes)
    
    plt.figure(figsize=(num_classes * 1.5, 6))
    plt.bar(index - 1.5*width, real_clean, width, color='#1f77b4', alpha=1.0, label='Real Images (Classifier1)')
    plt.bar(index - 0.5*width, real_noisy, width, color='#1f77b4', alpha=0.5, label='Real Images (Classifier2)')
    plt.bar(index + 0.5*width, generated_clean, width, color='#ff7f0e', alpha=1.0, label='Generated Images (Classifier1)')
    plt.bar(index + 1.5*width, generated_noisy, width, color='#ff7f0e', alpha=0.5, label='Generated Images (Classifier2)')

    plt.xlabel('Class ID')
    plt.ylabel('Accuracy')
    plt.title(f'Class Accuracies for Clean and Noisy Labels (Group {group_number})')
    plt.xticks(index, ids, rotation=0)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='large')
    plt.xlim(min(index) - 1, max(index + width) + 1)
    
    plt.tight_layout()
    plt.savefig(f'class_accuracies_group_{group_number}.png')
    plt.close()


# get class accuracies
class_ids_clean, acc_real_clean, acc_generated_clean = parse_accuracy_file('class_accuracies.txt')
class_ids_noisy, acc_real_noisy, acc_generated_noisy = parse_accuracy_file('class_accuracies_noisy.txt')
assert class_ids_clean == class_ids_noisy
class_ids = class_ids_clean

# combine and save data
combine_and_save_data(class_ids, acc_real_clean, acc_generated_clean, acc_real_noisy, acc_generated_noisy, 'class_accuracies_total.txt')

# plot accuracies by group
group_size = 25
num_groups = len(class_ids) // group_size
for i in range(num_groups):
    start_index = i * group_size
    end_index = start_index + group_size
    plot_accuracies_by_group(class_ids, acc_real_clean, acc_real_noisy, acc_generated_clean, acc_generated_noisy, start_index, end_index, i + 1)
