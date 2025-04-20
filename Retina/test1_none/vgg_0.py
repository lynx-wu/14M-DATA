import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
import glob
import random

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='val', sample_size=200):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sample_size = sample_size
        self.data_paths = []
        self.labels = []

        # 加载数据路径和标签
        self._load_dataset()

    def _load_dataset(self):
        classes = ['0', '1']  # 标签文件夹名称
        for label in classes:
            class_dir = os.path.join(self.root_dir, self.split, label)
            image_files = glob.glob(os.path.join(class_dir, '*.jpg'))
            # 随机抽取指定数量的图片
            sampled_files = random.sample(image_files, min(len(image_files), self.sample_size // 2))
            self.data_paths.extend(sampled_files)
            self.labels.extend([int(label)] * len(sampled_files))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def val_data_process():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    root_dir = '/media/disk/01drive/07qiulin/pj1/datasets2/dr'
    val_dataset = CustomDataset(root_dir=root_dir, transform=transform, split='val', sample_size=200)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    return val_loader
    
# 自定义模型
class CustomVGG(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomVGG, self).__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vgg.classifier[-1].in_features, 2)
        )

    def forward(self, x):
        return self.vgg(x)

def val_model_process(model, val_loader, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    val_accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    roc_aucs = []
    
    # 创建CSV文件以保存指标
    metrics_file = '/media/disk/01drive/07qiulin/pj1/indexs_0.csv'
    metrics_df = pd.DataFrame(columns=['Epoch', 'Train Accuracy', 'Val Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC AUC'])

    for epoch in range(num_epochs):
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())


            # 计算评估指标
            val_acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            roc_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr') if len(set(all_labels)) > 1 else 0.0

        # 记录指标
        val_accuracies.append(val_acc)
        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        roc_aucs.append(roc_auc)

        # 保存指标到CSV文件
        new_row = pd.DataFrame({
            'Epoch': [epoch + 1],
            'Val Accuracy': [val_acc],
            'F1 Score': [f1],
            'Recall': [recall],
            'Precision': [precision],
            'ROC AUC': [roc_auc]
        })

        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

        metrics_df.to_csv(metrics_file, index=False)

        # 第五轮次后输出图像
        if epoch >= 4:
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 2, 1)
            plt.plot(range(1, epoch + 2), val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics_0/_{epoch + 1}_Accuracy.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), f1_scores, label='F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('F1 Score over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics_0/_{epoch + 1}_F1_Score.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), recalls, label='Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('Recall over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics_0/_{epoch + 1}_Recall.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), precisions, label='Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.title('Precision over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics_0/_{epoch + 1}_Precision.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), roc_aucs, label='ROC AUC')
            plt.xlabel('Epoch')
            plt.ylabel('ROC AUC')
            plt.title('ROC AUC over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics_0/_{epoch + 1}_ROC_AUC.png')
            plt.close()

        # 保存模型
        model_save_path = f'/media/disk/01drive/07qiulin/pj1/models_0/_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_save_path)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Val Accuracy: {val_acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')


if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CustomVGG(dropout_rate=0.5).to(device)
    print('test')
    val_loader = val_data_process()
    print('test1')
    val_model_process(model, val_loader, 1)
    print('test2')