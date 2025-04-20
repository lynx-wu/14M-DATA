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

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, split='train', sample_size=1000):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.sample_size = sample_size

        self.data_frame = self.data_frame.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
        
        # 划分数据集
        self._split_dataset()

    def _split_dataset(self):
        total_length = len(self.data_frame)
        train_length = int(total_length * 0.8)
        
        if self.split == 'train':
            self.data_frame = self.data_frame.iloc[:train_length]
        elif self.split == 'val':
            self.data_frame = self.data_frame.iloc[train_length:]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        original_label = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('dr_grade')]
        label = 0 if original_label == 0 else 1

        if self.transform:
            image = self.transform(image)

        return image, label

def train_val_data_process():
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    csv_file = '/media/disk/01drive/07qiulin/pj1/SynFundus-1M_gen_cond_hunhe_0jpg.csv'
    root_dir = '/media/disk/01drive/06chengyang/14M/00'

    # 创建训练集数据加载器
    train_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, split='train')
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

    # 创建验证集数据加载器
    val_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform, split='val')
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    return train_loader, val_loader
    
# 自定义模型
class CustomResNet50(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomResNet50, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.resnet.fc.in_features, 2)
        )

    def forward(self, x):
        return self.resnet(x)

def train_val_model_process(model, train_loader, val_loader, num_epochs):
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    train_accuracies = []
    val_accuracies = []
    f1_scores = []
    recalls = []
    precisions = []
    roc_aucs = []
    
    # 创建CSV文件以保存指标
    metrics_file = '/media/disk/01drive/07qiulin/pj1/indexs.csv'
    metrics_df = pd.DataFrame(columns=['Epoch', 'Train Accuracy', 'Val Accuracy', 'F1 Score', 'Recall', 'Precision', 'ROC AUC'])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader, desc='Training', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train

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
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        roc_aucs.append(roc_auc)

        # 保存指标到CSV文件
        new_row = pd.DataFrame({
            'Epoch': [epoch + 1],
            'Train Accuracy': [train_acc],
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
            plt.plot(range(1, epoch + 2), train_accuracies, label='Train Accuracy')
            plt.plot(range(1, epoch + 2), val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics/_{epoch + 1}_Accuracy.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), f1_scores, label='F1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.title('F1 Score over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics/_{epoch + 1}_F1_Score.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), recalls, label='Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('Recall over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics/_{epoch + 1}_Recall.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), precisions, label='Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.title('Precision over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics/_{epoch + 1}_Precision.png')
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.plot(range(1, epoch + 2), roc_aucs, label='ROC AUC')
            plt.xlabel('Epoch')
            plt.ylabel('ROC AUC')
            plt.title('ROC AUC over Epochs')
            plt.legend()
            plt.savefig(f'/media/disk/01drive/07qiulin/pj1/graphics/_{epoch + 1}_ROC_AUC.png')
            plt.close()

        # 保存模型
        model_save_path = f'/media/disk/01drive/07qiulin/pj1/models/_{epoch + 1}.pth'
        torch.save(model.state_dict(), model_save_path)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {running_loss / len(train_loader.dataset):.4f}')
        print(f'Train Accuracy: {train_acc:.4f}')
        print(f'Val Accuracy: {val_acc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')


if __name__ == '__main__':
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CustomResNet50(dropout_rate=0.5).to(device)
    print('test')
    train_loader,val_loader = train_val_data_process()
    print('test1')
    train_val_model_process(model, train_loader, val_loader, 50)
    print('test2')