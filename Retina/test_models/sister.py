import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

n_classes = 2  # 二分类
epoches = 10

# 记录各个指标的列表,同于画图
train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
train_roc_aucs, val_roc_aucs = [], []
train_f1s, val_f1s = [], []


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx]['idx'])  # 获取图片路径
        image = Image.open(img_name).convert('RGB')
        # label = int(self.data_frame.iloc[idx]['label'])
        label = self.data_frame.iloc[idx, self.data_frame.columns.get_loc('dr_grade')]

        if self.transform:
            image = self.transform(image)

        return image, label

# 加载数据集
csv_file = '/media/disk/01drive/07qiulin/pj1/SynFundus-1M_gen_cond_hunhe_0jpg.csv'
root_dir = '/media/disk/01drive/06chengyang/14M/00'

# 图像的变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class ResNet18WithDropout(nn.Module):
    def __init__(self, n_classes=2, dropout_prob=0.5):
        super(ResNet18WithDropout, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)  # 使用 ResNet18
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(in_features, n_classes)
        )

    def forward(self, x):
        return self.resnet(x)


model = ResNet18WithDropout(n_classes=n_classes, dropout_prob=0.5).to(device)


def save_metrics(epoch, train_metrics, val_metrics, save_path):
    with open(save_path, "a") as f:
        f.write(f"vgg--Epoch {epoch + 1}\n")
        f.write(
            f"Train Loss: {train_metrics[0]:.5f}, Acc: {train_metrics[1]:.2f}%, Precision: {train_metrics[2]:.4f}, Recall: {train_metrics[3]:.4f}, F1 Score: {train_metrics[4]:.4f}, ROC AUC: {train_metrics[5]:.4f}\n")
        f.write(
            f"Val Loss: {val_metrics[0]:.5f}, Acc: {val_metrics[1]:.2f}%, Precision: {val_metrics[2]:.4f}, Recall: {val_metrics[3]:.4f}, F1 Score: {val_metrics[4]:.4f}, ROC AUC: {val_metrics[5]:.4f}\n")
        f.write("--------------\n")


# 指标
def calculate_metrics(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    roc_auc = roc_auc_score(labels, probs[:, 1])  # 针对二分类问题
    return acc, precision, recall, f1, roc_auc


# 训练函数
def train_model(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    all_train_labels = []
    all_train_preds = []
    all_train_probs = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        probs = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()

        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(preds.cpu().numpy())
        all_train_probs.extend(probs)

        total_corrects += torch.sum(preds.eq(labels))
        total_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

    total_loss = total_loss / total
    acc = 100 * total_corrects / total

    train_acc, train_precision, train_recall, train_f1, train_roc_auc = calculate_metrics(
        np.array(all_train_labels), np.array(all_train_preds), np.array(all_train_probs)
    )

    # 保存训练集的指标
    train_losses.append(total_loss)
    train_accs.append(train_acc)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1s.append(train_f1)
    train_roc_aucs.append(train_roc_auc)

    print(
        f"resnet18的 ——epoch: {epoch + 1} |训练集 loss: {total_loss:.5f} | 训练集acc: {acc:.2f}% |precision: {train_precision:.4f} |训练集 recall: {train_recall:.4f} | F1_score: {train_f1:.4f} | ROC AUC: {train_roc_auc:.4f}")

    return total_loss, acc, train_precision, train_recall, train_f1, train_roc_auc


# 验证函数
def validate_model(model, val_loader, loss_fn, optimizer, epoch):
    model.eval()
    total_loss = 0.
    total_corrects = 0.
    total = 0.
    all_val_labels = []
    all_val_preds = []
    all_val_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            preds = outputs.argmax(dim=1)
            probs = nn.Softmax(dim=1)(outputs).detach().cpu().numpy()  # 获取概率

            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(preds.cpu().numpy())
            all_val_probs.extend(probs)

            total_corrects += torch.sum(preds.eq(labels))
            total_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

        total_loss = total_loss / total
        accuracy = 100 * total_corrects / total

        val_acc, val_precision, val_recall, val_f1, val_roc_auc = calculate_metrics(
            np.array(all_val_labels), np.array(all_val_preds), np.array(all_val_probs)
        )

        # 保存指标
        val_losses.append(total_loss)
        val_accs.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_roc_aucs.append(val_roc_auc)

        print(
            f"epoch: {epoch + 1} | 验证集loss: {total_loss:.5f} | 验证集accuracy: {accuracy:.2f}% | precision: {val_precision:.4f} | val_recall: {val_recall:.4f} | F1_score: {val_f1:.4f} | ROC AUC: {val_roc_auc:.4f}")

        return total_loss, accuracy, val_precision, val_recall, val_f1, val_roc_auc


loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

save_dir = '/media/disk/01drive/07qiulin/pj1/models'

for epoch in range(epoches):
    train_m = train_model(model, train_loader, loss_fn, optimizer, epoch)
    val_m = validate_model(model, val_loader, loss_fn, optimizer, epoch)
    # 存每次的指标到 txt 文件
    metrics_save_path = '/media/disk/01drive/07qiulin/pj1/resnet18.txt'  # 指定保存路径
    save_metrics(epoch, train_m, val_m, metrics_save_path)

    save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存: {save_path}")


# 画图
def plot():
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(16, 12))  # 调整图表大小，容纳5个图

    # Loss 图
    plt.subplot(3, 2, 1)
    plt.plot(epochs, train_losses, 'p-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy 图
    plt.subplot(3, 2, 2)
    plt.plot(epochs, train_accs, 'p-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Precision 图
    plt.subplot(3, 2, 3)
    plt.plot(epochs, train_precisions, 'p-', label='Train Precision')
    plt.plot(epochs, val_precisions, 'r-', label='Val Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Recall 图
    plt.subplot(3, 2, 4)
    plt.plot(epochs, train_recalls, 'p-', label='Train Recall')
    plt.plot(epochs, val_recalls, 'r-', label='Val Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # F1 Score 图
    plt.subplot(3, 2, 5)
    plt.plot(epochs, train_f1s, 'p-', label='Train F1 Score')
    plt.plot(epochs, val_f1s, 'r-', label='Val F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    # 指定保存路径
    save_dir = "/media/disk/01drive/07qiulin/pj1/graphics"
    save_path = os.path.join(save_dir, "resnet18_2.png")

    # 保存图像
    plt.savefig(save_path)


plot()
