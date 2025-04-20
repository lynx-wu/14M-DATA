import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import numpy as np
from tqdm import tqdm

# 设置TensorFlow日志级别，抑制警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# 检查是否有GPU可用，并设置按需分配显存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs available, using CPU instead.")

# 数据加载和预处理
image_dir = '/media/disk/Backup/02drive/SynFundus-1M/SunFundus-1M-unzip/Fundus'
csv_file = '/media/disk/Backup/02drive/11yudian/data_enhancement/SynFundus-1M_annotation_with_is_readable.csv'
data = pd.read_csv(csv_file)
data['file_path'] = data['file_name'].apply(lambda x: os.path.join(image_dir, x))
data = data[data['file_path'].apply(os.path.exists)]
data['is_readable'] = data[['is_fundus', 'is_macular_readable', 'is_optic_disc_readable', 'is_retinal_region_readable']].all(axis=1).astype(int)

# 调整数据集划分，使用较小的验证集
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 缩小数据集规模以加快实验
train_data = train_data.sample(n=5000, random_state=42)
val_data = val_data.sample(n=1000, random_state=42)
test_data = test_data.sample(n=5000, random_state=42)

# 计算每个类别的样本数量
sample_counts = train_data[['is_fundus', 'is_macular_readable', 'is_optic_disc_readable', 'is_retinal_region_readable']].sum()
total_samples = len(train_data)

# 计算权重
class_weight = {
    i: total_samples / (4 * sample_counts[i]) for i in range(4)
}

print(f"Calculated class weights based on sample proportions: {class_weight}")

# 定义加载和预处理图像的函数，包含数据增强
def load_image(file_path, label, training=False):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = image / 255.0
    return image, label

# 创建TensorFlow数据集的函数
def create_dataset(data, batch_size=64, training=False):
    paths = data['file_path'].values
    labels = data[['is_fundus', 'is_macular_readable', 'is_optic_disc_readable', 'is_retinal_region_readable']].values
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda x, y: load_image(x, y, training), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(buffer_size=len(data)).repeat()
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_data, training=True)
val_dataset = create_dataset(val_data)
test_dataset = create_dataset(test_data)

# 自定义回调函数，用于在每个epoch结束后打印验证集的准确率和其他信息
class PrintValMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        val_steps = logs.get('val_steps', 'N/A')
        print(f"Epoch {epoch + 1} finished.")
        print(f" - Validation steps completed: {val_steps}")
        if val_accuracy is not None and val_loss is not None:
            print(f" - Validation Accuracy = {val_accuracy:.4f}, Validation Loss = {val_loss:.4f}")
        else:
            print(f" - Validation metrics are not available. Logs: {logs}")
        print(f" - Validation dataset length: {len(val_dataset)} batches")

# 自定义回调函数，显示训练进度条
class ProgressBarCallback(Callback):
    def __init__(self, total_epochs, total_steps):
        super().__init__()
        self.total_epochs = total_epochs
        self.total_steps = total_steps

    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.total_epochs, desc='Epochs', position=0)
        self.step_bar = tqdm(total=self.total_steps, desc='Steps', position=1, leave=True)
    
    def on_epoch_begin(self, epoch, logs=None):
        self.step_bar.reset()
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
    
    def on_batch_end(self, batch, logs=None):
        self.step_bar.update(1)
    
    def on_train_end(self, logs=None):
        self.epoch_bar.close()
        self.step_bar.close()

# 构建InceptionV3模型
def build_inception_v3(input_shape=(224, 224, 3), num_classes=4):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

# 构建并编译模型
model = build_inception_v3()
model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# 定义训练模型的函数
def train_model(model, train_dataset, val_dataset, epochs=100, val_steps=16):
    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    print_val_metrics = PrintValMetricsCallback()  # 实例化自定义回调
    progress_bar = ProgressBarCallback(total_epochs=epochs, total_steps=len(train_data) // 32)  # 实例化进度条回调
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=0,  # 禁止默认的详细日志输出
        steps_per_epoch=len(train_data) // 32,
        validation_steps=val_steps,  # 显式设置验证步数
        callbacks=[early_stopping, reduce_lr, print_val_metrics, progress_bar],  # 添加自定义回调到回调列表
        class_weight=class_weight  # 添加class_weight参数
    )
    return history

history = train_model(model, train_dataset, val_dataset)

# 定义绘制训练历史和验证集指标的函数
def plot_training_history_with_validation(history, save_path):
    plt.figure(figsize=(20, 6))
    
    # 绘制训练和验证的准确率
    plt.subplot(1, 4, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Epochs and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制训练和验证的精确率
    plt.subplot(1, 4, 2)
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Epochs and Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    
    # 绘制训练和验证的召回率
    plt.subplot(1, 4, 3)
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Epochs and Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    
    # 绘制训练和验证的损失
    plt.subplot(1, 4, 4)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Epochs and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"训练历史和验证集指标图像已保存到 {save_path}")

# 保存训练和验证历史到图像
plot_training_history_with_validation(history, save_path='/media/disk/Backup/02drive/11yudian/data_enhancement/keduxing/google/training_history_with_validation.png')

# 定义评估模型并绘制泛化能力评估的函数
def evaluate_and_plot_generalization(model, test_dataset, save_path):
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_pred = model.predict(test_dataset)
    y_pred_labels = np.where(y_pred > 0.5, 1, 0)
    
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred_labels.argmax(axis=1))
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 4, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    print(classification_report(y_true.argmax(axis=1), y_pred_labels.argmax(axis=1)))
    
    plt.subplot(1, 4, 2)
    for i in range(4): 
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f}) for class {i}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.subplot(1, 4, 3)
    for i in range(4):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    
    # 添加 Recall 的绘图
    plt.subplot(1, 4, 4)
    recall_values = [np.mean(recall) for recall in history.history['recall']]
    plt.plot(recall_values, label='Recall Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"泛化能力评估图像已保存到 {save_path}")

# 评估模型并保存泛化能力评估图像
evaluate_and_plot_generalization(model, test_dataset, save_path='/media/disk/Backup/02drive/11yudian/data_enhancement/keduxing/google/generalization_performance.png')

# 保存训练好的模型
save_path = '/media/disk/Backup/02drive/11yudian/data_enhancement/keduxing/google/pretrained_inceptionv3_model.h5'
model.save(save_path)
