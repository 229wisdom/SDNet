import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from ST_model import SwinTransformer
import matplotlib.pyplot as plt
import re
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, header=0)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns in the loaded CSV: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        raise

def convert_time_to_seconds(time_str):
    """将时间字符串（格式为 MM:SS.s）转换为秒数"""
    match = re.match(r'^(\d{2}):(\d{2})\.(\d+)', time_str)
    if not match:
        raise ValueError(f"Time data '{time_str}' does not match format 'MM:SS.s'")

    minutes, seconds, decimal_seconds = map(float, match.groups())
    total_seconds = int(minutes) * 60 + int(seconds) + float(f"0.{int(decimal_seconds)}")
    return total_seconds


def preprocess_time_data(time_array, column_index=0):
    try:
        time_series = time_array[:, column_index]


        time_seconds = np.array([convert_time_to_seconds(str(t)) for t in time_series])

        return time_seconds
    except Exception as e:
        print(f"An error occurred while processing time data: {e}")
        raise

def preprocess_data(X_pressure):
    scaler = MinMaxScaler()
    X_pressure_scaled = scaler.fit_transform(X_pressure)


    X_pressure_reshaped = X_pressure_scaled.reshape(X_pressure_scaled.shape[0], 1, 44, 52)


    X_pressure_padded = np.pad(X_pressure_reshaped,
                               ((0, 0), (0, 0), (0, 20), (0, 12)),
                               mode='constant', constant_values=0)

    return X_pressure_padded


def train_model(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")
    for data in progress_bar:
        X_batch_pressure, y_batch, time_batch = data
        X_batch_pressure, y_batch, time_batch = X_batch_pressure.to(device), y_batch.to(device), time_batch.to(device)
        y_batch = y_batch.squeeze(1)

        optimizer.zero_grad()
        outputs = model(X_batch_pressure, time_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = outputs.max(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
        batch_count += 1
        progress_bar.set_postfix({'Loss': running_loss / batch_count, 'Accuracy': 100 * correct / total})
    accuracy = correct / total
    return running_loss / len(train_loader), accuracy


def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch_pressure, y_batch, time_batch in test_loader:
            X_batch_pressure, y_batch, time_batch = X_batch_pressure.to(device), y_batch.to(device), time_batch.to(device)
            y_batch = y_batch.squeeze(1)
            outputs = model(X_batch_pressure,time_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            _, preds = outputs.max(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total
    return running_loss / len(test_loader), accuracy

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, epochs):
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='train-loss')
    plt.plot(epochs_range, test_losses, label='test-loss')
    plt.title('Loss curve for each epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accuracies, label='train-accuracy')
    plt.plot(epochs_range, test_accuracies, label='test-accuracy')
    plt.title('Accuracy curve for each epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_evaluate(X_pressure, y, time_data, batch_size=32, epochs=100):

    assert len(X_pressure) == len(y) == len(time_data), "Length of datasets must match"

    indices = np.arange(len(X_pressure))
    np.random.seed(42)
    np.random.shuffle(indices)

    X_pressure_shuffled = X_pressure[indices]
    y_shuffled = y[indices]
    time_data_shuffled = time_data[indices]


    split_idx = int(len(X_pressure) * 0.8)
    X_train_pressure, X_test_pressure = X_pressure_shuffled[:split_idx], X_pressure_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    time_train, time_test = time_data_shuffled[:split_idx], time_data_shuffled[split_idx:]

    X_train_pressure_tensor = torch.tensor(X_train_pressure, dtype=torch.float32)
    X_test_pressure_tensor = torch.tensor(X_test_pressure, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    time_train_pressure_tensor = torch.tensor(time_train, dtype=torch.float32)
    time_test_pressure_tensor = torch.tensor(time_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_pressure_tensor, y_train_tensor, time_train_pressure_tensor)
    test_dataset = TensorDataset(X_test_pressure_tensor, y_test_tensor, time_test_pressure_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    swin_model = SwinTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(swin_model.parameters(), lr=0.0001, weight_decay=1e-5)



    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    max_train_accuracy = 0
    max_test_accuracy = 0

    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(swin_model, train_loader, criterion, optimizer, epoch=epoch)
        test_loss, test_accuracy = evaluate_model(swin_model, test_loader, criterion)


        if(train_accuracy > max_train_accuracy):
            max_train_accuracy = train_accuracy
        if(test_accuracy > max_test_accuracy):
            max_test_accuracy = test_accuracy



        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, epochs)
    print("训练集最大准确率为： ", max_train_accuracy)
    print("测试集最大准确率为： ", max_test_accuracy)

    mean_accuracy = np.mean(test_accuracies)
    sem_accuracy = stats.sem(test_accuracies)
    print(f"Accuracy: {mean_accuracy:.4f} ± {sem_accuracy:.4f}")



def main():

    features_file_path = 'D:/PycharmProjects/chongxin_wisdom/未窗口化未预处理的坐垫数据.csv'  # 数据按照1，2，3，4的顺序排列
    labels_file_path ='D:/PycharmProjects/chongxin_wisdom/xu_所有数据的标签.csv'
    time_file_path = 'D:/PycharmProjects/chongxin_wisdom/zuo/x_time/output.csv'

    X_pressure = load_csv(features_file_path).values
    y = load_csv(labels_file_path).values
    time = load_csv(time_file_path).values
    print("x.type:",type(X_pressure))

    X_pressure = preprocess_data(X_pressure)

    time_seconds = preprocess_time_data(time)


    train_and_evaluate(X_pressure, y, time_seconds, batch_size=32, epochs=100)


if __name__ == '__main__':
    main()
