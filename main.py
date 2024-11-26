import os
import random
import zipfile
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
# from transforms3d.axangles import axangle2mat  # for rotation
from scipy.interpolate import CubicSpline  # for warping
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

realpath = r'\data\real'

# Create the directory
os.makedirs(realpath, exist_ok=True)

print(f"Directory '{realpath}' created successfully.")

virtpath = r'\data\virtual'

# Create the directory
os.makedirs(virtpath, exist_ok=True)

print(f"Directory '{virtpath}' created successfully.")

rootdir = r'D:\code\OpenPackChallenge2025'  # replace with your project path
real_directory = rootdir + realpath
virt_directory = rootdir + virtpath
# directory = r'D:\code\OpenPackChallenge2025\data\real'


device = 'cpu'

user_paths = {}
for root, dirs, files in os.walk(real_directory):
    for file in files:
        if file.endswith('S0100.csv'):
            user_paths[file[:-10]] = os.path.join(root, file)
for u, d in user_paths.items():
    print('%s at: %s'% (u,d))

userIDs = list(user_paths.keys())

# Shuffle the list to ensure randomness
random.shuffle(userIDs)

# Calculate the split indices
total_length = len(userIDs)
train_size = int(total_length * 0.7)  # 70% of 10
val_size = int(total_length * 0.1)  # 10% of 10
test_size = total_length - train_size - val_size  # 20% of 10

# Split the list according to the calculated sizes
train_users = np.sort(userIDs[:train_size])      # First 70%
val_users = np.sort(userIDs[train_size:train_size + val_size])  # Next 10%
test_users = np.sort(userIDs[train_size + val_size:])  # Last 20%

print('Training set: %s'%train_users)
print('Validation set: %s'%val_users)
print('Test set: %s'%test_users)

selected_columns = ['atr01/acc_x', 'atr01/acc_y', 'atr01/acc_z', 'atr02/acc_x', 'atr02/acc_y', 'atr02/acc_z',
                    'timestamp', 'operation']
train_data_dict = {}
for u in train_users:
    # Load the CSV file with only the selected columns
    train_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

val_data_dict = {}
for u in val_users:
    # Load the CSV file with only the selected columns
    val_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

test_data_dict = {}
for u in test_users:
    # Load the CSV file with only the selected columns
    test_data_dict[u] = pd.read_csv(user_paths[u], usecols=selected_columns)

# only select data and label columns
new_columns = selected_columns[:6] + [selected_columns[-1]]
print('Data for train, validation, and test: %s'%new_columns)

# find csv files in 'data/virtual'
virt_paths = []
for root, dirs, files in os.walk(virt_directory):
    for file in files:
        if file.endswith('.csv'):
            virt_paths.append(os.path.join(root, file))
print('Virtual csv file paths are as shown follows:')
virt_paths

# real and virtual training data

## real data
train_data = []
for u, data in train_data_dict.items():
    train_data.append(data[new_columns].values)
    # print(data[new_columns].values.shape)

## virtual data
for p in virt_paths:
    # Load the CSV file with only the selected columns
    data = pd.read_csv(p, usecols=new_columns)
    train_data.append(data.values)

train_data = np.concatenate(train_data, axis=0)
print('Shape of train data is %s' % str(train_data.shape))

# validatation and test data
val_data = []
for u, data in val_data_dict.items():
    val_data.append(data[new_columns].values)

test_data = []
for u, data in test_data_dict.items():
    test_data.append(data[new_columns].values)

val_data = np.concatenate(val_data, axis=0)
test_data = np.concatenate(test_data, axis=0)

print('Shape of validation data is %s' % str(val_data.shape))
print('Shape of test data is %s' % str(test_data.shape))

# convert operation ID to labels (from 0 to n)
labels = np.unique(train_data[:, -1])
label_dict = dict(zip(labels, np.arange(len(labels))))
train_data[:,-1] = np.array([label_dict[i] for i in train_data[:,-1]])
val_data[:,-1] =  np.array([label_dict[i] for i in val_data[:,-1]])
test_data[:,-1] =  np.array([label_dict[i] for i in test_data[:,-1]])


class data_loader_umineko(Dataset):
    def __init__(self, samples, labels, device='cpu'):
        self.samples = torch.tensor(samples).to(device)  # check data type
        self.labels = torch.tensor(labels)  # check data type

    def __getitem__(self, index):
        target = self.labels[index]
        sample = self.samples[index]
        return sample, target

    def __len__(self):
        return len(self.labels)


def sliding_window(datanp, len_sw, step):
    '''
    :param datanp: shape=(data length, dim) raw sensor data and the labels. The last column is the label column.
    :param len_sw: length of the segmented sensor data
    :param step: overlapping length of the segmented data
    :return: shape=(N, len_sw, dim) batch of sensor data segment.
    '''

    # generate batch of data by overlapping the training set
    data_batch = []
    for idx in range(0, datanp.shape[0] - len_sw - step, step):
        data_batch.append(datanp[idx: idx + len_sw, :])
    data_batch.append(datanp[-1 - len_sw: -1, :])  # last batch
    xlist = np.stack(data_batch, axis=0)  # [B, data length, dim]

    return xlist


def generate_dataloader(data, len_sw, step, if_shuffle=True):
    tmp_b = sliding_window(data, len_sw, step)
    data_b = tmp_b[:, :, :-1]
    label_b = tmp_b[:, :, -1]
    data_set_r = data_loader_umineko(data_b, label_b, device=device)
    data_loader = DataLoader(data_set_r, batch_size=batch_size,
                             shuffle=if_shuffle, drop_last=False)
    return data_loader

len_sw = 600
step = 300
batch_size = 512

train_loader = generate_dataloader(train_data, len_sw, step, if_shuffle=True)
val_loader = generate_dataloader(val_data, len_sw, step, if_shuffle=False)
test_loader = generate_dataloader(test_data, len_sw, step, if_shuffle=False)


class DeepConvLSTMSelfAttn(nn.Module):
    """Imprementation of a DeepConvLSTM with Self-Attention used in ''Deep ConvLSTM with
    self-attention for human activity decoding using wearable sensors'' (Sensors 2020).

    Note:
        https://ieeexplore.ieee.org/document/9296308 (Sensors 2020)
    """

    def __init__(
            self,
            in_ch: int = 6,
            num_classes: int = None,
            cnn_filters=3,
            lstm_units=32,
            num_attn_heads: int = 1,
    ):
        super().__init__()

        # NOTE: The first block is input layer.

        # -- [1] Embedding Layer --
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, cnn_filters, kernel_size=1, padding=0),
            nn.BatchNorm2d(cnn_filters),
            nn.ReLU(),
        )

        # -- [2] LSTM Encoder --
        self.lstm = nn.LSTM(cnn_filters, lstm_units, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)

        # -- [3] Self-Attention --
        self.attention = nn.MultiheadAttention(
            lstm_units,
            num_attn_heads,
            batch_first=True,
        )

        # -- [4] Softmax Layer (Output Layer) --
        self.out = nn.Conv2d(
            lstm_units,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape = (B, T, CH)
        Returns:
            torch.Tensor: shape = (B, N_CLASSES, T)
        """
        # -- [0] Convert Input Shape --
        x = x.permute(0, 2, 1)
        print(x.shape)
        x = x.unsqueeze(3)  # output shape = (B, CH, T, 1)

        # -- [1] Embedding Layer --
        x = self.conv(x)

        # -- [2] LSTM Encoder --
        # Reshape: (B, CH, 1, T) -> (B, T, CH)
        x = x.squeeze(3).transpose(1, 2)

        x, _ = self.lstm(x)
        x = self.dropout(x)

        # -- [3] Self-Attention --
        x, w = self.attention(x.clone(), x.clone(), x.clone())

        # -- [4] Softmax Layer (Output Layer) --
        # Reshape: (B, T, CH) -> (B, CH, T, 1)
        x = x.transpose(1, 2).unsqueeze(3)
        x = self.out(x)
        x = x.squeeze(3)
        return x  # (B, N_CLASSES, T)

device = 'cpu'
model = DeepConvLSTMSelfAttn(num_classes=len(label_dict))
model = model.to(device)

num_epochs = 2

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

learning_rate = 0.0001
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, amsgrad=True
)
optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=True
        )
lambda1 = lambda epoch: 1.0**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                               factor=0.5,
#                               patience=4,
#                               verbose=True)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered")
                return True
        return False
early_stopping = EarlyStopping()
from sklearn.metrics import f1_score


train_losses, val_losses = [], []
for epoch in tqdm(range(num_epochs)):
    train_loss, val_loss = [], []
    ###################
    # train the model #
    ###################
    model.train()
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.long)

        output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(output, label)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(np.average(train_loss))

    ######################
    # validate the model #
    ######################
    with torch.no_grad():
        model.eval()
        for i, (sample, label) in enumerate(val_loader):
            sample = sample.to(device=device, dtype=torch.float)
            label = label.to(device=device, dtype=torch.long)

            output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
            loss = criterion(output, label)
            val_loss.append(loss.item())
        val_losses.append(np.average(val_loss))

test_loss, test_losses = [], []
for epoch in tqdm(range(num_epochs)):
    model.eval()
    true_labels, pred_labels = [], []
    for i, (sample, label) in enumerate(test_loader):
        sample = sample.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.long)

        output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(output, label)
        test_loss.append(loss.item())

        true_labels.append(label.detach().cpu().numpy())
        pred_labels.append(output.detach().cpu().numpy())

    # break
    test_losses.append(np.average(test_loss))
    # break
    # Calculate F1 scores
    y_true = np.concatenate(true_labels, axis=0)
    y_prob = np.concatenate(pred_labels, axis=0)

    # Get the predicted class labels (argmax along the class dimension)
    y_pred = np.argmax(y_prob, axis=1)  # output Shape: (batch_size, time_steps)

    # Flatten the tensors for F1 score calculation
    y_pred_flat = y_true.flatten()  # Flatten to 1D array
    y_true_flat = y_pred.flatten()  # Flatten to 1D array

    # Calculate F1 score (macro F1 score)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro')

    print(f'F1 Score: {f1:.4f}')

    # Check early stopping
    if early_stopping(np.average(test_loss)):
        print("Stopping at epoch %s." % str(epoch))
        break


