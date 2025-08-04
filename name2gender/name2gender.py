from math import e
from unittest import result
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from loguru import logger


# 1. 数据预处理：将名称转换为特征
class NameDataset(Dataset):
    def __init__(self, names, genders, max_len=10):
        self.names = names
        self.genders = genders
        self.max_len = max_len
        # names to characters and create a character set
        self.names = [name.lower() for name in names]  # 转为小写
        # self.name_chars = set(''.join(self.names))
        self.chars = set(''.join(self.names))
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}  # 0留作填充
        self.vocab_size = len(self.chars) + 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx].lower()
        # 字符编码：将名称转换为固定长度的索引序列
        name_encoded = [self.char_to_idx.get(c, 0) for c in name[:self.max_len]]
        # 填充到固定长度
        name_encoded += [0] * (self.max_len - len(name_encoded))
        # 性别编码：0->女，1->男
        gender = 1 if self.genders[idx] == 'M' else 0
        return (torch.tensor(name_encoded, dtype=torch.float32),   # <-- changed here
                torch.tensor(gender, dtype=torch.float32))
    
class NameGenderMLP:
    def __init__(self, csv_path, batch_size=32 * 100, epochs=20, learning_rate=0.001, skip_hidden3=True): 
        """
        Initializes the NameGenderMLP model.
        :param csv_path: Path to the CSV file containing names and genders.
        :param batch_size: Number of samples per batch. in m2, 16g, while batch_size is 32, it takes 160s to train 1 epoch, while batch_size is 32 * 100, it takes 50s to train 1 epoch
        :param epochs: Number of epochs to train the model.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.skip_hidden3 = skip_hidden3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If mps available, use it
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')

        logger.info(f'Using device: {self.device}')

        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.label_encoder = LabelEncoder()

        self.model = None

    def pad_name(self, name, length=3):
        name = str(name)
        if len(name) < length:
            name = name + '_' * (length - len(name))
        else:
            name = name[:length]
        return name

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_path, on_bad_lines='skip')
        # Drop rows which Gender is not M or F
        df = df[df['Gender'].isin(['M', 'F'])] # # M           12773969, F            6479097

        # Get 6479097 M and 6479097 F
        df = df.groupby('Gender').sample(n=6479097, random_state=42)

        # names to str
        df['Name'] = df['Name'].astype(str)
        df['Name2'] = df['Name'].apply(self.pad_name)
        df['Char1'] = df['Name2'].str[0]
        df['Char2'] = df['Name2'].str[1]
        df['Char3'] = df['Name2'].str[2]

        X_chars = df[['Char1', 'Char2', 'Char3']]
        X_encoded = self.encoder.fit_transform(X_chars)
        y_encoded = self.label_encoder.fit_transform(df['Gender'])

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )

        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        train_names, val_names, train_genders, val_genders = train_test_split(
            df['Name'].tolist(), df['Gender'].tolist(), test_size=0.2, random_state=42
        )
        train_dataset = NameDataset(train_names, train_genders)
        val_dataset = NameDataset(val_names, val_genders)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        # print some stats
        logger.info(f'Training set size: {len(train_dataset)}')
        logger.info(f'Test set size: {len(test_dataset)}')
        
        # logger some sample training data
        logger.info(f'Sample training data: {train_dataset[0]}')

    class MLP(nn.Module):
        def __init__(self, input_dim=10, hidden1=64 * 100, hidden2=32 * 100, hidden3=32 * 100, output_dim=2, skip_hidden3=True): # input_dim=10
            super().__init__()
            self.skip_hidden3 = skip_hidden3
            self.fc1 = nn.Linear(input_dim, hidden1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden2, hidden3)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden3, output_dim)
            self.softmax = nn.Softmax(dim=1)


        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            if not self.skip_hidden3:
                x = self.fc3(x)
                x = self.relu3(x)
            x = self.fc4(x)
            x = self.softmax(x)  # Softmax is applied here, but CrossEntropyLoss expects raw logits
            return x  # CrossEntropyLoss includes softmax

    def build_model(self):
        self.model = self.MLP(input_dim=10, skip_hidden3=self.skip_hidden3).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):

        model_dir = 'name2gender/models'
        model_name = f'name_gender_mlp.e{self.epochs}.b{self.batch_size}.lr{self.learning_rate}.pth'
        model_path = os.path.join(model_dir, model_name)
        # if model.pth exists, load it
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f'Loaded existing model from {model_path}')
            return
        else:
            logger.info(f'No existing model found, training from scratch')
            
        self.model.train() # set the model to training mode
        # Log the model architecture
        logger.info(f'Model architecture: {self.model}')
        logger.info(f'Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        logger.info(f'Batch size: {self.batch_size}')
        logger.info(f'Learning rate: {self.learning_rate}')
        logger.info('Starting training...')

        # Setup tensorboard logging
        from torch.utils.tensorboard import SummaryWriter 
        self.writer = SummaryWriter(log_dir='logs/name2gender')
        self.writer.add_graph(self.model, torch.randn(1, 10).to(self.device))  # Add model graph to TensorBoard
        logger.info('TensorBoard logging initialized.')

        # Start training loop
        # Make sure the train is resumable
        logger.info(f'Starting training for {self.epochs} epochs...')
        # Track time for each epoch
        for epoch in range(self.epochs):
            # start of time tracking
            logger.info(f'Starting epoch {epoch+1}/{self.epochs}')
            start_time = time.time()
            total_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward() # 计算梯度
                self.optimizer.step() # 更新参数
                total_loss += loss.item() # Accumulate loss for the epoch
                # Log the loss to TensorBoard
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.train_loader) + len(self.train_loader))

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(self.train_loader)
            # Log the average loss for the epoch, and time taken
            logger.info(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s')
        # Persist model
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.writer.close()  # Close the TensorBoard writer
        logger.info('Training complete. Saving model...')
        torch.save(self.model.state_dict(), os.path.join(model_dir, model_name))

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        acc = accuracy_score(all_labels, all_preds)
        logger.info(f'Test Accuracy: {acc:.4f}')
        logger.info('Classification Report:')
        logger.info(classification_report(all_labels, all_preds, target_names=self.label_encoder.classes_))
        return acc
    
    def predict(self, names):
        self.model.eval()
        padded_names = [self.pad_name(name) for name in names]
        X_chars = pd.DataFrame({'Char1': [name[0] for name in padded_names],
                                'Char2': [name[1] for name in padded_names],
                                'Char3': [name[2] for name in padded_names]})
        X_encoded = self.encoder.transform(X_chars)
        X_tensor = torch.tensor(X_encoded, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)
        return self.label_encoder.inverse_transform(preds.cpu().numpy())
    
    def run(self, skip_evaluation=False):
        self.load_and_preprocess_data()
        self.build_model()
        self.train()
        if not skip_evaluation:
            self.evaluate()

def benchmark():
    """
    Benchmark the model training and evaluation.
    """
    start_time = time.time()
    model = NameGenderMLP('name2gender/nameGender.Shifenzheng.20050144.csv')
    model.run()
    logger.info(f'Total time taken: {time.time() - start_time:.2f}s')

def tune_hyperparameters():
    """
    Tune hyperparameters for the model.
    """
    # go through different batch sizes, learning rates, and epochs
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]  # Batch sizes to try
    learning_rates = [0.001, 0.0001, 0.00001]
    epochs = [10, 20, 30] # 10, 20, 30 epochs
    best_acc = 0
    best_params = {}
    results = []
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for epoch in epochs:
                start_time = time.time()
                logger.info(f'Starting hyperparameter tuning with batch_size={batch_size}, learning_rate={learning_rate}, epochs={epoch}')
                # Initialize the model with the current hyperparameters
                logger.info(f'Tuning hyperparameters: batch_size={batch_size}, learning_rate={learning_rate}, epochs={epoch}')
                model = NameGenderMLP('name2gender/nameGender.Shifenzheng.20050144.csv', batch_size=batch_size, learning_rate=learning_rate, epochs=epoch)
                model.run()
                acc = model.evaluate()
                # Calculate velocity
                elapsed_time = time.time() - start_time
                velocity = elapsed_time/epoch if elapsed_time > 0 else 0
                logger.info(f'Hyperparameters: batch_size={batch_size}, learning_rate={learning_rate}, epochs={epoch}, Accuracy: {acc:.4f}, Velocity: {velocity:.4f}')
                
                results.append((batch_size, learning_rate, epoch, acc))
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epoch}
                logger.info(f'Finished training with batch_size={batch_size}, learning_rate={learning_rate}, epochs={epoch}, Accuracy: {acc:.4f}')
    logger.info(f'Best parameters: {best_params}, Best accuracy: {best_acc:.4f}')

# Run this function to find the fastest batch size
# Try in command line: python name2gender/name2gender.py --find-fast-batch-size
def find_the_fast_batch_size():
    """
    Find the fastest batch size for the model.
    """
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]  # Batch sizes to try
    best_time = float('inf')
    best_batch_size = None
    results = []
    logger.info('Finding the fastest batch size...')
    for batch_size in batch_sizes:
        start_time = time.time()
        logger.info(f'Starting training with batch_size={batch_size}')
        model = NameGenderMLP('name2gender/nameGender.Shifenzheng.20050144.csv', batch_size=batch_size, epochs=1, learning_rate=0.001)
        model.run(skip_evaluation=True)  # Skip evaluation for speed
        elapsed_time = time.time() - start_time
        results.append((batch_size, elapsed_time))
        logger.info(f'Finished training with batch_size={batch_size}, Time taken: {elapsed_time:.2f}s')
        if elapsed_time < best_time:
            best_time = elapsed_time
            best_batch_size = batch_size
    logger.info(f'Best batch size: {best_batch_size}, Time taken: {best_time:.2f}s')

    results.sort(key=lambda x: x[1])  # Sort results by time
    logger.info('Batch Size\tTime (s):')
    for batch_size, elapsed_time in results:
        logger.info(f'{batch_size}\t{elapsed_time:.2f}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--find-fast-batch-size', action='store_true', help='Find the fastest batch size')
    args = parser.parse_args()
    if args.find_fast_batch_size:
        find_the_fast_batch_size()
    else:
        csv_path = 'name2gender/nameGender.Shifenzheng.20050144.csv' # M           12773969, F            6479097
        model = NameGenderMLP(csv_path, epochs=200, batch_size=32 * 100, learning_rate=0.001)
        # model = NameGenderMLP(csv_path)
        model.run()
        test_names = ['王欢', '刘亦菲', '张曼玉', '成龙']  # Example names for prediction
        predictions = model.predict(test_names)  # Example predictions
        logger.info(f'Predictions: {list(zip(test_names, predictions))}')

        # Launch tensorboard
        from tensorboard import program
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', 'name2gender'])
        url = tb.launch()
        logger.info(f"TensorBoard launched at {url}")
