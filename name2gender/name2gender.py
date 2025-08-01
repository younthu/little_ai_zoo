from math import e
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
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

class NameGenderMLP:
    def __init__(self, csv_path, batch_size=32, epochs=20, learning_rate=0.001):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

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
        df = df[df['Gender'].isin(['M', 'F'])]

        df['Name'] = df['Name'].apply(self.pad_name)
        df['Char1'] = df['Name'].str[0]
        df['Char2'] = df['Name'].str[1]
        df['Char3'] = df['Name'].str[2]

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

        # print some stats
        logger.info(f'Training set size: {len(train_dataset)}')
        logger.info(f'Test set size: {len(test_dataset)}')

    class MLP(nn.Module):
        def __init__(self, input_dim=3, hidden1=64 , hidden2=32 , output_dim=2): # hidden1 and hidden2 are hyperparameters, 64, 32 by default
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden1)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden2, output_dim)
            self.softmax = nn.Softmax(dim=1)


        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x  # CrossEntropyLoss includes softmax

    def build_model(self):
        self.model = self.MLP().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):

        # if model.pth exists, load it
        if os.path.exists('name_gender_mlp.pth'):
            self.model.load_state_dict(torch.load('name_gender_mlp.pth'))
            logger.info('Loaded existing model from name_gender_mlp.pth')
            return
        else:
            logger.info('No existing model found, training from scratch')
            
        self.model.train()
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
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            # Log the average loss for the epoch, and time taken
            logger.info(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s')
        # Persist model
        torch.save(self.model.state_dict(), 'name_gender_mlp.pth')

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

    def run(self):
        self.load_and_preprocess_data()
        self.build_model()
        self.train()
        self.evaluate()

if __name__ == '__main__':
    csv_path = 'name2gender/nameGender.Shifenzheng.20050144.csv'
    model = NameGenderMLP(csv_path)
    model.run()
