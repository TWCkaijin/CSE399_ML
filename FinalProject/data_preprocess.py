import os
import json
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

EPOCH = 12
BATCH_SIZE = 64

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256,64)
        self.fc5 = nn.Linear(64,num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

if __name__ == "__main__":
    
    train_data = pd.read_csv('./data/train.csv')
    mappings = {}
    
    
    target_col = train_data.columns[-1]
    X = train_data.drop(columns=[target_col])
    y = train_data[target_col]
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
        elif X[col].dtype == 'object':
            # 類別型欄位用眾數填充（在轉換為 codes 之前）
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
        
    
    for col in X.columns:
        if X[col].dtype == 'object':
            cat = X[col].astype('category')
            mappings[col] = list(cat.cat.categories)
            X[col] = cat.cat.codes
    
    
    if y.dtype == 'object':
        y_cat = y.astype('category')
        mappings[target_col] = list(y_cat.cat.categories)
        y = y_cat.cat.codes
        
    
    num_classes = len(y.unique())
    print("Unique labels:", sorted(y.unique()))
    print("Data shape:", X.shape)
    print("Label distribution:", y.value_counts())
    
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42, stratify=y
    )
    
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(torch.isnan(input=X_train_tensor).sum())
    input()
    
    input_size = X_train.shape[1]
    model = Model(input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    
    EPOCH = 12
    best_val_acc = 0.0
    
    
    train_losses = []
    val_accuracies = []
    
    for epoch in tqdm(range(EPOCH), desc="Training Progress"):
        
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCH}", leave=False)
        for batch_X, batch_y in train_bar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCH}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Accuracy: {val_accuracy:.2f}%')
        
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            
            
            torch.save(scaler, 'models/scaler.pth')
            with open('models/label_mappings.json', 'w', encoding='utf-8') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png')
    plt.show()