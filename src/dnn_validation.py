import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from src.preprocessing import build_preprocessor

class TitanicMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)
    
def train_one_fold(X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, lr=3e-3, 
                   seed=42, return_probs=False):
    torch.manual_seed(seed)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = TitanicMLP(input_dim=X_train.shape[1])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            
            loss.backward()
            optimizer.step()
        
    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor)
        probs = torch.sigmoid(logits)
        
        predictions = (probs >= 0.5).float()
        
        accuracy = (predictions == y_val_tensor).float().mean().item()
        
    if return_probs:
        return probs.squeeze(1).numpy()
        
    return accuracy
    
def evaluate_dnn(X, y, n_splits=5, epochs=100, batch_size=32, 
                 lr=3e-3, seed=42, use_family_size=False, use_fare_log=False, use_title=False):
    
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy()
        y_val = y.iloc[val_idx].to_numpy()
        
        preprocessor = build_preprocessor(
            model_type='standard',
            use_family_size=use_family_size,
            use_fare_log=use_fare_log,
            use_title=use_title,
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        accuracy = train_one_fold(X_train=X_train_processed, y_train=y_train,
                                  X_val=X_val_processed, y_val=y_val,
                                  epochs=epochs, batch_size=batch_size,
                                  lr=lr, seed=seed)
        
        scores.append(accuracy)
        
    scores = np.array(scores)
    
    return {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std(),
    }
        
def train_final_dnn(X_train, y_train, X_test, epochs=100, batch_size=32, lr=3e-3, seed=42):
    torch.manual_seed(seed)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = TitanicMLP(input_dim=X_train.shape[1])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits)
        
    return probs.squeeze(1).numpy()
        
        
    
    
    
    