import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(21312)
torch.manual_seed(21312)


# MODELLI

class ModelloTarget(nn.Module):
    """Stesso modello di modello target"""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    
    def get_probabilities(self, x):
        with torch.no_grad():
            logits = self.net(x)
            return F.softmax(logits, dim=1)


class ShadowMLP(nn.Module):
    """Shadow model"""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)
    
    def get_probabilities(self, x):
        with torch.no_grad():
            logits = self.net(x)
            return F.softmax(logits, dim=1)


class AttackModel(nn.Module):
    """Attack model - usa probabilità + correttezza predizione"""
    def __init__(self, n_classes=2):
        super().__init__()
        # Input: n_classes probabilities + 1 correctness flag + 1 max confidence
        self.net = nn.Sequential(
            nn.Linear(n_classes + 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


# MAIN

def main():
    print("\n")
    print("\n")
    print("MEMBERSHIP INFERENCE ATTACK")
        
    # Carica dati (creati da target.py)
    if not os.path.exists("data.npz"):
        print("ERROR: data.npz not found")
        print("Run target.py first.")
        return
    
    data = np.load("data.npz")
    
    # Dati del TARGET 
    X_train = data['X_train']  # MEMBRI del target
    y_train = data['y_train']
    X_test = data['X_test']    # NON-MEMBRI del target
    y_test = data['y_test']
    
    # Dati per SHADOW (SEPARATI dal target)
    X_shadow = data['X_shadow']
    y_shadow = data['y_shadow']


    print("\n--- DIVISIONE  ---")
    print(f"Target - Membri (train): {len(X_train)}")
    print(f"Target - Non-membri (test): {len(X_test)}")
    print(f"Shadow data (SEPARATI!): {len(X_shadow)}")
    
    #Carica target model
    print("\n")
    print("\n")
    print("carica TARGET MODEL")

    
    target_model = ModelloTarget(input_size=X_train.shape[1])
    
    MODEL_PATH = "target_model_dp_sigma1.2_eps20.48.pth"
    # MODEL_PATH = "target_model_no_dp.pth"

    if os.path.exists(MODEL_PATH):
        target_model.load_state_dict(torch.load(MODEL_PATH, weights_only=True)) 
        print(f"Loaded {MODEL_PATH}")
    else:
        print(f"ERROR: {MODEL_PATH} not found!")
        return
        
    target_model.eval()
    
  
    #  Train Shadow Models (su dati SEPARATI)
    print("\n")
    print("\n")

    print("TRAINING SHADOW MODELS (su dati SEPARATI dal target)")
    
    n_shadow = 10
    shadow_epochs = 100
    
    # USA SOLO I DATI SHADOW (mai visti dal target!) 
    print(f"\nUsando {len(X_shadow)} campioni per gli shadow (0% overlap con target)")
    
    attack_train_x = []
    attack_train_y = []
    
    for i in range(n_shadow):
        print(f"Training shadow model {i+1}/{n_shadow}...")
        
        # Campiona dati SOLO da X_shadow (SEPARATI dal target)
        train_size = len(X_train)
        indices = np.random.permutation(len(X_shadow))
        
        shadow_train_x = X_shadow[indices[:train_size]]
        shadow_train_y = y_shadow[indices[:train_size]]
        shadow_test_x = X_shadow[indices[train_size:2*train_size]]
        shadow_test_y = y_shadow[indices[train_size:2*train_size]]
        
        # Allena shadow model (con overfitting simile al target)
        shadow_model = ShadowMLP(input_size=X_train.shape[1])
        dataset = TensorDataset(
            torch.tensor(shadow_train_x, dtype=torch.float32),
            torch.tensor(shadow_train_y, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        optimizer = optim.Adam(shadow_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(shadow_epochs):
            shadow_model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(shadow_model(xb), yb)
                loss.backward()
                optimizer.step()
        
        shadow_model.eval()
        
        # Raccogli dati per attack (inclusi correttezza e max confidence)
        with torch.no_grad():
            train_probs = shadow_model.get_probabilities(torch.tensor(shadow_train_x, dtype=torch.float32)).numpy()
            test_probs = shadow_model.get_probabilities(torch.tensor(shadow_test_x, dtype=torch.float32)).numpy()
            
            train_pred = np.argmax(train_probs, axis=1)
            test_pred = np.argmax(test_probs, axis=1)
        
        # Correttezza: 1 se predizione corretta, 0 altrimenti
        train_correct = (train_pred == shadow_train_y).astype(np.float32).reshape(-1, 1)
        test_correct = (test_pred == shadow_test_y).astype(np.float32).reshape(-1, 1)
        
        # Max confidence
        train_max_conf = train_probs.max(axis=1).reshape(-1, 1)
        test_max_conf = test_probs.max(axis=1).reshape(-1, 1)
        
        # Features: [probs, correctness, max_confidence]
        train_features = np.hstack([train_probs, train_correct, train_max_conf])
        test_features = np.hstack([test_probs, test_correct, test_max_conf])
        
        # Bilancia le classi
        min_size = min(len(train_features), len(test_features))
        attack_train_x.append(train_features[:min_size])
        attack_train_x.append(test_features[:min_size])
        attack_train_y.extend([1] * min_size)  # Members
        attack_train_y.extend([0] * min_size)  # Nonimembers
    
    attack_train_x = np.vstack(attack_train_x)
    attack_train_y = np.array(attack_train_y)
    
    # Shuffle
    indices = np.random.permutation(len(attack_train_x))
    attack_train_x = attack_train_x[indices]
    attack_train_y = attack_train_y[indices]
    
    print(f"\nAttack training data: {len(attack_train_x)} samples")
    print(f"  Members: {(attack_train_y == 1).sum()}")
    print(f"  Non-members: {(attack_train_y == 0).sum()}")
    
    # Train Attack Model 
    print("\n")
    print("\n")
    print("TRAINING ATTACK MODEL")
    
    attack_model = AttackModel(n_classes=2)
    dataset = TensorDataset(
        torch.tensor(attack_train_x, dtype=torch.float32),
        torch.tensor(attack_train_y, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        attack_model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(attack_model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            attack_model.eval()
            with torch.no_grad():
                pred = torch.argmax(attack_model(torch.tensor(attack_train_x, dtype=torch.float32)), dim=1)
                acc = (pred.numpy() == attack_train_y).mean()
            print(f"  Epoch {epoch}: loss={total_loss:.4f}, train_acc={acc:.4f}")
    
    # Test
    print("\n")
    print("\n")   
    print("ATTACCO")
  
    # Prepara i dati per testare l'attacco sul TARGET MODEL
    # (usando lo stesso formato di features usato nel training dell'attack model)
    target_model.eval()
    with torch.no_grad():
        member_probs = target_model.get_probabilities(torch.tensor(X_train, dtype=torch.float32)).numpy()
        nonmember_probs = target_model.get_probabilities(torch.tensor(X_test, dtype=torch.float32)).numpy()
        
        member_pred = np.argmax(member_probs, axis=1)
        nonmember_pred = np.argmax(nonmember_probs, axis=1)
    
    # Correttezza
    member_correct = (member_pred == y_train).astype(np.float32).reshape(-1, 1)
    nonmember_correct = (nonmember_pred == y_test).astype(np.float32).reshape(-1, 1)
    
    # Max confidence
    member_max_conf = member_probs.max(axis=1).reshape(-1, 1)
    nonmember_max_conf = nonmember_probs.max(axis=1).reshape(-1, 1)
    
    # Features complete
    member_features = np.hstack([member_probs, member_correct, member_max_conf])
    nonmember_features = np.hstack([nonmember_probs, nonmember_correct, nonmember_max_conf])
    
    # Bilancia per valutazione corretta
    min_size = min(len(member_features), len(nonmember_features))
    attack_test_x = np.vstack([member_features[:min_size], nonmember_features[:min_size]])
    attack_test_y = np.array([1] * min_size + [0] * min_size)
    
    # Predizioni
    attack_model.eval()
    with torch.no_grad():
        attack_probs = F.softmax(attack_model(torch.tensor(attack_test_x, dtype=torch.float32)), dim=1)
        attack_pred = torch.argmax(attack_probs, dim=1).numpy()
    
    #Risultati
    
    print("RESULTS")
    
    accuracy = accuracy_score(attack_test_y, attack_pred)
    print(f"\nAttack Accuracy: {accuracy:.4f}")
    print(f"Random Baseline: 0.5000")
    print(f"Attack Advantage: {accuracy - 0.5:.4f}")
    
    print("\nDetailed Report:")
    print(classification_report(
        attack_test_y, attack_pred,
        target_names=['Non-Member', 'Member']
    ))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(attack_test_y, attack_pred)
    print(f"                 Predicted")
    print(f"              Non-Mem  Member")
    print(f"Actual Non-Mem  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Actual Member   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
if __name__ == '__main__':
    main()
