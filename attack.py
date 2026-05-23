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

# Definisce il modello target
class ModelloTarget(nn.Module):
    """Stesso modello del target model"""
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


# Definisce gli shadow model
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


# Modello di attacco
class AttackModel(nn.Module):
    """Attack model - usa probabilità + correttezza predizione + max confidence"""
    def __init__(self, n_classes=2):
        super().__init__()

        # Input:
        # - probabilità classe 0
        # - probabilità classe 1
        # - correctness flag
        # - max confidence
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


# FUNZIONE PER BILANCIARE GLI SHADOW MODEL

def sample_balanced_shadow(X_shadow, y_shadow, n_train_total, n_test_total, seed=None):
    """
    Crea uno split shadow train/test bilanciato per classe.

    Esempio:
    se n_train_total = 250 e n_test_total = 250, allora ottieni:

    Shadow train:
        - 125 campioni classe 0
        - 125 campioni classe 1

    Shadow test:
        - 125 campioni classe 0
        - 125 campioni classe 1

    In questo modo ogni shadow model viene allenato e testato
    su una distribuzione bilanciata, coerente con il target model.
    """

    rng = np.random.default_rng(seed)

    if n_train_total % 2 != 0 or n_test_total % 2 != 0:
        raise ValueError(
            "n_train_total e n_test_total devono essere pari per fare uno split bilanciato."
        )

    n_train_per_class = n_train_total // 2
    n_test_per_class = n_test_total // 2

    idx0 = np.where(y_shadow == 0)[0]
    idx1 = np.where(y_shadow == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    needed_per_class = n_train_per_class + n_test_per_class

    if len(idx0) < needed_per_class or len(idx1) < needed_per_class:
        raise ValueError(
            "Campioni insufficienti nello shadow set per creare uno split bilanciato."
        )

    # Classe 0
    train_idx0 = idx0[:n_train_per_class]
    test_idx0 = idx0[n_train_per_class:n_train_per_class + n_test_per_class]

    # Classe 1
    train_idx1 = idx1[:n_train_per_class]
    test_idx1 = idx1[n_train_per_class:n_train_per_class + n_test_per_class]

    # Unione degli indici
    train_idx = np.concatenate([train_idx0, train_idx1])
    test_idx = np.concatenate([test_idx0, test_idx1])

    # Shuffle finale per non avere prima tutti 0 e poi tutti 1
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    shadow_train_x = X_shadow[train_idx]
    shadow_train_y = y_shadow[train_idx]

    shadow_test_x = X_shadow[test_idx]
    shadow_test_y = y_shadow[test_idx]

    return shadow_train_x, shadow_train_y, shadow_test_x, shadow_test_y


# MAIN

def main():
    print("\n")
    print("\n")
    print("MEMBERSHIP INFERENCE ATTACK")

    # Carica dati creati da target.py
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

    # Dati per SHADOW, separati dal target
    X_shadow = data['X_shadow']
    y_shadow = data['y_shadow']

    print("\n--- DIVISIONE ---")
    print(f"Target - Membri (train): {len(X_train)}")
    print(f"Target - Non-membri (test): {len(X_test)}")
    print(f"Shadow data (SEPARATI!): {len(X_shadow)}")

    print("\nDistribuzione target:")
    print(f"  Target train classi: {np.bincount(y_train)}")
    print(f"  Target test classi:  {np.bincount(y_test)}")

    print("\nDistribuzione shadow totale:")
    print(f"  Shadow classi: {np.bincount(y_shadow)}")

    # Carica target model
    print("\n")
    print("\n")
    print("CARICA TARGET MODEL")

    target_model = ModelloTarget(input_size=X_train.shape[1])

    if os.path.exists("target_model.pth"):
        target_model.load_state_dict(torch.load("target_model.pth", weights_only=True))
        print("Loaded target_model.pth")
    else:
        print("ERROR: target_model.pth not found!")
        return

    target_model.eval()

    # Train Shadow Models su dati separati
    print("\n")
    print("\n")
    print("TRAINING SHADOW MODELS - DATI BILANCIATI")

    n_shadow = 10
    shadow_epochs = 100

    print(f"\nUsando {len(X_shadow)} campioni per gli shadow, con 0% overlap con il target")

    attack_train_x = []
    attack_train_y = []

    for i in range(n_shadow):
        print(f"\nTraining shadow model {i + 1}/{n_shadow}...")

        train_size = len(X_train)
        test_size = len(X_test)

        # Split bilanciato per classe
        shadow_train_x, shadow_train_y, shadow_test_x, shadow_test_y = sample_balanced_shadow(
            X_shadow=X_shadow,
            y_shadow=y_shadow,
            n_train_total=train_size,
            n_test_total=test_size,
            seed=21312 + i
        )

        print(f"  Shadow train samples: {len(shadow_train_x)}")
        print(f"    Classi train: {np.bincount(shadow_train_y)}")

        print(f"  Shadow test samples: {len(shadow_test_x)}")
        print(f"    Classi test:  {np.bincount(shadow_test_y)}")

        # Allena shadow model
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

        # Raccogli dati per addestrare l'attack model
        with torch.no_grad():
            train_probs = shadow_model.get_probabilities(
                torch.tensor(shadow_train_x, dtype=torch.float32)
            ).numpy()

            test_probs = shadow_model.get_probabilities(
                torch.tensor(shadow_test_x, dtype=torch.float32)
            ).numpy()

            train_pred = np.argmax(train_probs, axis=1)
            test_pred = np.argmax(test_probs, axis=1)

        # Correttezza: 1 se la predizione è corretta, 0 altrimenti
        train_correct = (train_pred == shadow_train_y).astype(np.float32).reshape(-1, 1)
        test_correct = (test_pred == shadow_test_y).astype(np.float32).reshape(-1, 1)

        # Max confidence
        train_max_conf = train_probs.max(axis=1).reshape(-1, 1)
        test_max_conf = test_probs.max(axis=1).reshape(-1, 1)

        # Features:
        # [probabilità classe 0, probabilità classe 1, correctness, max confidence]
        train_features = np.hstack([
            train_probs,
            train_correct,
            train_max_conf
        ])

        test_features = np.hstack([
            test_probs,
            test_correct,
            test_max_conf
        ])

        # Bilancia membri e non-membri nel dataset dell'attack model
        min_size = min(len(train_features), len(test_features))

        attack_train_x.append(train_features[:min_size])
        attack_train_x.append(test_features[:min_size])

        attack_train_y.extend([1] * min_size)  # Members
        attack_train_y.extend([0] * min_size)  # Non-members

    attack_train_x = np.vstack(attack_train_x)
    attack_train_y = np.array(attack_train_y)

    # Shuffle finale del dataset dell'attack model
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
                pred = torch.argmax(
                    attack_model(torch.tensor(attack_train_x, dtype=torch.float32)),
                    dim=1
                )

                acc = (pred.numpy() == attack_train_y).mean()

            print(f"  Epoch {epoch}: loss={total_loss:.4f}, train_acc={acc:.4f}")

    # Test dell'attacco sul target model
    print("\n")
    print("\n")
    print("ATTACCO")

    target_model.eval()

    with torch.no_grad():
        member_probs = target_model.get_probabilities(
            torch.tensor(X_train, dtype=torch.float32)
        ).numpy()

        nonmember_probs = target_model.get_probabilities(
            torch.tensor(X_test, dtype=torch.float32)
        ).numpy()

        member_pred = np.argmax(member_probs, axis=1)
        nonmember_pred = np.argmax(nonmember_probs, axis=1)

    # Correttezza
    member_correct = (member_pred == y_train).astype(np.float32).reshape(-1, 1)
    nonmember_correct = (nonmember_pred == y_test).astype(np.float32).reshape(-1, 1)

    # Max confidence
    member_max_conf = member_probs.max(axis=1).reshape(-1, 1)
    nonmember_max_conf = nonmember_probs.max(axis=1).reshape(-1, 1)

    # Features complete
    member_features = np.hstack([
        member_probs,
        member_correct,
        member_max_conf
    ])

    nonmember_features = np.hstack([
        nonmember_probs,
        nonmember_correct,
        nonmember_max_conf
    ])

    # Bilancia membri e non-membri per la valutazione dell'attacco
    min_size = min(len(member_features), len(nonmember_features))

    attack_test_x = np.vstack([
        member_features[:min_size],
        nonmember_features[:min_size]
    ])

    attack_test_y = np.array(
        [1] * min_size + [0] * min_size
    )

    # Predizioni dell'attack model
    attack_model.eval()

    with torch.no_grad():
        attack_probs = F.softmax(
            attack_model(torch.tensor(attack_test_x, dtype=torch.float32)),
            dim=1
        )

        attack_pred = torch.argmax(attack_probs, dim=1).numpy()

    # Risultati
    print("RESULTS")

    accuracy = accuracy_score(attack_test_y, attack_pred)

    print(f"\nAttack Accuracy: {accuracy:.4f}")
    print(f"Random Baseline: 0.5000")
    print(f"Attack Advantage: {accuracy - 0.5:.4f}")

    print("\nDetailed Report:")
    print(
        classification_report(
            attack_test_y,
            attack_pred,
            target_names=['Non-Member', 'Member']
        )
    )

    print("Confusion Matrix:")
    cm = confusion_matrix(attack_test_y, attack_pred)

    print(f"                 Predicted")
    print(f"              Non-Mem  Member")
    print(f"Actual Non-Mem  {cm[0, 0]:4d}    {cm[0, 1]:4d}")
    print(f"Actual Member   {cm[1, 0]:4d}    {cm[1, 1]:4d}")


if __name__ == '__main__':
    main()