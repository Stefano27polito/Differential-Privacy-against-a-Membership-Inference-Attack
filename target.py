import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
torch.manual_seed(42)


# 1. CARICAMENTO DATI

df = pd.read_csv("adult.csv")

# Rimuove colonne identificative
for col in ['name', 'ssn']:
    # Se queste colonne esistono, le rimuoviamo perché potrebbero essere identificative
    # e non vogliamo che il modello le usi per memorizzare
    if col in df.columns:
        df = df.drop(columns=[col])

# Rimuove righe con valori mancanti
df = df.replace(" ?", np.nan).replace("?", np.nan).dropna()
df = df.reset_index(drop=True)

# IMPORTANTE: Preprocessa TUTTO il dataset prima di dividere

label_encoder = LabelEncoder()
df["income"] = label_encoder.fit_transform(df["income"].astype(str))  # trasforma in 0/1

y_all = df["income"].values
X_df = df.drop(columns=["income"])

for col in X_df.columns:
    if X_df[col].dtype == "object" or str(X_df[col].dtype) == "object":
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

X_all = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

print(f"Dataset totale: {len(X_all)} campioni")


# 2. DIVISIONE BILANCIATA DEL DATASET

# 500 campioni per il target model:
# - 250 campioni di classe 0
# - 250 campioni di classe 1

np.random.seed(42)

TARGET_SIZE = 500
N_PER_CLASS = TARGET_SIZE // 2

# Indici dei campioni appartenenti alle due classi
idx_class_0 = np.where(y_all == 0)[0]
idx_class_1 = np.where(y_all == 1)[0]

# Mescola separatamente gli indici delle due classi
np.random.shuffle(idx_class_0)
np.random.shuffle(idx_class_1)

# Seleziona 250 campioni per classe per il target
target_idx_0 = idx_class_0[:N_PER_CLASS]
target_idx_1 = idx_class_1[:N_PER_CLASS]

# Unisce gli indici del target
target_indices = np.concatenate([target_idx_0, target_idx_1])
np.random.shuffle(target_indices)

# Gli shadow sono tutti i campioni NON usati dal target
all_indices = np.arange(len(X_all))
shadow_indices = np.setdiff1d(all_indices, target_indices)

X_target = X_all[target_indices]
y_target = y_all[target_indices]

print(f"\nDIVISIONE BILANCIATA")
print(f"D_target (vittima): {len(target_indices)} campioni")
print(f"  Classe 0: {(y_target == 0).sum()}")
print(f"  Classe 1: {(y_target == 1).sum()}")
print(f"D_shadow (attacker): {len(shadow_indices)} campioni")


# Split del target:
# - 50% train = membri
# - 50% test = non-membri
# stratify=y_target garantisce che anche train e test siano bilanciati

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X_target,
    y_target,
    np.arange(len(X_target)),
    test_size=0.5,
    random_state=42,
    stratify=y_target
)

print(f"\nTarget model:")
print(f"  Train samples (MEMBRI): {len(X_train)}")
print(f"    Classe 0: {(y_train == 0).sum()}")
print(f"    Classe 1: {(y_train == 1).sum()}")

print(f"  Test samples (NON-MEMBRI del target): {len(X_test)}")
print(f"    Classe 0: {(y_test == 0).sum()}")
print(f"    Classe 1: {(y_test == 1).sum()}")


# 3. MODELLO GRANDE

class OverfittingMLP(nn.Module):
    """Modello GRANDE che può memorizzare i dati"""
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


# 4. TRAINING

model = OverfittingMLP(input_size=X_train.shape[1])
print(model)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

criterion = nn.CrossEntropyLoss()

# Nessun weight_decay = nessuna regolarizzazione L2
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Molte epoche per favorire overfitting
epochs = 200

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 20 == 0 or epoch == epochs - 1:
        model.eval()
        with torch.no_grad():
            train_pred = torch.argmax(
                model(torch.tensor(X_train, dtype=torch.float32)),
                dim=1
            ).numpy()

            test_pred = torch.argmax(
                model(torch.tensor(X_test, dtype=torch.float32)),
                dim=1
            ).numpy()

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # Se vuoi monitorare l'overfitting durante il training,
        # togli il commento alla riga seguente.
        # print(f"Epoch {epoch:3d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Gap: {train_acc - test_acc:.4f}")


# 5. VALUTAZIONE FINALE

model.eval()

with torch.no_grad():
    train_pred = torch.argmax(
        model(torch.tensor(X_train, dtype=torch.float32)),
        dim=1
    ).numpy()

    test_pred = torch.argmax(
        model(torch.tensor(X_test, dtype=torch.float32)),
        dim=1
    ).numpy()

    # Confidence analysis
    train_probs = model.get_probabilities(
        torch.tensor(X_train, dtype=torch.float32)
    ).numpy()

    test_probs = model.get_probabilities(
        torch.tensor(X_test, dtype=torch.float32)
    ).numpy()

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n")
print("\n")
print("\n")

print("FINAL RESULTS")
print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Overfitting Gap: {train_acc - test_acc:.4f}")

print(f"\nConfidence Analysis:")
print(
    f"  Train max confidence: mean={train_probs.max(axis=1).mean():.4f}, "
    f"std={train_probs.max(axis=1).std():.4f}"
)

print(
    f"  Test max confidence:  mean={test_probs.max(axis=1).mean():.4f}, "
    f"std={test_probs.max(axis=1).std():.4f}"
)

print(
    f"  Confidence gap: "
    f"{train_probs.max(axis=1).mean() - test_probs.max(axis=1).mean():.4f}"
)


# 6. SALVATAGGIO

torch.save(model.state_dict(), "target_model.pth")

# Salvo i dati per l'attacco
# - X_train, y_train: MEMBRI del target
# - X_test, y_test: NON-MEMBRI del target
# - X_shadow, y_shadow: dati SEPARATI per addestrare gli shadow models

np.savez(
    "data.npz",

    # Dati del TARGET
    X_train=X_train,
    y_train=y_train,

    X_test=X_test,
    y_test=y_test,

    # Dati per SHADOW
    X_shadow=X_all[shadow_indices],
    y_shadow=y_all[shadow_indices],

    # Indici per tracciabilità
    target_indices=target_indices,
    shadow_indices=shadow_indices
)

print("\nModello salvato in: target_model.pth")
print("Dati salvati in: data.npz")