import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from opacus import PrivacyEngine

np.random.seed(42)
torch.manual_seed(42)



# 1. CARICAMENTO DATI 

df = pd.read_csv("adult.csv")

# Rimuove colonne identificative
for col in ['name', 'ssn']: # Se queste colonne esistono, le rimuoviamo perché potrebbero essere identificative e non vogliamo che il modello le usi per memorizzare
    if col in df.columns:
        df = df.drop(columns=[col])

df = df.replace(" ?", np.nan).replace("?", np.nan).dropna() # Rimuove righe con valori mancanti (se presenti)
df = df.reset_index(drop=True) # Reset degli indici dopo la rimozione delle righe

# IMPORTANTE: Preprocessa TUTTO il dataset prima di dividere

label_encoder = LabelEncoder()
df["income"] = label_encoder.fit_transform(df["income"].astype(str)) #trasforma in 0/1

y_all = df["income"].values 
X_df = df.drop(columns=["income"]) 

for col in X_df.columns:
    if X_df[col].dtype == "object" or str(X_df[col].dtype) == "object":
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

X_all = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

print(f"Dataset totale: {len(X_all)} campioni") 

# ==========================================
# DIVISIONE BILANCIATA TARGET / SHADOW
# ==========================================

np.random.seed(42)

# 1. Indici delle classi nel dataset completo
idx_class_0 = np.where(y_all == 0)[0]
idx_class_1 = np.where(y_all == 1)[0]

# 2. Numero di campioni per classe da usare nel TARGET
TARGET_PER_CLASS = 250   # totale target = 500
TEST_SIZE = 0.5          # 250 membri, 250 non-membri

# Controllo di sicurezza
if len(idx_class_0) < TARGET_PER_CLASS or len(idx_class_1) < TARGET_PER_CLASS:
    raise ValueError("Campioni insufficienti in una delle due classi per costruire un target bilanciato.")

# 3. Campionamento bilanciato del target (senza duplicati)
target_idx_0 = np.random.choice(idx_class_0, TARGET_PER_CLASS, replace=False)
target_idx_1 = np.random.choice(idx_class_1, TARGET_PER_CLASS, replace=False)

# Indici target complessivi
target_indices = np.concatenate([target_idx_0, target_idx_1])
np.random.shuffle(target_indices)

# 4. Tutto il resto va nello shadow set
all_indices = np.arange(len(X_all))
shadow_indices = np.setdiff1d(all_indices, target_indices)

# 5. Split del target in membri (train) e non-membri (test)
# Facciamo lo split DIRETTAMENTE sugli indici originali
train_indices, test_indices = train_test_split(
    target_indices,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=y_all[target_indices]
)

# 6. Costruzione dei dataset finali
X_train = X_all[train_indices]
y_train = y_all[train_indices]

X_test = X_all[test_indices]
y_test = y_all[test_indices]

X_shadow = X_all[shadow_indices]
y_shadow = y_all[shadow_indices]


# 8. Stampe di controllo
print(f"\nDIVISIONE BILANCIATA")
print(f"D_target totale: {len(target_indices)} campioni")
print(f"  - Train target (MEMBRI): {len(train_indices)} campioni")
print(f"  - Test target (NON-MEMBRI): {len(test_indices)} campioni")
print(f"D_shadow (attacker): {len(shadow_indices)} campioni")

print("\nDistribuzione classi:")
print(f"y_train:  {np.bincount(y_train)}")
print(f"y_test:   {np.bincount(y_test)}")
print(f"y_shadow: {np.bincount(y_shadow)}")

# ==========================================
# SCALING CORRETTO: fit solo sul train target
# ==========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_shadow = scaler.transform(X_shadow)

# Manteniamo anche gli alias aggiornati
X_train_bal = X_train

# 2. MODELLO GRANDE (più capacità = più overfitting)

class OverfittingMLP(nn.Module):
    """Modello GRANDE che può memorizzare i dati"""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),  # Più neuroni
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




# 3. TRAINING (molte epoche, nessuna regolarizzazione)


model = OverfittingMLP(input_size=X_train.shape[1])
print(model)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #fornisce i dati al modello in batch di 16 campioni, mescolati ad ogni epoca (prova: aumento a 64 )

#VECCHIA SOLUZIONE CON MANUAL TUNING 
#prova a bilanciare la loss per le classi sbilanciate (se ci sono), in modo che il modello non si concentri solo sulla classe più frequente
#class_counts = np.bincount(y_train) 

#weights = torch.tensor([ 
#    len(y_train) / (2 * class_counts[0]), 
#    len(y_train) / (2 * class_counts[1]) 
#], dtype=torch.float32)

#weights[1] = weights[1] * 2.0 # Raddoppia il peso della classe 1 (reddito >50K) per bilanciare ulteriormente

criterion = nn.CrossEntropyLoss() 

# Nessun weight_decay = nessuna regolarizzazione L2
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Implementiamo DP-SGD con Opacus
USE_DP = True
max_grad_norm = 1.5 # clippinhg del gradiente, serve a impedire che un singolo dato influenzi troppo il modello
                    #1.0 valore comune, più basso = più privacy ma meno accuratezza, più alto = più accuratezza ma meno privacy (perché un singolo dato può influenzare di più il modello)
noise_multiplier = 1.5 # quantità di rumore da aggiungere (più alto = più privacy ma meno accuratezza)
                        # 1.0 è un buon compromesso, ma si puo aumentare se vogliamo piu privacy
delta = 1 / len(X_train) # probabilità di fallimento della privacy, calcolata facendo 1/N


privacy_engine = PrivacyEngine() #crea un oggetto che gestisce la privacy,
                                #ci permette di wrappare il modello e l'ottimizzatore per rendere il training privato,
                                #e di calcolare epsilon durante il training

if USE_DP:
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    ) 
    print("\nDifferential Privacy ATTIVA")
else:
    print("\nDifferential Privacy DISATTIVA")


epochs = 40 #riduco le epoche per evitare un training troppo lungo, cercando di dimuiire epsilon (da 50 a 40)

epsilons = [] #per tracciare l'evoluzione di epsilon durante il training (se DP è attivo)
epochs_list = [] #per tracciare le epoche corrispondenti agli epsilon

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    
    for xb, yb in train_loader: 
        optimizer.zero_grad() 
        loss = criterion(model(xb), yb) #confronta le predizioni con le label vere e calcola la loss
        loss.backward() 
        optimizer.step() 
        total_loss += loss.item() 

    #traccia epsilon ad ogni epoca
    if USE_DP:
        epsilon = privacy_engine.get_epsilon(delta)
        epsilons.append(epsilon)
        epochs_list.append(epoch)    
    
    if epoch % 20 == 0 or epoch == epochs - 1: # Ogni 20 epoche (o all'ultima) stampiamo l'accuracy per monitorare l'overfitting
        
        if USE_DP:
            print(f"Epoch {epoch:3d} | ε = {epsilon:.4f}")
        else:
            print(f"Epoch {epoch:3d}")

        # Valuta train e test accuracy
        model.eval()
        with torch.no_grad():
            train_pred = torch.argmax(model(torch.tensor(X_train, dtype=torch.float32)), dim=1).numpy() #prendi la classe con il logit piu alto
            test_pred = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1).numpy()
        
        train_acc = accuracy_score(y_train, train_pred) #percentuale di predizioni corrette sui membri (train)
                                                        #se alta significa che il modello ha memorizzato bene i membri
        test_acc = accuracy_score(y_test, test_pred) #percentuale di predizioni corrette sui non-membri (test)
        
        print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# traccia l'evoluzione di epsilon
if USE_DP:
    plt.figure()
    plt.plot(epochs_list, epsilons, label="Epsilon")
    plt.axhline(y=10, linestyle='--', label="Epsilon = 10") # Linea di riferimento orizzontale a epsilon=10 
                                                            #(10 è un valore comunemente considerato come limite superiore per una buona privacy)
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon (ε)")
    plt.title("Privacy Budget (ε) over Training")
    plt.grid()
    plt.show()

#test per vedere se il modello sta predicendo principalmente una classe (overfitting a una classe) o se è più bilanciato
train_pred = torch.argmax(model(torch.tensor(X_train, dtype=torch.float32)), dim=1).numpy()
test_pred = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1).numpy()

from sklearn.metrics import classification_report, confusion_matrix

print("\nCONFUSION MATRIX (TEST):")
print(confusion_matrix(y_test, test_pred))

print("\nCLASSIFICATION REPORT (TEST):")
print(classification_report(y_test, test_pred, digits=4))

print("\nDistribuzione predizioni TRAIN:")
print(np.bincount(train_pred))

print("\nDistribuzione predizioni TEST:")
print(np.bincount(test_pred))



# 4. VALUTAZIONE FINALE


model.eval()
# Se abbiamo usato DP con opacus, il modello viene wrappato, è avvolto in DataParallel, quindi dobbiamo accedere al modulo originale per ottenere le probabilità
if USE_DP:
    eval_model = model._module # Se il modello è stato wrappato da Opacus, accediamo al modulo originale per ottenere le probabilità (perché il wrapper potrebbe non implementare get_probabilities)
else:
    eval_model = model

with torch.no_grad():
    train_tensor = torch.tensor(X_train, dtype=torch.float32) 
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_pred = torch.argmax(model(torch.tensor(X_train, dtype=torch.float32)), dim=1).numpy()
    test_pred = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1).numpy()
    
    # Confidence analysis
    train_probs = eval_model.get_probabilities(train_tensor).numpy()
    test_probs = eval_model.get_probabilities(test_tensor).numpy()

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n")
print("\n")
print("\n")

print("FINAL RESULTS")
print(f"\nTraining Accuracy: {train_acc:.4f}") #performance sui membri (overfitting)
print(f"Test Accuracy: {test_acc:.4f}") #performance sui non-membri (generalizzazione)
print(f"Overfitting Gap: {train_acc - test_acc:.4f}") #gap tra train e test, più è grande più il modello ha memorizzato i membri senza generalizzare

print(f"\nConfidence Analysis:")
print(f"  Train max confidence: mean={train_probs.max(axis=1).mean():.4f}, std={train_probs.max(axis=1).std():.4f}") #calcola in media quanto è alta la confidenza massima (probabilità della classe predetta) sui membri, e quanto varia (std)
print(f"  Test max confidence:  mean={test_probs.max(axis=1).mean():.4f}, std={test_probs.max(axis=1).std():.4f}")
print(f"  Confidence gap: {train_probs.max(axis=1).mean() - test_probs.max(axis=1).mean():.4f}")




# 5. SALVATAGGIO

if USE_DP:
    epsilon_final = privacy_engine.get_epsilon(delta)
    model_name = f"target_model_dp_sigma{noise_multiplier}_eps{epsilon_final:.2f}.pth"
else:
    model_name = "target_model_non_dp.pth"

if USE_DP:
    torch.save(model._module.state_dict(), model_name)
else:
    torch.save(model.state_dict(), model_name)

print(f"\nModello salvato come: {model_name}")   

# salvo i dati per l'attacco
# - X_train, y_train: MEMBRI del target (l'attaccante non li conosce ma servono per valutare)
# - X_test, y_test: NON-MEMBRI del target (usati per valutare l'attacco)
# - X_shadow, y_shadow: dati SEPARATI per addestrare gli shadow models
np.savez("data.npz", 
         # Dati del TARGET (ground truth per valutazione)
         X_train=X_train, y_train=y_train,  # Membri
         X_test=X_test, y_test=y_test,      # Non-membri
         # Dati per SHADOW (separati dal target!)
         X_shadow=X_shadow, y_shadow=y_shadow,
         # Indici per tracciabilità
         target_indices=target_indices,
         shadow_indices=shadow_indices)