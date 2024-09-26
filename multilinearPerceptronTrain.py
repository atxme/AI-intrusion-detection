import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Vérification de la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Chargement des données
df = pd.read_csv('dataSet/newDataSet.csv')

# 3. Prétraitement des données
X = df.drop('Attack Type', axis=1).values  # Variables explicatives
y = pd.get_dummies(df['Attack Type']).values  # Variable cible (one-hot encoded)

# Standardisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Mélange des données et stratification pour assurer une répartition équilibrée des classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Conversion des données en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Création d'un DataLoader pour la gestion des batchs
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Définition du modèle MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # Définition des couches
        self.hidden1 = nn.Linear(12, 32)  # 12 entrées -> 64 neurones cachés
        self.hidden2 = nn.Linear(32, 16) # 200 -> 100 neurones cachés
        self.hidden3 = nn.Linear(16, 8)  # 100 -> 50 neurones cachés
        self.output = nn.Linear(8, 3)     # 50 -> 3 neurones de sortie (Malware, DDoS, Intrusion)
        

        self.dropout = nn.Dropout(0.2)  # Dropout de 20% pour régularisation
        # Activation LeakyReLU pour les couches cachées
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Passage à travers les couches avec activations et Dropout
        x = self.activation(self.hidden1(x))

        x = self.activation(self.hidden2(x))

        x = self.activation(self.hidden3(x))
        x = self.dropout(x)
        x = self.output(x)  # Sortie brute (logits) sans Softmax car CrossEntropyLoss s'en charge
        return x

# Initialisation du modèle, de l'optimiseur et du critère de perte
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()  # Utilise directement CrossEntropyLoss sans Softmax dans la dernière couche
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Simulation d'entraînement (assure-toi que X_train_tensor et y_train sont bien des tenseurs)
epochs = 100
train_accuracies = []
test_accuracies = []
train_log_losses = []
test_log_losses = []

for epoch in range(epochs):
    model.train()  # Mode entraînement
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Réinitialisation du gradient

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(dim=1))  # Pas de Softmax dans les logits

        # Backward pass et optimisation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Évaluation sur l'ensemble d'entraînement et de test
    model.eval()  # Mode évaluation
    with torch.no_grad():
        # Prédictions pour l'ensemble d'entraînement
        train_outputs = model(X_train_tensor)
        train_pred = torch.argmax(train_outputs, dim=1)
        train_labels = torch.argmax(y_train_tensor, dim=1)
        train_accuracy = (train_pred == train_labels).float().mean().item()

        # Calcul de la log loss directement avec PyTorch
        train_log_loss = criterion(train_outputs, train_labels).item()

        # Prédictions pour l'ensemble de test
        test_outputs = model(X_test_tensor)
        test_pred = torch.argmax(test_outputs, dim=1)
        test_labels = torch.argmax(y_test_tensor, dim=1)
        test_accuracy = (test_pred == test_labels).float().mean().item()

        # Calcul de la log loss directement avec PyTorch
        test_log_loss = criterion(test_outputs, test_labels).item()

        # Sauvegarde des métriques
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        train_log_losses.append(train_log_loss)
        test_log_losses.append(test_log_loss)

    # Affichage des résultats par époque
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")

# 8. Graphique des résultats
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Graphique pour Accuracy
ax[0].plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
ax[0].plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy', marker='x')
ax[0].set_title('Training vs Test Accuracy over Epochs')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Accuracy')
ax[0].legend()
ax[0].grid(True)

# Graphique pour Log Loss
ax[1].plot(range(1, epochs + 1), train_log_losses, label='Training Log Loss', marker='o')
ax[1].plot(range(1, epochs + 1), test_log_losses, label='Test Log Loss', marker='x')
ax[1].set_title('Training vs Test Log Loss over Epochs')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Log Loss')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
