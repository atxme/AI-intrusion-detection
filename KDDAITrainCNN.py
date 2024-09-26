import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# 1. Vérification de la disponibilité du GPU
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# 2. Chargement des données avec pandas 
train = pd.read_csv('dataSet/KDDTrain_final.csv')

# 3. Prétraitement des données
# Encodage de la variable cible
le = LabelEncoder()
train['class'] = le.fit_transform(train['class'])

# Variables explicatives et cible
X = train.drop(['class'], axis=1).values  # 41 features
y = train['class'].values

# Standardisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Mélange des données et stratification pour assurer une répartition équilibrée des classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# Reshape des données pour CNN (batch, channels, height, width)
X_train = X_train.reshape(-1, 1, 41, 1)
X_test = X_test.reshape(-1, 1, 41, 1)

# Conversion des données en tenseurs PyTorch
X_train_tensor = th.tensor(X_train, dtype=th.float32).to(device)
y_train_tensor = th.tensor(y_train, dtype=th.long).to(device)
X_test_tensor = th.tensor(X_test, dtype=th.float32).to(device)
y_test_tensor = th.tensor(y_test, dtype=th.long).to(device)

# Création d'un DataLoader pour la gestion des batchs
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Afficher les noms de colonnes pour vérifier leur correspondance
print(train.columns)

# Définition du modèle CNN avec AdaptiveMaxPooling
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Couches convolutionnelles
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 1), stride=1, padding=(1, 0))  # 3x1 kernel
        self.pool = nn.AdaptiveMaxPool2d((20, 1))  # Adaptive pooling pour garantir une taille de sortie correcte

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool2 = nn.AdaptiveMaxPool2d((10, 1))  # Deuxième couche de pooling adaptatif

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 10 * 1, 128)  # l1=128
        self.fc2 = nn.Linear(128, 128)          # l2=128
        self.fc3 = nn.Linear(128, len(np.unique(y_train)))
        self.dropout = nn.Dropout(0.383)        # Dropout de 0.383
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Passage à travers les couches
        x = self.pool(self.activation(self.conv1(x)))  # Conv1 + Pool1
        x = self.pool2(self.activation(self.conv2(x)))  # Conv2 + Pool2
        x = x.view(-1, 32 * 10 * 1)  # Flatten
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialisation du modèle
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.000441295)

# Simulation d'entraînement
epochs = 40
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

for epoch in range(epochs):

    model.train()  # Mode entraînement
    
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = th.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    # Mode évaluation
    model.eval()

    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with th.no_grad():
        outputs = model(X_test_tensor)
        loss = criterion(outputs, y_test_tensor)

        test_loss += loss.item()

        _, predicted = th.max(outputs, 1)
        correct_test += (predicted == y_test_tensor).sum().item()
        total_test += y_test_tensor.size(0)

    test_losses.append(test_loss)
    test_accuracies.append(correct_test / total_test)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}")

# Évaluation du modèle 
model.eval()
with th.no_grad():
    test_outputs = model(X_test_tensor)
    _, test_pred = th.max(test_outputs, 1)

    correct = (test_pred == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = correct / total

    print(f'Final Test Accuracy: {accuracy:.4f}')

# Graphique des résultats
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Graphique pour la loss
ax[0].plot(train_losses, label='Train')
ax[0].plot(test_losses, label='Test')
ax[0].set_title('Loss over epochs')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].grid(True)

# Graphique pour l'accuracy
ax[1].plot(train_accuracies, label='Train')
ax[1].plot(test_accuracies, label='Test')
ax[1].set_title('Accuracy over epochs')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# Matrice de confusion
cm = confusion_matrix(y_test_tensor.cpu(), test_pred.cpu())
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 7. Sauvegarde du modèle
th.save(model.state_dict(), 'modelCNN.pth')