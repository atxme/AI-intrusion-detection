import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from scipy.io import arff

# 1. Vérification de la disponibilité du GPU
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# 2. Chargement des données ARFF avec pandas
def load_arff(file_path):
    data = arff.loadarff(file_path)
    df = pd.DataFrame(data[0])
    
    return df

train = pd.read_csv('dataSet/KDDTrain_final.csv')

# 3. Prétraitement des données
# Encodage de la variable cible
le = LabelEncoder()
train['class'] = le.fit_transform(train['class'])

# Variables explicatives et cible
X = train.drop(['class'], axis=1).values
y = train['class'].values

# Standardisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Mélange des données et stratification pour assurer une répartition équilibrée des classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

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

# Définition du modèle MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # Définition des couches
        self.hidden1 = nn.Linear(X_train.shape[1], 128)  # nombre de features -> 128 neurones cachés
        self.hidden2 = nn.Linear(128, 64)  # 128 -> 64 neurones cachés
        self.hidden3 = nn.Linear(64, 32)   # 64 -> 32 neurones cachés
        self.output = nn.Linear(32, len(np.unique(y_train)))    # 32 -> nombre de classes uniques dans 'class'
        
        self.dropout = nn.Dropout(0.2)  # Dropout de 20% pour régularisation
        # Activation LeakyReLU pour les couches cachées
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        # Passage à travers les couches avec activations et Dropout
        x = self.activation(self.hidden1(x))
        x = self.dropout(x)
        
        x = self.activation(self.hidden2(x))
        x = self.dropout(x)
        
        x = self.activation(self.hidden3(x))
        x = self.dropout(x)
        
        x = self.output(x)
        
        return x
    
# 4. Initialisation du modèle, de la fonction de coût et de l'optimiseur
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Simulation d'entraînement
epochs = 40

train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    model.train()  # Mode entraînement
    
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        _, predicted = th.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
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
        
        test_loss = loss.item()
        
        _, predicted = th.max(outputs, 1)
        correct_test = (predicted == y_test_tensor).sum().item()
        total_test = y_test_tensor.size(0)
    
    test_losses.append(test_loss)
    test_accuracies.append(correct_test / total_test)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')


# 6. Évaluation du modèle
model.eval()
with th.no_grad():
    # Prédictions sur l'ensemble de test
    outputs = model(X_test_tensor)
    _, predicted = th.max(outputs, 1)
    
    # Calcul de l'accuracy
    correct = (predicted == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = correct / total
    
    print(f'Final Test Accuracy: {accuracy:.4f}')

# Graphique des résultats
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Graphique pour Loss
ax[0].plot(train_losses, label='Train Loss')
ax[0].plot(test_losses, label='Test Loss')
ax[0].set_title('Training vs Test Loss over Epochs')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].grid(True)

# Graphique pour Accuracy
ax[1].plot(train_accuracies, label='Training Accuracy', marker='o')
ax[1].plot(test_accuracies, label='Test Accuracy', marker='x')
ax[1].set_title('Training vs Test Accuracy over Epochs')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# 7. Sauvegarde du modèle
th.save(model.state_dict(), 'model.pth')
print('Model saved successfully')

#print f1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
print(f'F1 Score: {f1:.4f}')


