import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import ray
from ray import tune
from ray.train import report
from ray.tune.schedulers import ASHAScheduler

# 1. Vérification de la disponibilité du GPU
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# 2. Chargement des données avec pandas 
train = pd.read_csv('dataSet/KDDTrain_final.csv')

# 3. Prétraitement des données
le = LabelEncoder()
train['class'] = le.fit_transform(train['class'])

X = train.drop(['class'], axis=1).values  # 41 features
y = train['class'].values

# Standardisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

X_train = X_train.reshape(-1, 1, 41, 1)
X_test = X_test.reshape(-1, 1, 41, 1)

X_train_tensor = th.tensor(X_train, dtype=th.float32).to(device)
y_train_tensor = th.tensor(y_train, dtype=th.long).to(device)
X_test_tensor = th.tensor(X_test, dtype=th.float32).to(device)
y_test_tensor = th.tensor(y_test, dtype=th.long).to(device)

train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 4. Définition du modèle CNN modifié pour Ray Tune
class CNN(nn.Module):
    def __init__(self, l1=128, l2=64, dropout=0.2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool = nn.AdaptiveMaxPool2d((20, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.pool2 = nn.AdaptiveMaxPool2d((10, 1))

        self.fc1 = nn.Linear(32 * 10 * 1, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, len(np.unique(y_train)))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        x = x.view(-1, 32 * 10 * 1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.fc3(x)
        return x

# 5. Fonction d'entraînement pour Ray Train
def train_cnn(config):
    model = CNN(l1=config["l1"], l2=config["l2"], dropout=config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(10):  
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = th.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Évaluation
        model.eval()
        correct_test = 0
        total_test = 0
        with th.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = th.max(outputs, 1)
            correct_test += (predicted == y_test_tensor).sum().item()
            total_test += y_test_tensor.size(0)

        accuracy = correct_test / total_test
        average_loss = train_loss / len(train_loader)

        # Utilisez ray.train.report pour rapporter les métriques
        report(dict(accuracy=accuracy, loss=average_loss))


# 6. Définition de l'espace de recherche pour les hyperparamètres
search_space = {
    "l1": tune.choice([64, 128, 256]),
    "l2": tune.choice([32, 64, 128]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "dropout": tune.uniform(0.2, 0.5)
}

# Configuration du scheduler ASHA
asha_scheduler = ASHAScheduler(
    metric="accuracy",
    mode="max",
    max_t=40,
    grace_period=10,  # Augmenter la période de grâce
    reduction_factor=2
)

# Exécution des essais avec Tune
analysis = tune.run(
    train_cnn,
    resources_per_trial={"cpu": 2, "gpu": 0.20}, 
    config=search_space,
    num_samples=8, 
    scheduler=asha_scheduler,  # Scheduler déjà configuré avec metric et mode
    progress_reporter=tune.CLIReporter(
        metric_columns=["accuracy", "loss", "training_iteration"]
    )
)

# Récupérer la meilleure configuration avec get_best_config
best_config = analysis.get_best_config(metric="accuracy", mode="max")
print("Best config: ", best_config)

# Récupérer les résultats sous forme de DataFrame
df = analysis.results_df
print(df)