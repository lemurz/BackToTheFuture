import pandas as pd
import numpy as np

import random
import torch
from mlp import MLP

from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

SEED = 42
TEST_SIZE = 0.20
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001

# load file

FILE_PATH = "file_path"

df = pd.read_csv(FILE_PATH)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", device)

# seperate features & label

X = df.drop(columns=["Label"]).copy()
y = df["Label"].copy()

print("\nFeature matrix: ", X.shape)
print("Labels: ", y.shape)

non_numeric_columns = X.select_dtypes(
    exclude = np.number
).columns.tolist()

print("\nNon-numeric columns: ")
print(non_numeric_columns)

if len(non_numeric_columns) > 0:
  raise ValueError(
      f"Found non-numeric features : {non_numeric_columns}"
  )

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

invalid_rows = X.isnull().any(axis=1)

print(
    "\nRows containing NaN/Inf:",
    invalid_rows.sum()
)

X = X.loc[~invalid_rows].reset_index(drop=True)
y = y.loc[~invalid_rows].reset_index(drop=True)

print("Rows remaining: ", len(X))


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

NUM_CLASSES = len(label_encoder.classes_)

print("\nNumber of encoded classes: ", NUM_CLASSES)

print("\nLabel mapping: ")

for number, name in enumerate(label_encoder.classes_):
  print(f"{number:2d} -> {name}")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_encoded
)

print("\nTrain samples: ", len(X_train))
print("Test samples: ", len(X_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype = torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype = torch.long
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

INPUT_SIZE = X_train_tensor.shape[1]

model = MLP(
    input_size=INPUT_SIZE,
    num_classes = NUM_CLASSES
).to(device)

print("\nModel: ")
print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


print("\n==================================")
print("TRAINING")
print("===================================")

for epoch in range(EPOCHS):

  model.train()

  total_loss = 0
  correct = 0
  total = 0

  for X_batch, y_batch in train_loader:
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    optimizer.zero_grad()

    outputs = model(X_batch)

    loss = criterion(outputs, y_batch)

    loss.backward()

    optimizer.step()

    total_loss += loss.item()

    predicted = torch.argmax(
        outputs,
        dim=1
    )

    correct += (
        predicted == y_batch
    ).sum().item()

    total += y_batch.size(0)

  average_loss = (
      total_loss / len(train_loader)
  )

  train_accuracy = (
      correct / total
  )

  print(
      f"Epoch {epoch + 1:02d}/{EPOCHS} |"
      f"Loss; {average_loss: .4f} | "
      f"Train Accuracy: {train_accuracy: .4f}"
  )

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

  for X_batch, y_batch in test_loader:

    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    outputs = model(X_batch)

    predicted = torch.argmax(
        outputs,
        dim=1
    )

    all_predictions.extend(
        predicted.cpu().numpy()
    )

    all_targets.extend(
        y_batch.cpu().numpy()
    )

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

macro_precision = precision_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0
)

print("\n==============================")
print("MERGED_001 CENTRALIZED RESULTS")
print("==============================")

print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")


print("\nClassification Report:\n")

print(
    classification_report(
        all_targets,
        all_predictions,
        labels=np.arange(NUM_CLASSES),
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


print("\nConfusion Matrix:\n")

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=np.arange(NUM_CLASSES)
)

print(cm)
