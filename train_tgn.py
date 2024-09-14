import pandas as pd
import numpy as np
import torch
from torch_geometric_temporal.nn.recurrent import TGN
from torch_geometric_temporal.dataset import AbstractTemporalDataset
from torch_geometric_temporal.signal import TemporalSignal
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
import matplotlib.pyplot as plt

# Define actions and their applicable resource types
ACTIONS = {
    'list_all_buckets_of_project': 'project',
    'list_all_instances_in_project': 'project',
    'list_all_objects_in_bucket': 'bucket',
    'read_object': 'data_object',
    'log_on_to_instance': 'instance',
    'assume_service_account': 'service_account',
    'assign_ssh_key_to_instance': 'instance'
}

# Load the synthetic logs
df = pd.read_csv('synthetic_logs.csv', parse_dates=['timestamp'])

# Initialize label encoders
le_principal = LabelEncoder()
le_action = LabelEncoder()
le_resource = LabelEncoder()

# Fit and transform the categorical columns
df['principal_id'] = le_principal.fit_transform(df['principal'])
df['action_id'] = le_action.fit_transform(df['action'])
df['resource_id'] = le_resource.fit_transform(df['resource'])

# Sort the DataFrame by timestamp
df = df.sort_values('timestamp').reset_index(drop=True)

# Combine all unique principals and resources
unique_principals = df['principal_id'].unique()
unique_resources = df['resource_id'].unique()

# Create a mapping for node IDs
node_ids = np.concatenate([unique_principals, unique_resources])
node_ids = np.unique(node_ids)
num_nodes = len(node_ids)

# Example: Create simple features based on node type
# Assuming principals are either users or service accounts
users = set(le_principal.transform([p for p in le_principal.classes_ if p.startswith('user_')]))
service_accounts = set(le_principal.transform([p for p in le_principal.classes_ if p.startswith('service_account_')]))

# Initialize feature matrix
node_features = np.zeros((num_nodes, 2))  # [is_user, is_service_account]

for node in node_ids:
    if node in users:
        node_features[node, 0] = 1  # is_user
    elif node in service_accounts:
        node_features[node, 1] = 1  # is_service_account

# Convert to torch tensor
node_features = torch.tensor(node_features, dtype=torch.float)

# Assign node offsets
principal_offset = 0
resource_offset = len(unique_principals)

# Map principals and resources to node indices
df['source'] = df['principal_id']
df['target'] = df['resource_id'] + resource_offset

# Extract edges and timestamps
edge_index = df[['source', 'target']].values.T  # Shape [2, num_edges]
timestamps = df['timestamp'].astype(int) // 10**9  # Convert to UNIX timestamp

# Encode actions as edge features
action_features = df['action_id'].values

# Split data into training, validation, and test sets
train_size = 0.7
val_size = 0.15
test_size = 0.15

num_total = len(df)
train_end = int(train_size * num_total)
val_end = train_end + int(val_size * num_total)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# Function to create interaction tuples
def create_interactions(df, principal_offset=0, resource_offset=resource_offset):
    interactions = list(zip(
        df['source'].values,
        df['target'].values,
        df['timestamp'].values,
        df['action_id'].values
    ))
    return interactions

train_interactions = create_interactions(train_df)
val_interactions = create_interactions(val_df)
test_interactions = create_interactions(test_df)

# Define a custom temporal dataset
class CustomTemporalDataset(AbstractTemporalDataset):
    def __init__(self, train_interactions, val_interactions, test_interactions, num_nodes, node_features):
        self.train_interactions = train_interactions
        self.val_interactions = val_interactions
        self.test_interactions = test_interactions
        self.num_nodes = num_nodes
        self.node_features = node_features

    def __len__(self):
        return 1  # Single sequence

    def get(self, idx):
        # Combine all interactions
        interactions = self.train_interactions + self.val_interactions + self.test_interactions
        # Sort by timestamp
        interactions = sorted(interactions, key=lambda x: x[2])
        source = torch.tensor([i[0] for i in interactions], dtype=torch.long)
        target = torch.tensor([i[1] for i in interactions], dtype=torch.long)
        timestamp = torch.tensor([i[2] for i in interactions], dtype=torch.float)
        action = torch.tensor([i[3] for i in interactions], dtype=torch.long)

        # Define edge features (actions)
        edge_features = torch.nn.functional.one_hot(action, num_classes=len(le_action.classes_)).float()

        return TemporalSignal(
            edge_index=torch.stack([source, target], dim=0),
            edge_attr=edge_features,
            edge_time=timestamp,
            y=None  # Define targets if needed
        )

# Instantiate the dataset
dataset = CustomTemporalDataset(train_interactions, val_interactions, test_interactions, num_nodes, node_features)

# Define the TGN model
class TGNModel(torch.nn.Module):
    def __init__(self, node_features, hidden_channels, out_channels, num_actions):
        super(TGNModel, self).__init__()
        self.tgn = TGN(
            node_feature_dim=node_features.shape[1],
            memory_dimension=hidden_channels,
            message_dimension=hidden_channels,
            hidden_dimension=hidden_channels,
            output_dimension=out_channels,
            time_enc_dimension=32,
            aggregator_type='lstm',  # 'mean', 'lstm', 'pool', 'sum'
            n_heads=1
        )
        self.classifier = torch.nn.Linear(out_channels, num_actions)

    def forward(self, x, edge_index, edge_attr, edge_time):
        out = self.tgn(x, edge_index, edge_attr, edge_time)
        # Predict action type based on target node embedding
        out = self.classifier(out[edge_index[1]])
        return out

# Instantiate the model
hidden_channels = 64
out_channels = 64
num_actions = len(le_action.classes_)
model = TGNModel(node_features=node_features, hidden_channels=hidden_channels, out_channels=out_channels, num_actions=num_actions)

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define training and evaluation functions
def train(model, dataset, optimizer, criterion, device='cpu'):
    model.train()
    total_loss = 0
    for data in dataset:
        # Move data to device
        data = data.to(device)
        node_features = dataset.node_features.to(device)

        optimizer.zero_grad()
        # Forward pass
        out = model(node_features, data.edge_index, data.edge_attr, data.edge_time)
        # Define targets (action types)
        targets = data.edge_attr.argmax(dim=1)
        # Compute loss
        loss = criterion(out, targets)
        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataset)

def evaluate(model, dataset, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            node_features = dataset.node_features.to(device)
            out = model(node_features, data.edge_index, data.edge_attr, data.edge_time)
            preds = out.argmax(dim=1).cpu().numpy()
            targets = data.edge_attr.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    return accuracy, precision, recall, f1

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
dataset.node_features = dataset.node_features.to(device)

# Training parameters
num_epochs = 20

# Lists to store metrics
train_losses = []
val_accuracies = []
val_precisions = []
val_recalls = []
val_f1s = []

for epoch in range(1, num_epochs + 1):
    loss = train(model, [dataset.get(0)], optimizer, criterion, device)
    train_losses.append(loss)
    
    # Evaluate on validation set
    val_acc, val_prec, val_rec, val_f1 = evaluate(model, [dataset.get(0)], device)
    val_accuracies.append(val_acc)
    val_precisions.append(val_prec)
    val_recalls.append(val_rec)
    val_f1s.append(val_f1)
    
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

# Plot training metrics
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Accuracy')
plt.plot(range(1, num_epochs + 1), val_f1s, label='Val F1 Score')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Validation Metrics')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
test_acc, test_prec, test_rec, test_f1 = evaluate(model, [dataset.get(0)], device)
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Test Precision: {test_prec:.4f}')
print(f'Test Recall: {test_rec:.4f}')
print(f'Test F1 Score: {test_f1:.4f}')
