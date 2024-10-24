import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_leukemia_dataset():
    np.random.seed(42)
    data_size = 1000
    num_features = 20
    features = np.random.randn(data_size, num_features)
    labels = np.random.randint(0, 2, size=(data_size,))
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
    return train_x, test_x, train_y, test_y

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def create_graph_data(train_x, train_y):
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(x=train_x, edge_index=edge_index, y=train_y)
    return data

def train_gnn(train_data, test_x, test_y):
    input_dim = train_data.x.shape[1]
    hidden_dim = 64
    output_dim = 2
    model = GNNModel(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_data.y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_data = Data(x=test_x, edge_index=train_data.edge_index, y=test_y)
        test_output = model(test_data)
        pred = test_output.argmax(dim=1)
        accuracy = accuracy_score(test_y.detach().numpy(), pred.detach().numpy())
        f1 = f1_score(test_y.detach().numpy(), pred.detach().numpy(), average='weighted')
        auc = roc_auc_score(test_y.detach().numpy(), pred.detach().numpy())
    return model, accuracy, f1, auc

def train_random_forest(train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    accuracy = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='weighted')
    auc = roc_auc_score(test_y, pred)
    return accuracy, f1, auc

def train_svm(train_x, train_y, test_x, test_y):
    clf = SVC(probability=True)
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    accuracy = accuracy_score(test_y, pred)
    f1 = f1_score(test_y, pred, average='weighted')
    auc = roc_auc_score(test_y, pred)
    return accuracy, f1, auc

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_true, y_score, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(y_true, y_score))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

def main():
    train_x, test_x, train_y, test_y = load_leukemia_dataset()
    train_data = create_graph_data(train_x, train_y)
    gnn_model, gnn_acc, gnn_f1, gnn_auc = train_gnn(train_data, test_x, test_y)
    rf_acc, rf_f1, rf_auc = train_random_forest(train_x.numpy(), train_y.numpy(), test_x.numpy(), test_y.numpy())
    svm_acc, svm_f1, svm_auc = train_svm(train_x.numpy(), train_y.numpy(), test_x.numpy(), test_y.numpy())
    print(f"GNN - Accuracy: {gnn_acc}, F1 Score: {gnn_f1}, AUC: {gnn_auc}")
    print(f"Random Forest - Accuracy: {rf_acc}, F1 Score: {rf_f1}, AUC: {rf_auc}")
    print(f"SVM - Accuracy: {svm_acc}, F1 Score: {svm_f1}, AUC: {svm_auc}")
    test_output = gnn_model(Data(x=test_x, edge_index=train_data.edge_index, y=test_y))
    plot_confusion_matrix(test_y.detach().numpy(), test_output.argmax(dim=1).detach().numpy(), "GNN Confusion Matrix")
    plot_roc_curve(test_y.detach().numpy(), test_output.softmax(dim=1)[:, 1].detach().numpy(), "GNN ROC Curve")

if __name__ == "__main__":
    main()
