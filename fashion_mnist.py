import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import csv

torch.manual_seed(42)

#Load dataset
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

train_set, val_set = random_split(
    train_data, [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

#Neural network
class FashionNetwork(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10)
        )

    def forward(self, x):
        return self.network(x)


loss_fn = nn.CrossEntropyLoss()

#Evaluate loss and accuracy
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in loader:
            predictions = model(X)
            total_loss += loss_fn(predictions, y).item() * len(y)
            correct += (predictions.argmax(1) == y).sum().item()

    return total_loss / len(loader.dataset), correct / len(loader.dataset)


#Training
def train_model(hidden, name):
    torch.manual_seed(42)
    model = FashionNetwork(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(5):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            predictions = model(X)
            loss = loss_fn(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_set)
        val_loss, val_acc = evaluate(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"{name} Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Val Loss={val_loss:.4f}, "
            f"Val Accuracy={val_acc:.2%}"
        )

    return model, train_losses, val_losses, val_acc


#Original model
print("Original Model")
model1, train1, val1, acc1 = train_model(128, "Original")

#Experiment: increase hidden neurons
print("\nModified Model")
model2, train2, val2, acc2 = train_model(256, "Modified")

#Choose model using validation accuracy
best_model = model2 if acc2 > acc1 else model1

#Final test evaluation
test_loss, test_acc = evaluate(best_model, test_loader)

print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.2%}")

#Save results
with open("results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Epoch", "Train Loss", "Validation Loss"])

    for name, training, validation in [
        ("Original", train1, val1),
        ("Modified", train2, val2)
    ]:
        for epoch in range(5):
            writer.writerow([
                name, epoch + 1,
                training[epoch], validation[epoch]
            ])

#Graph
plt.plot(train1, label="Original Train")
plt.plot(val1, label="Original Validation")
plt.plot(train2, label="Modified Train")
plt.plot(val2, label="Modified Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fashion-MNIST Training Results")
plt.legend()
plt.savefig("training_results.png")
plt.show()