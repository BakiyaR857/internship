import torch
from model.rnn_model import RNNModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# ✅ Load the correct data
data = torch.load("model/data.pth", weights_only=False)

X_train = torch.tensor(data["X_train"], dtype=torch.long)  # use long for embedding
y_train = torch.tensor(data["y_train"], dtype=torch.long)
all_words = data["all_words"]
tags = data["tags"]

input_size = len(all_words)
hidden_size = 128
output_size = len(tags)
batch_size = 8
learning_rate = 0.001
num_epochs = 1000

# ✅ Dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# ✅ Model
model = RNNModel(input_size, hidden_size, output_size)

# ✅ Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training
for epoch in range(num_epochs):
    for words, labels in train_loader:
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# ✅ Save the model
model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}
torch.save(model_data, "model/rnn_model.pth")
print("✅ Training complete. Model saved to model/rnn_model.pth")
