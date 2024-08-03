import torch
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Create the original model
original_model = SimpleModel()

# Compile the model
compiled_model = torch.compile(original_model)

# Set up training
optimizer = optim.SGD(compiled_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Initial weights
print("Initial weights:")
print(original_model.linear.weight.data)

# Train for a few steps
for _ in range(5):
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    outputs = compiled_model(inputs)
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Check weights after training
print("\nWeights after training compiled model:")
print(original_model.linear.weight.data)

# Verify that the weights are the same object in memory
print("\nAre the weights the same object?")
print(id(original_model.linear.weight) == id(compiled_model.linear.weight))

print(torch.equal(original_model(inputs), compiled_model(inputs)))