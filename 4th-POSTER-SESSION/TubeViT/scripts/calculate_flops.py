import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Linear

# Define or load your PyTorch model
# For example, a simple CNN:
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.fc1 = nn.Linear(10 * 6 * 6, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = SimpleCNN()

# Function to calculate FLOPs
def calculate_flops(model):
    total_flops = 0

    for module in model.modules():
        if isinstance(module, (_ConvNd, Linear)):
            output_shape = module.forward(torch.rand(1, *module.input_shape)).shape
            module_flops = torch.prod(torch.tensor(output_shape)).item()

            if isinstance(module, _ConvNd):
                module_flops *= torch.prod(torch.tensor(module.kernel_size)).item() * module.in_channels

            total_flops += module_flops

    return total_flops

# Set the input shape for each layer
model.conv1.input_shape = (3, 32, 32)
model.fc1.input_shape = (10 * 6 * 6,)
model.fc2.input_shape = (50,)

# Calculate FLOPs
flops = calculate_flops(model)
print(f"Total FLOPs: {flops}")
