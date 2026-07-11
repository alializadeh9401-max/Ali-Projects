import torch
from torch.optim import Adam


# --------------------------------------------------
# Data
# --------------------------------------------------

inputs = torch.tensor([
    [0., 0.5, 0.25, 1.],
    [1., 0.5, 0.25, 1.]
])

labels = torch.tensor([0., 1.])


# --------------------------------------------------
# Packed LSTM parameters
# --------------------------------------------------

# hidden_size = 1
# input_size = 1
# gates = 4
#
# gate order:
# 0 = forget gate
# 1 = input gate
# 2 = candidate memory
# 3 = output gate

W_h = torch.randn(1, 4, requires_grad=True)  # short_memory weights
W_x = torch.randn(1, 4, requires_grad=True)  # input weights
b = torch.zeros(4, requires_grad=True)       # gate biases

params = [W_h, W_x, b]

optimizer = Adam(params, lr=0.1)


# --------------------------------------------------
# LSTM unit using packed tensor parameters
# --------------------------------------------------

def lstm_unit(input_value, long_memory, short_memory):

    # Compute all four gate pre-activations at once
    gate_values = short_memory * W_h + input_value * W_x + b

    # Split gates
    forget_gate = torch.sigmoid(gate_values[0, 0])
    input_gate = torch.sigmoid(gate_values[0, 1])
    candidate_memory = torch.tanh(gate_values[0, 2])
    output_gate = torch.sigmoid(gate_values[0, 3])

    # Update cell state
    updated_long_memory = (
        forget_gate * long_memory
        + input_gate * candidate_memory
    )

    # Update hidden state
    updated_short_memory = output_gate * torch.tanh(updated_long_memory)

    return updated_long_memory, updated_short_memory


# --------------------------------------------------
# Forward pass through sequence
# --------------------------------------------------

def forward(input_sequence):
    long_memory = torch.tensor(0.0)
    short_memory = torch.tensor(0.0)

    for input_value in input_sequence:
        long_memory, short_memory = lstm_unit(
            input_value,
            long_memory,
            short_memory
        )

    return short_memory


# --------------------------------------------------
# Training loop
# --------------------------------------------------

epochs = 3001

for epoch in range(epochs):

    total_loss = 0.0

    for input_i, label_i in zip(inputs, labels):

        output_i = forward(input_i)

        loss = (output_i - label_i) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 300 == 0:
        print(f"Epoch {epoch}, Loss = {total_loss:.6f}")


# --------------------------------------------------
# Test predictions
# --------------------------------------------------

print("\nNow compare obs with pred...")

print(
    "Company A: Observed=0, Predicted =",
    forward(torch.tensor([0., 0.5, 0.25, 1.0])).detach()
)

print(
    "Company B: Observed=1, Predicted =",
    forward(torch.tensor([1., 0.5, 0.25, 1.0])).detach()
)