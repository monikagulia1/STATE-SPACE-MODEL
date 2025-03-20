# STATE-SPACE-MODEL
# State-Space Model (SSM) in PyTorch

This repository contains an implementation of a simple **State-Space Model (SSM)** using PyTorch. The model is designed to process sequential data by maintaining an internal hidden state that evolves over time, and it produces an output at each time step.

## Overview

The SSM consists of:
- **State Transition Matrix (A):** Determines how the hidden state evolves.
- **Input Matrix (B):** Processes the input data and influences the hidden state.
- **Output Matrix (C):** Transforms the hidden state into the model output.
- **Direct Input Matrix (D):** Provides a direct transformation from the input to the output.

The model is trained using the **Mean Squared Error (MSE)** loss function and the **Adam optimizer**.

## How It Works

1. **Forward Pass:**
   - **Hidden State Update:**  
     The next hidden state is computed as:
     ```
     next_state = state @ A + x @ B.T
     ```
     where `x` is the input at the current time step.
   - **Output Computation:**  
     The model output is calculated as:
     ```
     output = next_state @ C.T + x @ D.T
     ```
2. **Training Loop:**
   - The hidden state is initialized as zeros at the beginning of each epoch.
   - For each time step in the sequence, the model updates the hidden state and computes an output.
   - The outputs over the sequence are compared with the target data using the MSE loss.
   - Backpropagation is performed to update the model parameters.

## Issues & Fixes

During training, you might observe extremely large loss values (around \(10^{25}\)). This can be due to several issues:

- **Exploding Values in Parameter Initialization:**
  - **Issue:** Matrices `A`, `B`, `C`, and `D` are initialized using `torch.randn`, which can produce large values.
  - **Fix:** Use a smaller standard deviation (e.g., `torch.randn(..., std=0.01)`) or use methods like `nn.init.xavier_uniform_()` for better initialization.

- **State Growth Over Time (No Stability in A):**
  - **Issue:** The state transition matrix `A` might lead to exponential growth of the hidden state.
  - **Fix:** Apply spectral normalization (e.g., `self.A = nn.Parameter(self.A / self.A.norm())`) to stabilize the matrix.

- **Input Scaling:**
  - **Issue:** High variance in input and target values can result in unstable training.
  - **Fix:** Normalize or standardize your input and target data, e.g., scaling using `.div_(10)`.

- **Hidden State Handling:**
  - **Issue:** The hidden state is never detached from the computation graph during training, leading to gradient explosion.
  - **Fix:** Detach the hidden state inside the loop using `hidden_state = hidden_state.detach()`.

