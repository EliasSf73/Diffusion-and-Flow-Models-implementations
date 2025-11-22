Diffusion & Flow Models: From Scratch

A collection of minimal, clean, and mathematically grounded implementations of modern generative models. This repository explores the evolution from stochastic Diffusion Models to deterministic Flow Matching and Rectified Flow, using 2D toy datasets (Swiss Roll) to visualize the learned vector fields and probability paths.

📂 Repository Structure

Notebook

Description

Key Concepts

DDPM_DDIM.ipynb

The Foundations

Denoising Diffusion Probabilistic Models (DDPM), Deterministic Sampling (DDIM).

DPM_Solver.ipynb

Fast ODE Solvers

Higher-order solvers for Diffusion ODEs to reduce sampling steps.

flow_matching.ipynb

State-of-the-Art

Conditional Flow Matching (CFM), Optimal Transport Paths, and Rectified Flow for 1-step generation.

🧠 Theoretical Background

1. Denoising Diffusion (DDPM & DDIM)

Traditional diffusion models define a forward process that gradually adds Gaussian noise to data $x_0$ over time $t \in [0, T]$:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$The generative process reverses this by learning to predict the noise $\epsilon$ added at each step. The training objective is simply a weighted Mean Squared Error (MSE) between the actual noise and the predicted noise:

$$\mathcal{L}*{\text{simple}} = \mathbb{E}*{t, x\_0, \epsilon} \left[ | \epsilon - \epsilon\_\theta(x\_t, t) |^2 \right]
$$While DDPM models the process as a Stochastic Differential Equation (SDE), **DDIM** and probability flow formulations convert this into an Ordinary Differential Equation (ODE), allowing for deterministic sampling.

### 2\. Flow Matching (FM)

Flow Matching generalizes the Probability Flow ODE. Instead of simulating a diffusion process and deriving the vector field, we **directly define** a probability path and regress the vector field.

We use the **Optimal Transport (OT) Conditional Flow**, which defines a straight line path between noise $x_0$ and data $x_1$:$$

x_t = (1 - t)x_0 + t x_1
$$The target vector field (velocity) for this path is constant:

$$u_t(x | x_1, x_0) = x_1 - x_0
$$Objective Function (Conditional Flow Matching):
We train a neural network $v_\theta(x, t)$ to match this target velocity:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$This results in a model that learns "straight" paths between the noise distribution and the data distribution, which are much easier to integrate than the curved paths of diffusion models.

### 3\. Rectified Flow (Reflow)

Even with Flow Matching, the learned vector field $v_\theta$ may still be slightly curved due to the coupling between arbitrary noise and data points. **Rectified Flow** solves this by "straightening" the paths iteratively.

**The Reflow Procedure:**

1.  **Generate Pairs:** Use the pre-trained Flow model to generate data $z_1$ starting from noise $z_0$. This creates a coupling $(z_0, z_1)$ that lies on the same flow trajectory.
2.  **Retrain:** Train a new model on these specific pairs $(z_0, z_1)$.
3.  **Result:** The model learns the direct straight line between $z_0$ and $z_1$.

$$\text{Trajectory } (z\_0 \to z\_1) \implies \text{Perfectly Straight Line}
$$This effectively reduces the discretization error to near zero, enabling high-quality generation in a **single Euler step ($N=1$)**.

-----

## 📊 Results

### Flow Matching vs. Rectified Flow

| Method | Steps | Chamfer Distance (Lower is Better) | Quality |
| :--- | :--- | :--- | :--- |
| **Standard FM** | 100 | \~24.1 | ✅ High Fidelity |
| **Standard FM** | 10 | \~42.8 | ❌ Distorted / Broken |
| **Rectified Flow** | **10** | **\~29.1** | ✅ **High Fidelity (Fast)** |
| **Rectified Flow** | **1** | **\< 35.0** | 🚀 **Real-time (1-Step)** |

*(Visualizations of the Swiss Roll manifold transformation are available in the notebooks.)*

-----

## 🚀 Usage

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/EliasSf73/Diffusion-and-Flow-Models-implementations.git](https://github.com/EliasSf73/Diffusion-and-Flow-Models-implementations.git)
    cd Diffusion-and-Flow-Models-implementations
    ```

2.  **Install dependencies:**

    ```bash
    pip install torch numpy matplotlib tqdm
    ```

3.  **Run the notebooks:**
    Start with `flow_matching.ipynb` to see the latest Rectified Flow implementation.

    ```bash
    jupyter notebook
    ```

-----

## 📚 References

1.  **Denoising Diffusion Probabilistic Models** (Ho et al., 2020)
2.  **Denoising Diffusion Implicit Models** (Song et al., 2020)
3.  **Flow Matching for Generative Modeling** (Lipman et al., 2023)
4.  **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** (Liu et al., 2023)

-----

*Author: EliasSf73*$$
