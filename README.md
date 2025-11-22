🌫️ Diffusion & Flow Models — Minimal, Clean Implementations

This repository contains a set of mathematically transparent mini-projects exploring diffusion models, ODE samplers, and flow-matching. The goal is to understand generative modeling from first principles, with clear visualizations on simple 2D datasets.

🧩 Project Topics

🌪️ DDPM (Denoising Diffusion Probabilistic Models)
Forward diffusion, reverse denoising, and noise-matching objective.

🔁 DDIM (Deterministic Sampling)
ODE interpretation of sampling, exact reconstruction of x₀.

⚡ DPM-Solver (1st & 2nd Order)
Exponential-integrator ODE solvers in λ-space.

🧷 Flow Matching (FM)
Neural ODE generative modeling, velocity fields.

🌀 Toy Dataset Transformations
Spiral → Gaussian, checkerboard → noise, and reverse sampling demos.

📂 Repository Structure
Notebooks
Notebook	Description
DDPM_DDIM.ipynb	DDPM training + DDIM sampling
DPM_Solver.ipynb	DPM-Solver (α, σ, λ schedules; 1st/2nd order solvers)
flow_matching.ipynb	Gaussian Flow Matching + velocity networks
🔍 Key Concepts Explained
Forward Process

The diffusion forward step is:

x_t = α(t) · x₀ + σ(t) · ε


where ε ~ N(0, I).

You will see visualizations of q(x_t) gradually becoming a spherical Gaussian.

Noise Prediction Objective (DDPM loss)

The simplified training loss:

L = E[ || εθ(x_t, t) − ε ||² ]

Sampling ODE (Probability Flow ODE)
dx_t/dt = f(t) · x_t + g(t)² · ∇ₓ log p_t(x)


DPM-Solver integrates this ODE in closed form using λ-parameterization.

DPM-Solver Midpoint Rule (2nd Order)

Predictor step to λ-midpoint

Corrector step using midpoint score
Improves sample quality and stability with low NFEs.

Flow Matching

Learn a velocity field:

vθ(x_t, t) ≈ dx_t/dt


and integrate it to generate new samples.

📊 Included Visualizations

Spiral → Gaussian under diffusion

DDIM vs DPM-Solver trajectories

Score vector fields in 2D

Flow Matching velocity fields

Reconstruction trajectories of x₀

🧪 Requirements

Install dependencies:

pip install torch numpy matplotlib tqdm scikit-learn


Run notebooks:

pip install notebook
jupyter notebook

🛠️ Roadmap

 Add DPM-Solver-3

 Add MNIST / CIFAR-10 DDPM

 Add Consistency Models

 Move to /src package structure

 FM vs Diffusion comparison study

📚 References

Ho et al., DDPM (2020)

Song et al., DDIM (2020)

Lu et al., DPM-Solver (2022)

Lipman et al., Flow Matching (2023)
