🧠 Diffusion & Flow Models — Mini-Projects

This repository contains a set of mini-projects exploring modern generative modeling techniques, including diffusion models, ODE-based solvers, and flow-matching.
The goal is to understand these models from first principles — using clear mathematics, visualization, and simple 2D datasets.

🧩 Project Topics

Each notebook focuses on a fundamental concept in diffusion-based generative modeling:

🌪️ DDPM (Denoising Diffusion Probabilistic Models)
Forward diffusion, reverse denoising, and the noise-matching objective.

🔁 DDIM (Deterministic Sampling)
Non-stochastic sampling based on ODE interpretation.

⚡ DPM-Solver (1st & 2nd Order)
Fast sampling using ODE solvers in continuous time.

🧷 Flow Matching (FM)
Learning continuous flows via velocity field supervision.

🌀 Toy Dataset Transformations
Spiral, swiss-roll, checkerboard, and Gaussian transitions.

📂 Repository Structure

This repository contains several Jupyter notebooks.
Each notebook includes implementation, visualizations, and explanations.

<details> <summary><strong>Click to view notebook breakdown</strong></summary>
Notebook	Description
DDPM_DDIM.ipynb	DDPM training and DDIM sampling on 2D datasets
DPM_Solver.ipynb	DPM-Solver implementation (1st and 2nd order solvers)
flow_matching.ipynb	Gaussian Flow Matching and velocity-based generation
</details>
🔍 Key Concepts Explained
Forward Process

The diffusion forward step is defined as:

x_t = alpha(t) * x0 + sigma(t) * noise


As t increases, the data is gradually transformed into Gaussian noise.

Noise Prediction Objective (DDPM Loss)

The model is trained to predict the noise added at time t:

Loss = E[ || predicted_noise - true_noise ||^2 ]

Sampling ODE (Probability Flow ODE)

A continuous-time ODE version of the reverse process:

dx/dt = drift(x, t) + score_term(x, t)


DPM-Solver integrates this ODE efficiently using closed-form updates.

DPM-Solver (2nd Order Midpoint Rule)

A fast and accurate solver using:

A predictor step

A midpoint evaluation

A corrector step

This yields better quality at lower number of function evaluations.

Flow Matching

Instead of predicting noise, flow matching predicts the velocity:

v(x, t) = d/dt [ x_t ]


Generation is performed by integrating the learned flow field.

📊 Visualizations Included

Forward diffusion: spiral → Gaussian

DDIM and DPM-Solver sampling trajectories

Flow Matching velocity fields

Final reverse-sampled generations

Evolution of q(x_t) over time

🧪 Requirements

Install dependencies:

pip install torch numpy matplotlib tqdm scikit-learn


To launch notebooks:

pip install notebook
jupyter notebook
