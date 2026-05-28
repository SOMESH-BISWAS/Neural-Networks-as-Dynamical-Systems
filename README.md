# Neural Networks as Continuous Dynamical Systems 🧠♾️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![SciPy](https://img.shields.io/badge/SciPy-Integration-lightgrey)
![Math](https://img.shields.io/badge/Mathematics-Differential_Equations-red)

**Author:** Somesh Biswas  
**Institution:** Indian Statistical Institute (ISI), Kolkata  
**Course:** Mathematics IV  

## 📖 About the Project

This repository contains the mathematical analysis and computational demonstrations for the study of **Neural Ordinary Differential Equations (Neural ODEs)**. 

Historically, deep neural networks have been viewed as discrete sequences of hidden layers. This project bridges modern machine learning architectures with classical continuous mathematics by proving that a standard Residual Network (ResNet) layer is mathematically identical to a single forward Euler step in numerical integration. By taking the limit as the step size approaches zero, we recast deep learning completely in the language of continuous dynamical systems.

### Key Mathematical Concepts Explored:
* **Initial Value Problems (IVPs):** Formulating the forward pass of a neural network as the continuous evolution of an ODE.
* **Picard-Lindelöf Theorem:** Proving the existence and uniqueness of network outputs by evaluating the Lipschitz continuity of modern activation functions (Tanh, ReLU).
* **Boundary Value Problems (BVPs):** Reformulating network training (backpropagation) as a BVP solved backward in time using the **Adjoint Sensitivity Method**.
* **Topological Constraints:** Demonstrating the strict limits of continuous models—specifically that 1D continuous trajectories cannot cross, restricting the network from learning mappings that require "tearing" the feature space.

## 💻 Computational Demonstration

The project includes a Python implementation that compares a standard discrete ResNet layer against a continuous Neural ODE forward pass using Runge-Kutta integration (`RK45`). 

The code proves that discrete ResNets accumulate numerical integration errors due to large step sizes ($\Delta t = 1$), whereas continuous ODE solvers accurately track the true topological flow.

### Prerequisites
To run the computational analysis, you need `numpy` and `scipy`:
```bash
pip install numpy scipy
