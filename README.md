# **On-policy Control with Function Approximation for MountainCar**

## **Overview**

This project delves into **on-policy control methods using function approximation**, specifically focusing on the challenging **MountainCar-v0** environment from Gymnasium. The primary objective is to implement and analyze Episodic Semi-Gradient SARSA and its n-step variant, leveraging Fourier basis functions for efficient state-space generalization in a continuous control task. This work explores how different approximation orders and n-step values influence learning speed and policy optimality.

## **Environment: MountainCar-v0**

The **MountainCar-v0** environment presents a classic control problem: an underpowered car must learn to leverage gravity by moving away from its goal to build enough momentum to reach the top of a steep mountain.

* **Observation Space:** 2-dimensional, comprising the car's position and velocity.  
* **Action Space:** 3 discrete, deterministic actions: accelerate left, do nothing, or accelerate right.  
* **Reward:** \-1 at each time step until the goal is reached, terminating the episode.  
* **Initial State:** Random position in \[-0.6, \-0.4) with zero velocity.  
* **Goal:** Reach a position ≥0.5 on the right hill.

## **Key Implementations & Concepts**

This notebook includes a robust implementation of the following core Reinforcement Learning components:

1. **Fourier Basis Function Approximation:**  
   * A function to generate Fourier basis features X(s)=\[cos(0⋅π⋅s),…,cos(n⋅π⋅s)\] for 1D continuous states.  
   * A function to calculate the state-value V(S,w) as a dot product of weights w and Fourier features X(S).  
   * Implementation of a bound function to normalize state values to \[0, 1\] for effective Fourier basis application.  
2. **Episodic Semi-Gradient SARSA:**  
   * A complete implementation of the Semi-Gradient SARSA algorithm, an on-policy TD control method.  
   * Utilizes function approximation to handle the continuous state space of MountainCar.  
   * Employs an epsilon-greedy policy for action selection, balancing exploration and exploitation.  
3. **Episodic Semi-Gradient n-step SARSA:**  
   * An extension of the basic SARSA algorithm incorporating n-step returns. This allows for a flexible trade-off between bias and variance in value estimation, potentially leading to faster learning.  
   * The implementation allows for varying n values (e.g., 1, 8, 16\) to observe their impact on performance.

## **Experiments & Analysis**

The project includes sections for:

* **Environment Verification:** Initial simulation to confirm correct observation, action, reward, and termination logging.  
* **Hyperparameter Tuning:** Guidance on tuning the step-size (α), function approximation order (num\_basis), discount factor (γ), and exploration probability (ε).  
* **Averaged Runs & Plotting:** Functions to run the algorithms multiple times (e.g., 50-100 runs) and average the results to reduce variance.  
  * Plots of "Sum of Reward per Episode" vs. "Number of Episodes".  
  * Plots of "Steps per Episode (Log Scale)" vs. "Number of Episodes".  
* **Policy Animation:** Visualization of the learned policy in the MountainCar environment for qualitative assessment.  
* **Comparative Analysis:** Investigation into how different n values in n-step SARSA affect learning speed and overall performance.

## **Usage**

To run this notebook:

1. Ensure you have Python installed.  
2. Install the required libraries:  
   pip install numpy matplotlib gymnasium

3. Open the On\_policy\_control.ipynb file in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab, VS Code with Jupyter extension).  
4. Run all cells sequentially.

The notebook is designed for interactive execution, allowing you to modify parameters and observe results directly.

## **Dependencies**

* numpy: For numerical operations.  
* matplotlib: For plotting results.  
* gymnasium: For the Reinforcement Learning environment.

## **General Notes**

* All vector or matrix operations use numpy arrays.  
* np.max and np.random.normal are used for consistency.  
* All plots include xlabel, ylabel, legend, title, and grid for clarity.
