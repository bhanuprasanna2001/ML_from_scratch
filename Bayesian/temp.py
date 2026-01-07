import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# 1. Generate Synthetic Data
# True relationship: Sales = 100 + (2 * Temp)
np.random.seed(42)
true_alpha = 100
true_beta = 2
true_sigma = 10 # Noise
size = 100

temperature = np.random.normal(25, 5, size) # Avg temp 25C
sales = true_alpha + true_beta * temperature + np.random.normal(0, true_sigma, size)

# 2. Build the PyMC Model
with pm.Model() as model:
    # --- PRIORS (Our beliefs before seeing data) ---
    
    # Alpha: Intercept. Sales can't be 0, likely around 50-200?
    # We use Normal, centered at 0 but wide, to be "weakly informative".
    alpha = pm.Normal('alpha', mu=0, sigma=100)
    
    # Beta: Slope. Does temp increase sales? Maybe? 
    # We guess it's around 0 with some wiggle room.
    beta = pm.Normal('beta', mu=0, sigma=10)
    
    # Sigma: The noise/error. Must be positive.
    # HalfNormal is great for positive-only values.
    sigma = pm.HalfNormal('sigma', sigma=20)

    # --- EXPECTED VALUE (The deterministic math) ---
    # This is the line equation y = mx + c
    mu = alpha + beta * temperature

    # --- LIKELIHOOD (The connection to Data) ---
    # We say our observed 'sales' comes from a Normal distribution
    # centered at 'mu' with noise 'sigma'.
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=sales)

    # --- INFERENCE (The Engine) ---
    # We ask the hiker to explore the mountain (MCMC)
    # This creates the "Trace" (the history of the hiker's steps)
    trace = pm.sample(1000, return_inferencedata=True)

# 3. Analyze the Results
# Plot the distributions of Alpha and Beta
az.plot_posterior(trace, var_names=['alpha', 'beta', 'sigma'])
plt.show()

# Print summary
print(az.summary(trace, var_names=['alpha', 'beta', 'sigma']))