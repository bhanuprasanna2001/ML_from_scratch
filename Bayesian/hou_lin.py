import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

# Synthetic Data
size = np.random.rand(100)
price = 2 + 3 * size + np.random.normal(0, 0.5, 100)

with pm.Model() as linear_model:
    # --- PRIORS ---
    alpha = pm.Normal('alpha', mu=0, sigma=10) # Intercept
    beta = pm.Normal('beta', mu=0, sigma=10)   # Slope
    sigma = pm.HalfNormal('sigma', sigma=1)    # Noise
    
    # --- EXPECTED VALUE ---
    mu = alpha + beta * size
    
    # --- LIKELIHOOD ---
    # "I believe the observed prices come from a Normal distribution 
    # centered around mu with some noise sigma"
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=price)
    
    # --- INFERENCE ---
    trace_linear = pm.sample(1000)
    
az.plot_posterior(trace_linear, var_names=['alpha', 'beta', 'sigma'])
plt.show()

# Print summary
print(az.summary(trace_linear, var_names=['alpha', 'beta', 'sigma']))
    
    
# Synthetic Data
hours = np.random.uniform(0, 10, 100)
# Probability of passing increases with hours
true_p = 1 / (1 + np.exp(-( -4 + 1 * hours))) 
pass_fail = np.random.binomial(1, true_p)

with pm.Model() as logistic_model:
    # --- PRIORS ---
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    
    # --- DETERMINISTIC (The Link Function) ---
    # This turns our line (-infinity to +infinity) into a probability (0 to 1)
    # pm.math.invlogit is the PyMC name for Sigmoid
    p = pm.math.invlogit(alpha + beta * hours)
    
    # --- LIKELIHOOD ---
    # "I believe the data are coin flips where the chance of heads 
    # is determined by 'p'"
    Y_obs = pm.Bernoulli('Y_obs', p=p, observed=pass_fail)
    
    trace_logistic = pm.sample(1000)
    
az.plot_posterior(trace_logistic, var_names=['alpha', 'beta'])
plt.show()

# Print summary
print(az.summary(trace_logistic, var_names=['alpha', 'beta']))
    
    
# Synthetic Data: Apples ~150g, Bananas ~120g
weights_apple = np.random.normal(150, 10, 50)
weights_banana = np.random.normal(120, 10, 50)
weights = np.concatenate([weights_apple, weights_banana])
# Labels: 0 for apple, 1 for banana
labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)

with pm.Model() as naive_bayes_model:
    # --- PRIORS ---
    # We need to learn the parameters for TWO distributions
    
    # Mean weight for Apples (class 0) and Bananas (class 1)
    # shape=2 tells PyMC we want two separate variables
    mus = pm.Normal('mus', mu=135, sigma=50, shape=2) 
    
    # Standard deviation (width) for both
    sigmas = pm.HalfNormal('sigmas', sigma=20, shape=2)
    
    # --- LIKELIHOOD ---
    # This is the "Naive Bayes" magic.
    # We select the mu and sigma corresponding to the label of that specific data point.
    # If label[i] is 0, use mus[0]. If label[i] is 1, use mus[1].
    
    weight_obs = pm.Normal('weight_obs', 
                           mu=mus[labels], 
                           sigma=sigmas[labels], 
                           observed=weights)
    
    trace_nb = pm.sample(1000)
    
az.plot_posterior(trace_nb, var_names=['mus', 'sigmas'])
plt.show()

# Print summary
print(az.summary(trace_nb, var_names=['mus', 'sigmas']))
    
    
# Synthetic Data
# 3 players with different number of trials (at_bats)
at_bats = np.array([10, 1000, 100]) 
hits = np.array([4, 300, 35]) 
player_idx = [0, 1, 2] # IDs for the 3 players

with pm.Model() as hierarchical_model:
    # --- HYPER-PRIORS (The League Level) ---
    # "phi" is the League Average batting capability
    phi = pm.Beta('phi', alpha=1, beta=1) 
    # "kappa" is how consistent the league is (concentration)
    kappa = pm.HalfNormal('kappa', sigma=100)

    # Convert phi/kappa to alpha/beta for the Beta distribution
    # (Just math conversion for the Beta parametrization)
    alpha_league = phi * kappa
    beta_league = (1 - phi) * kappa

    # --- PRIORS (The Player Level) ---
    # Each player has their own 'theta' (batting avg), 
    # BUT it is drawn from the League Distribution defined above!
    thetas = pm.Beta('thetas', alpha=alpha_league, beta=beta_league, shape=3)

    # --- LIKELIHOOD ---
    # Binomial: k successes in n trials
    y = pm.Binomial('y', n=at_bats, p=thetas, observed=hits)

    trace_hierarchical = pm.sample(1000)
    
    
az.plot_posterior(trace_hierarchical, var_names=['phi', 'kappa', 'thetas'])
plt.show()

# Print summary
print(az.summary(trace_hierarchical, var_names=['phi', 'kappa', 'thetas']))