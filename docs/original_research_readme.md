# ESO-project

## Understanding the feasibility of storage technology solutions for ESO

The project was developed to propose a solution to ESO to achieve the Next Zero target by 2050 for the GB-wide energy system. In particular, the project was about understanding where different storage technology (e.g., short-duration tech solution in this case) solutions are needed when there exists a shortfall of energy from renewable sources.

### Objective:
The objective was to identify Grid Supply Points (GSPs) suitable for installing short-duration storage technology solutions like battery by signal analysis and develop a probabilistic machine learning model to visualise those suitable GSPs in 2D latent space derived from their very high dimensional space.

### Assumptions:
6-8 hours of off-peak time interval has been considered sufficient for a recharge of short-duration storage like battery. The period is sensible because of the general trend we see across the UK of an overnight low demand followed by a peak in the morning or a low demand in the afternoon followed by a peak in the evening.

### Mathematical and Statistical method applied:

**Steps:**

**1. Fourier analysis –** Frequency analysis of each time series GSP data has been done by Fourier transform. It helps us to label or categorize GSPs for battery or no-battery based on its Fourier spectral density measure

**2. GPLVM –** The unsupervised non-parametric statistical and machine learning model such as Gaussian Process Latent Variable Model has been applied on the labelled GSP data to visualize them in 2D latent space

### Python Package used for GPLVM model: 
pyro library from the PyTorch framework

### Results with conclusion:
**1.** A Very High dimensional GSP data can be visualised in 2-dimensional latent space by the GPLVM

**2.** As we move along from left to right in the 2D latent space, the scope for short term storage (battery) installation lessens and the variance of the latent points increases

**3.** Usually, in the area of separation or coexistence of the points from two categories (battery or no-battery), the priority of storage installation should be the least because of their nearly close variance and thereby proximity to each other











