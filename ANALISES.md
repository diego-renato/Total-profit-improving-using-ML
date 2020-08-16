## Table of contents
- [1. The problem](#1-the-problem)
- [2. The data](#2-the-data)
- [3. Comparing different scaler functions](#3-comparing-different-scaler-functions)
    -[3.1. Normalization scaler: scatter and hist (01-06-2020 to 08-06-2020)](#3-normalization-scaler)
    -[3.2. Standard scaler: scatter and hist (01-06-2020 to 08-06-2020)](#3-standard-scaler)
    -[3.3. Max abs scaler: scatter and hist (01-06-2020 to 08-06-2020)](#3-max-abs-scaler)
    -[3.4. Min max scaler: scatter and hist (01-06-2020 to 08-06-2020)](#3-min-max-scaler)
    -[3.5. Power transformer: scatter and hist (01-06-2020 to 08-06-2020)](#3-power-transformer-scaler)
    -[3.6. Robust scaler: scatter and hist (01-06-2020 to 08-06-2020)](#3-robust--scaler)

- [4. The optimal number of clusters](#4-the-optimal-number-of-clusters)

# 1. The problem

# 2. The data

# 3. Comparing different scaler functions
After scaling the data, is important to see the behavior of the scaled or transformed data.
In this section I compared the different scaler jobs from the considered in this project for the week 01-06-2020 to 08-06-2020 and
the week 24/07/2020 - 31/08/2020 by scatter plots and histograms. The variables selected are <b>longitude</b>, <b>latitude</b> and <b>preço</b> because are variables
to be considered to obtain clusters in the next section.  

* I considered in the scatter plot the variables <b>longitude</b> and <b>latitude</b>, because this variables generate a map.
* And <b>preço</b> for the histogram.

#### Data transformation: scatter and hist (01-06-2020 to 07-06-2020)

##### Scatter plots
<div align="center">
    <img width="350" src="https://github.com/diego-renato/desafio-iafront/blob/branch-edit/resultados_gr%C3%A1ficos/semana1/before_scatter_semana1.png" />
    <p>Figure 1.1: Longitude vs latitude of the customers before transformation.</p>
</div>
<br>
<br>
<br>

Normalization              |  Standard scaler          | Max abs scaler
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="https://github.com/diego-renato/desafio-iafront/blob/branch-edit/resultados_gr%C3%A1ficos/semana1/after_normalization_scatter_semana1.png" /> | <img width="750" src="https://github.com/diego-renato/desafio-iafront/blob/branch-edit/resultados_gr%C3%A1ficos/semana1/aftet_scatter_standard_semana1.png" /> | <img width="750" src="https://github.com/diego-renato/desafio-iafront/blob/branch-edit/resultados_gr%C3%A1ficos/semana1/after_scatter_max_abs_semana1.png" />

Min max scaler             |  Power transformer        | Robust scaler 
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="https://github.com/diego-renato/desafio-iafront/blob/branch-edit/resultados_gr%C3%A1ficos/semana1/after_min_max_scatter_semana1.png" /> | <img width="750" src="https://github.com/diego-renato/desafio-iafront/blob/branch-edit/resultados_gr%C3%A1ficos/semana1/after_scatter_power-transformer_semana1.png" /> | <img width="750" src="https://github.com/diego-renato/desafio-iafront/blob/branch-edit/resultados_gr%C3%A1ficos/semana1/after_scatter_robust_scaler_semana1.png" />

<p>Figure 1.2: Longitude vs latitude of the customers after transformation.</p>

##### Histograms plot
<div align="center">
    <img width="750" src="" />
    <p>Figure 1.1: Frequency histogram of preço of the considered products from customers before transformation.</p>
</div>
<br>
<br>
<br>

Normalization              |  Standard scaler          | Max abs scaler
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="" /> | <img width="750" src="" /> | <img width="750" src="" />

Min max scaler             |  Power transformer        | Robust scaler 
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="" /> | <img width="750" src="" /> | <img width="750" src="" />

<p>Figure 2.2: Frequency histogram of preço of the considered products from customers after transformation.</p>

#### Data transformation: scatter and hist (24-06-2020 to 30-06-2020)

##### Scatter plots
<div align="center">
    <img width="750" src="" />
    <p>Figure 3.1: Longitude vs latitude of the customers before transformation.</p>
</div>
<br>
<br>
<br>

Normalization              |  Standard scaler          | Max abs scaler
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="" /> | <img width="750" src="" /> | <img width="750" src="" />

Min max scaler             |  Power transformer        | Robust scaler 
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="" /> | <img width="750" src="" /> | <img width="750" src="" />

<p>Figure 1.2: Longitude vs latitude of the customers after transformation.</p>

##### Histograms plot
<div align="center">
    <img width="750" src="" />
    <p>Figure 3.1: Frequency histogram of preço of the considered products from customers before transformation.</p>
</div>
<br>
<br>
<br>

Normalization              |  Standard scaler          | Max abs scaler
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="" /> | <img width="750" src="" /> | <img width="750" src="" />

Min max scaler             |  Power transformer        | Robust scaler 
:-------------------------:|:-------------------------:|:-------------------------:
<img width="750" src="" /> | <img width="750" src="" /> | <img width="750" src="" />

<p>Figure 3.2: Frequency histogram of preço of the considered products from customers after transformation.</p>




Scatter plots an histograms from the other week considered can be founded in [plots from the last week]().

Some observation for this section:
* The standard, min-max, max abs and robust scalers have the best results in the sense that the variables have the same behaviour.
* The power transformer and normalization modify the distribution of variables (not good for our case).
* The standard, min-max and max abs scalers(in this order) have the best results because they don't modify the distributions of the points and also have similar statistics(mean and variance).
 
* The methods selected are standard, min-max and max abs scalers.

# 4. The optimal number of clusters
The number of cluster is considered a hyperparameter, in the literature there are different ways to find the optimal number of cluster. 
For example from short datasets the hierarchical clustering is a good option(not in our case), the elbow method(graphics method) or 
the GAP statistic(ca be used in any clustering method), but there is another and powered method using a distribution probability actually a 
finite mixture distribution and computing some information criterion like AIC or BIC, it is the finite gaussian mixture. The gaussian mixture 
is a parametric probability distribution from *K* gaussian or normal random variables. For short explanation let see the equation above
 
<img src=https://wikimedia.org/api/rest_v1/media/math/render/svg/2f13843df7f69545e27b06c4b59f1d8fe9690ce1>

The *N(.)* denote a multivariate normal distribution with a vector of means and matrix of covariance, the 
phi denote the probability for a *latent variable* belong a one population, the *latent variable* is actually the cluster and
the number *K* is the number of cluster or populations. Please see [gaussian mixture](http://leap.ee.iisc.ac.in/sriram/teaching/MLSP_16/refs/GMM_Tutorial_Reynolds.pdf)
if you want to see more details.

The benefit of using the gaussian mixture is that the dataset belows a multivariate distribution, they don´t need to be scaled
 and the log likelihood can be computed and also be compared for different k values (clusters). And also, we can predict the cluster from a given
 vector of observation with their respective probability.
 
So, for our case, it is considered a grid of values, *K = 2,3,...,15* ,to compute the AIC (Akaike information criterion) using train-test split. Then,
the optimal number of cluster is the value of *K* that minimize the AIC in the test data (30% of original data). This is important because we can evaluate in a
unobserved dataset(Observation: The AIC penalizes the -2 log likelihood by the number of parameters.). The following graph shows the results obtained.

<div align="center">
    <img width="750" src="" />
    <p>Figure 4: AIC values from validation set using finite gaussian mixture .</p>
</div>
<br>
<br>
<br>

Also we can see the behaviour of the data by dimensional reduction for data visualization, in this context the PCA can be useful,
keep in main that the data have to be standard scaled or have the same variance because the PCA works with the covariance matrix. 
The following graph shows the results obtained from 2 component that represent more than the 50% of the total variability explained. 

<div align="center">
    <img width="750" src="./doc/source/dataset/lagarta/lagarta.jpeg" />
    <p>Figure 8: Principal components scatter plot from 2 components .</p>
</div>
<br>
<br>
<br>

# 5. Obtaining clusters

