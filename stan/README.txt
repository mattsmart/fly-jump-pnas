MODELS

The following directories contain models that may be fit to the fly
jumping data.  The dynamics models are those that follow the Smart et
al. papers and are of most interest.  

* dynamics: habituation models using differential equation dynamics
  (cf. Smart et al. PNAS paper)

  - habituation-pool.stan
    complete pooling---all data combined into single estimate
    1s per data set fit time---could fit MLE via optimization
    
  - habituation-no-pool.stan
    no pooling---each fly estimated separately
    10m per data set fit time---could fit MLE via optimization

  - habituation-partial-pool.stan
    partial pooling---estimate flies regularized toward population
        average
    10m per data set fit time---MLE does not exist

  - habituation-partial-pool-multi.stan
    multivariate partial pooling---estimate flies regularized toward
        population average with correlation among parameters
    10m per data set fit time---MLE does not exist
    
* hmm: hidden Markov models (HMM) for regular pulse design (Persikov) 

* glm: generalized linear model (GLM) for regular pulse design
  (Carpenter) 
