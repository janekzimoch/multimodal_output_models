# multimodel_output_models
 
In this repository i explore different methods of modeling problems where for a given input there are many potential solutions. You can also think of it as one-to-many mapping or input having multimodal output. Standard, discriminative regression machine learning models orclassification models with a softmax head are unable to output a set of potential solutions and willeither fail at learning the task or will consistently output only single output. 

To model problems with multimodal output we need to output a probability distribution p(y|x) where rather than a scalar our model outputs adsitribution. In this repository we explore following models:

* Mixture Density Networks (MDN) - you can read more about them in C. Bishop's book ["Machine Learning and PAttern Recognition" Chapter 5.6 "Mixture Density networks pages 272-277](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
* Generative models - constructed using Bayesian Network and Monte Carlo integration (to marginalise out mixture variables) (see notes and derivations

To fully explore different Multimodal Output Models I create 3 toy-datasets:
* 1. Mixture of Linear Functions
* 2. Mixture of inverted Sinusoidal functions - where each inverted sinusoid may hav multiple y outputs for some x values.
* 3. A custom dataset which is meant to be a extreme simplification of camera relocalisation task which i do for my disertation
