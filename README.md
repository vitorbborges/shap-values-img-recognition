# shap-values-img-recognoition

This repository is the final presentation for the subject Numerical Methods and Computational Models in the post-graduation department of Economics at the University of Brasília.

It is replicates the methodology used in the paper [Data valuation for medical imaging using Shapley value and application to a large-scale chest X-ray dataset](https://www.semanticscholar.org/paper/Data-valuation-for-medical-imaging-using-Shapley-to-Tang-Ghorbani/8a4a77347f274b58325ef6c5575611b589d4ba6c) trying to evaluate the amount of predictbility that a single data point contributes to the power of a Deep Learning Model.

## References and Motivation

The article by Siyi Tang, Amirata Ghorbani, Rikiya Yamashita, Sameer Rehman, Jared A. Dunnmon, James Zou & Daniel L. Rubin use the Shapley value from Game Theory to rank the training data of a convolutional neural network according to its importance. 
The main objective is to assess whether a low quality database can compromise the accuracy of the model. The article used X-ray images of the lung to predict pneumonia.

## Game Theory

Shapley value is a game theory concept that measures the contribution individual agent for the 'payoff' of a cooperative game. The formal definition of this estimator is as follows:

$$
\phi_i(v) = \sum_{S \subseteq N \textbackslash \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} (v(S \cup \{i\}) - v(S))
$$

If we consider a neural network model as a cooperative game in which each training data point is responsible for a part of the model's effectiveness, we can measure the Shapley value for each data point as a metric of quality of this data.

## Dataset

The empirical exercise was the training of a convolutional neural network for the forecast of stock price movement based on the candlestick pattern of the previous periods, and subsequent estimation of their Shapley values. An example sample data:

![alt text](https://github.com/vitorbborges/shap-values-img-recognoition/blob/main/Test/up/AAL_weekly_from_2008-12-05_to_2009-03-13.png){ width=50% }
