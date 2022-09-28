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

The empirical exercise was the training of a convolutional neural network for the forecast of stock price movement based on the candlestick pattern of the previous periods, and subsequent estimation of their Shapley values. An example of sample data:

![AAL_weekly_from_2008-12-05_to_2009-03-13](https://github.com/vitorbborges/shap-values-img-recognoition/blob/main/Test/up/AAL_weekly_from_2008-12-05_to_2009-03-13.png)

The programming language used was Python 3. A total of 1000 images using the library 'mpl_finance.candlestick2_ohlc' to draw the charts and 'alpha_vantage.timeseries' for price changes. If the share price rose after observing that pattern of 'candles' the image was classified as 'up', and if the opposite happened, as 'down'.

This base was separated into 700 for training and 300 for testing. The binary model was trained using the 'keras' package from 'TensorFlow', and Shapley values for the data of training were calculated with the 'shap' library.

## Results

**High Quality** \\

After estimating the Shapley values for each data point, new models were trained by iteratively removing the x% best data with the objective to observe what effect this would have on the precision, accuracy, loss function and recall of the neural network.

![Removendo%20dados%20de%20alta%20qualidade.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20alta%20qualidade.png)


**Low Quality** \\
When we remove the low quality data from the training base, we continue to observe the effect of Shapley values on recall, but now the correlation with precision becomes much more expressive. These correlations will be studied more deeply in the econometric results.

![Removendo%20dados%20de%20baixa%20qualidade.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20baixa%20qualidade.png)

### Comparing Results

The results of the article are much more expressive than the experiment carried out by me, in the case of removing 'bad' values the researchers managed to raise the model accuracy up to more than 70%.


![Removendo%20dados%20de%20baixa%20qualidade%20artigo.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20baixa%20qualidade%20artigo.png)



