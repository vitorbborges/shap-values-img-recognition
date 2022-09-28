# shap-values-img-recognoition

This repository is the final presentation for the subject Numerical Methods and Computational Models in the post-graduation department of Economics at the University of Brasília.

It is replicates the methodology used in the paper [Data valuation for medical imaging using Shapley value and application to a large-scale chest X-ray dataset](https://www.semanticscholar.org/paper/Data-valuation-for-medical-imaging-using-Shapley-to-Tang-Ghorbani/8a4a77347f274b58325ef6c5575611b589d4ba6c) trying to evaluate the amount of predictbility that a single data point contributes to the power of a Deep Learning Model.

## References and Motivation

The article by Siyi Tang, Amirata Ghorbani, Rikiya Yamashita, Sameer Rehman, Jared A. Dunnmon, James Zou & Daniel L. Rubin use the Shapley value from Game Theory to rank the training data of a convolutional neural network according to its importance. 
The main objective is to assess whether a low quality database can compromise the accuracy of the model. The article used X-ray images of the lung to predict pneumonia.

## Game Theory

Shapley's Value is contextualized in a cooperative game with $` n `$ agents in a coalition. The function $ v: S \rightarrow \mathbb{R} $, assigns to the coalition $ S $ a value that corresponds to the sum of the expected payoffs that the members of the coalition can obtain.

$$
\phi_i(v) = \sum_{S \subseteq N \textbackslash \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} (v(S \cup \{i\}) - v(S))
$$

One way to interpret what is being explained in this formula is as follows:

$$
\varphi_{i}(v) = \frac{1}{number \space of \space agents} \sum_{coalition \space that \space excludes \space i} \frac{marginal \space contribution \space of \space i \space for \space this \space coalition}{number \space of \space coalitions \space that \space exclude \space i \space with \space this \space size}
$$

If we consider a neural network model as a cooperative game in which each training data point is responsible for a part of the model's effectiveness, we can measure the Shapley value for each data point as a metric of quality of this data.

## Dataset

The empirical exercise was the training of a convolutional neural network for the forecast of stock price movement based on the candlestick pattern of the previous periods, and subsequent estimation of their Shapley values. An example of sample data:

![AAL_weekly_from_2008-12-05_to_2009-03-13](https://github.com/vitorbborges/shap-values-img-recognoition/blob/main/Test/up/AAL_weekly_from_2008-12-05_to_2009-03-13.png)

The programming language used was Python 3. A total of 1000 images using the library 'mpl_finance.candlestick2_ohlc' to draw the charts and 'alpha_vantage.timeseries' for price changes. If the share price rose after observing that pattern of 'candles' the image was classified as 'up', and if the opposite happened, as 'down'.

This base was separated into 700 for training and 300 for testing. The binary model was trained using the 'keras' package from 'TensorFlow', and Shapley values for the data of training were calculated with the 'shap' library.

## Results

**High Quality** \

After estimating the Shapley values for each data point, new models were trained by iteratively removing the x% best data with the objective to observe what effect this would have on the precision, accuracy, loss function and recall of the neural network.

![Removendo%20dados%20de%20alta%20qualidade.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20alta%20qualidade.png)

The effect that drew the most attention in this case was the progressive reduction of the 'recall' when the model was losing its high quality data. The rest looks like random variations.

### Comparing Results

In the article, the main result was the reduction of accuracy with the removal of data of high quality, showing that the data with the high Shapley value were really the ones that most caused the prediction.

![Removendo%20dados%20de%20alta%20qualidade%20artigo.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20alta%20qualidade%20artigo.png)


**Low Quality** \

When we remove the low quality data from the training base, we continue to observe the effect of Shapley values on recall, but now the correlation with precision becomes much more expressive. These correlations will be studied more deeply in the econometric results.

![Removendo%20dados%20de%20baixa%20qualidade.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20baixa%20qualidade.png)

### Comparing Results

The results of the article are much more expressive than the experiment carried out by me, in the case of removing 'bad' values the researchers managed to raise the model accuracy up to more than 70%.


![Removendo%20dados%20de%20baixa%20qualidade%20artigo.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20baixa%20qualidade%20artigo.png)

## Discussions

The researchers attributed this significant improvement in accuracy to the loss of data misclassified in its base. X-ray imaging data can often sometimes contain low definition and misclassification problems.

In the case of the experiment carried out, we do not have this problem because the variations in the share prices are observed with almost absolute certainty, just check if the price was higher or lower after that pattern to get the correct rating. Therefore we probably did not observe a significant improvement in accuracy, but an improvement in precision/recall.

To measure this, the researchers reassessed their database, using the opinion of three radiologists, and observed that indeed the data points with more extreme Shapley values mostly had some sorting problem.

For the sole purpose of reproducing this experiment, I was asked to two colleagues with experience in the field of 'Day Trade' and graphic analysis to make an assessment of these extreme values found. The result was quite interesting:

![resultado%20colega%201.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/resultado%20colega%201.png)

![resultado%20colega%202.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/resultado%20colega%202.png)

The result becomes more interesting when we compare these results with the
predictions from a person who had just discovered what candlestick charts were:

![resultado%20leigo.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/resultado%20leigo.png)

The layman had better foresight than one of his colleagues with experience in the field. subject, and even better than some of the complex models of convolutional neural networks. This result raises hypotheses about the effectiveness of this type of investment strategy, however it is outside the scope of this presentation.
