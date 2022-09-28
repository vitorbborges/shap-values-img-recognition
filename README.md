# shap-values-img-recognoition

This repository is the final presentation for the subject Numerical Methods and Computational Models in the post-graduation department of Economics at the University of Brasília.

It is replicates the methodology used in the paper [Data valuation for medical imaging using Shapley value and application to a large-scale chest X-ray dataset](https://www.semanticscholar.org/paper/Data-valuation-for-medical-imaging-using-Shapley-to-Tang-Ghorbani/8a4a77347f274b58325ef6c5575611b589d4ba6c) trying to evaluate the amount of predictbility that a single data point contributes to the power of a Deep Learning Model.

The article in question explores a method of quantifying the quality of each data point of the training base of a convolutional neural network (CNN). The neural network trained in the article uses X-ray images of the lung to diagnose pneumonia.

Images of a section of the Candlesticks chart, from different time scales, were used for S&P500 assets. The purpose of the network is to classify these images according to the price variation 5 periods after the observed pattern, the classes are 'buy' and 'sell'. The main objective is to assess whether a low quality database can compromise the accuracy of the model.

An example of sample data:

![AAL_weekly_from_2008-12-05_to_2009-03-13](https://github.com/vitorbborges/shap-values-img-recognoition/blob/main/Test/up/AAL_weekly_from_2008-12-05_to_2009-03-13.png)

The image above corresponds to the price variation per share of 'American Airlines' between the dates of 2008-12-05 and 2009-03-13, each candle corresponds to the price variation in one of the weeks of the period.

One of the main features of the X-ray database used in the article is the misclassification of data points according to the actual diagnosis of the image. According to the researchers, this can compromise the accuracy of the model, and in sequence a method of ordering the training data of the network is proposed, according to the individual contribution of that data to the joint accuracy.

## Game Theory

Shapley's Value is contextualized in a cooperative game with $n$ agents in a coalition. The function $v: S \rightarrow \mathbb{R}$, assigns to the coalition $S$ a value that corresponds to the sum of the expected payoffs that the members of the coalition can obtain.


The function $\varphi_{i}(v)$ returns a 'fair' proportion of distributing the coalition payoff according to the individual contribution of each agent. This function is defined as follows:

$$
\varphi_{i}(v) = \sum_{S \subseteq N / \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} (v(S \cup \{ i \}) - v(S))
$$

One way to interpret what is being explained in this formula is as follows:

$$
\varphi_{i}(v) = \frac{1}{number \space of \space agents} \sum_{coalition \space that \space excludes \space i} \frac{marginal \space contribution \space of \space i \space for \space this \space coalition}{number \space of \space coalitions \space that \space exclude \space i \space with \space this \space size}
$$

If we consider a neural network model as a cooperative game in which each training data point is responsible for a part of the model's effectiveness, we can measure the Shapley value for each data point as a metric of quality of this data.

## Empirical Exercise

The experiments were performed in the Python 3 programming language. The trained neural network models were imported from the tensorflow keras library and the shapley values were estimated with the shap library.

The architecture of all trained models is the same, their layers will be explained in the following code:

```python

    ########## RNC
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras.preprocessing.image import ImageDataGenerator
    import tensorflow as tf 
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'Adamax',
                  metrics = ['accuracy',tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    batch_size = 1
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       shear_range = 0.2,
                                       zoom_range = 0.2,
                                       horizontal_flip = True)
    # Testing Augmentation - Only Rescaling
    test_datagen = ImageDataGenerator(rescale = 1./255)
    # Generates batches of Augmented Image data
    train_generator = train_datagen.flow_from_directory('Train/', target_size = (300, 300), 
                                                        batch_size = batch_size,
                                                        class_mode = 'binary') 
    # Generator for validation data
    validation_generator = test_datagen.flow_from_directory('Test/', 
                                                            target_size = (300, 300),
                                                            batch_size = batch_size,
                                                            class_mode = 'binary')
    # Fit the model on Training data
    model.fit_generator(train_generator,
                        epochs = 5,
                        validation_data = validation_generator,
                        verbose = 1)
    # Evaluating model performance on Testing data
    loss, accuracy, precision, recall = model.evaluate(validation_generator)
    print("\nModel's Evaluation Metrics: ")
    print("---------------------------")
    print("Accuracy: {} \nLoss: {} \nPrecision: {} \nRecall: {}".format(accuracy, loss, precision, recall))

```

The training base of the main model contains 700 images, and the test base 300. The bases were reduced due to the computational complexity of estimating the shapley value, which, as explained in the formula, is a few orders above the factorial complexity. The program ran for several hours.

The code that calculates the shaley value for each of the training images is as follows:

```python

    import os
    import numpy as np
    import pandas as pd
    import cv2
    import shap
    import random
    import time
    from IPython.display import clear_output
    
    def create_train_dataset():
    
    img_width = 300
    img_height = 300
    
    img_data_array = []
    class_name = []
    file_name = []
    
    for class_ in os.listdir('Train/'):
        
        for file in os.listdir(os.path.join('Train/', class_)):
            
            image_path = os.path.join('Train/', class_,  file)
            image = cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_height, img_width),interpolation = cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(class_)
            file_name.append(file)
            
            
    return file_name, img_data_array, class_name

    def create_test_dataset():
    
    img_width = 300
    img_height = 300
    
    img_data_array = []
    class_name = []
    file_name = []
    
    for class_ in os.listdir('Test/'):
        
        for file in os.listdir(os.path.join('Test/', class_)):
            
            image_path = os.path.join('Test/', class_,  file)
            image = cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_height, img_width),interpolation = cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(class_)
            file_name.append(file)
            
            
    return file_name, img_data_array, class_name

    ifile_name, iimg_data, iclass_name = create_train_dataset()
    tfile_name, timg_data, tclass_name = create_test_dataset()
    
    icontent = dict(zip(ifile_name,iimg_data))
    tcontent = dict(zip(tfile_name,timg_data))
    iclasses = dict(zip(ifile_name,iclass_name))
    tclasses = dict(zip(tfile_name,tclass_name))
    
    iimg_data = np.array(iimg_data)
    timg_data = np.array(timg_data)
    
    start_time = time.time()
    
    explainer = shap.DeepExplainer(loaded_model, np.array(timg_data)[0:64])
        
    shapley_values = open('shapley_values.csv', 'a')
    
    for i in range(len(pd.read_csv('shapley_values.csv')['file_name']),700):
        
        clear_output(wait = True)
        
        img = iimg_data[i:i+1]
        shap_values = explainer.shap_values(img)
        
        shapley_values.write(f'{ifile_name[i]},{sum(sum(sum(shap_values[0][0])))},{iclass_name[i]}\n')
        
        print(f'{i/700:.3f}%')
        
    print("--- %s seconds ---" % (time.time() - start_time))

```

After estimating the shapley values, the images were classified and stored in an orderly manner from the best to the worst contribution to effectiveness. The experiment iteratively removes the x% best data, and retrains the network to measure the effects of this on the measures of: Accuracy, Precision, Recall and Loss Function.

The results found were the following:


![Removendo%20dados%20de%20alta%20qualidade.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20alta%20qualidade.png)

The effect that drew the most attention in this case was the progressive reduction of the 'recall' when the model was losing its high quality data. The rest looks like random variations.

### Comparing Results

In the article, the main result was the reduction of accuracy with the removal of data of high quality, showing that the data with the high Shapley value were really the ones that most caused the prediction.

![Removendo%20dados%20de%20alta%20qualidade%20artigo.png](https://github.com/vitorbborges/shap-values-img-recognoition/raw/main/Graphs%20and%20Tables/Removendo%20dados%20de%20alta%20qualidade%20artigo.png)


**Low Quality** 

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
