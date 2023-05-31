## Pneumonia Detection:

![](https://storage.googleapis.com/kaggle-competitions/kaggle/10338/logos/header.png)

Detects pneumonia from X-ray images. Use the following commandline to run:
```
$ python PneumoniaDetection.py
```

## Results:
Illustration of Normal vs. Pneumonia X-rays. Untrained eye can not distinguish pneumonia from images. Training & testing dataset is composed of 1200 normal + 1200 pneumonia images .

Sample X-rays              |  Train & Test Dataset
:-------------------------:|:-------------------------:
<img src="results/Pneumonia.png" width="800">   |  <img src="results/Dataset.png" width="300">

Tested four classification algortihms as Baseline with accuracies as listed below. MLP has the highest accuracy of ~75%:
```
Running KNeighbors ...
Running LogisticRegression ...
Running DecisionTree ...
Running MLP ...
     KNeighbors      LogisticRegression     DecisionTree      MLP (Perceptron)  
       0.7025              0.6775              0.6575              0.7575      
```

## Related Work:
1. Paul Mooney, Detecting Pneumonia in X-Ray Images https://www.kaggle.com/code/paultimothymooney/detecting-pneumonia-in-x-ray-images/input
2. D. S. Kermany, M. Goldbaum, W. Cai, M. A. Lewis, K. Zhang
Huimin Xia, "[Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)", Cell, Feb 2018

## Reference:
"[Classifier comparison"] (https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)