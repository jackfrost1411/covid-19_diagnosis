# Detecting COVID-19 Using Convolution Neural Networks
During this course of international emergency, diagnosing patients infected with COVID-19 at an early stage with the help of deep learning models is a crucial development. The paper aims to evaluate the deep learning models available for the image classification task for detecting COVID-19. The dataset containing 956 X-ray images of three classes, namely COVID-19, viral pneumonia and normal, is used. Standard deep learning models like AlexNet, ResNets and Inception v3 along with various custom models of convolution neural networks (CNNs) have been trained and tested on the dataset. The Inception v3 model gave the best training accuracy of 99.22%, while custom made CNN3 gave a promising training accuracy of 96.61%. Both models gave a similar validation accuracy of 97.89%. Sensitivity and specificity for COVID-19 were (100% and 98.5%) and (100% and 100%) for Inception v3 and CNN3, respectively.

## Please cite the paper if any code from this repository is used. Link to the paper:
https://link.springer.com/chapter/10.1007%2F978-981-33-6691-6_17

## Instructions:
### Training
1. To train the CNN3 model switch to the repository after clonning.
2. Run `python TrainCNN.py`
3. The model will be saved once the training is finished.
4. The same steps apply for TrainInceptionV3.py file.

If importing keras.processing gives an error, try keras_processing.

## Dataset
![dataset](https://github.com/jackfrost1411/covid-19_diagnosis/blob/main/dataset.png)
