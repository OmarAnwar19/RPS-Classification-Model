# CNN-RockPaperScissors-ImageClassification

## Overview

This repository contains an exciting image classification project that utilizes Convolutional Neural Networks (CNNs) to classify images as "rock," "paper," or "scissors." The project aims to accurately predict the hand gesture depicted in the input images, making it a fun and interactive tool for users.

**Image Classification:**

- The CNN model is designed to recognize intricate patterns and features in the input images to classify them into one of the three classes: "rock," "paper," or "scissors." The model undergoes thorough hyperparameter tuning and training on a diverse dataset of labeled images, capturing different hand gestures. This process enables the model to achieve impressive accuracy and robustness in classification.

## Features

- Utilizes Convolutional Neural Networks (CNNs) for image classification.
- Classifies input images into "rock," "paper," or "scissors" categories with high accuracy.
- Hyperparameter tuning for optimal model performance.

## Installation

To get started with the CNN-RockPaperScissors-ImageClassification project, install the required dependencies directly using the following command:

```bash
pip install tensorflow keras pandas numpy matplotlib
```

## Dataset

The dataset used for training and evaluating the CNN model consists of a diverse collection of labeled images representing "rock," "paper," and "scissors" hand gestures. To ensure robustness, the dataset is augmented with various transformations, enabling the model to learn intricate patterns and achieve excellent generalization.

## Files and Folders

The `rock_paper_scissors.ipynb` Jupyter notebook contains all of the work related to the CNN and the classification of images. It contains the entire model's evolution from a basic model, to a convolutional network with automatic hyperparamter tuning to obtain the highest accuracy. The notebook showcases the model's evolution and demonstrates how to evaluate its performance using various metrics, ensuring confidence in its classification results. The `rps_best` folder contains the best model generated using this combination of hyperparamters.

## Conclusion

In conclusion, the CNN-RockPaperScissors-ImageClassification project is an exciting demonstration of the potential of Convolutional Neural Networks in solving image classification tasks. The model's accuracy and robustness in distinguishing between "rock," "paper," and "scissors" hand gestures exhibit its effectiveness in real-world applications.

Through hyperparameter tuning and thorough training on a diverse dataset, the CNN model achieves impressive classification results, even in the presence of challenges such as variations in lighting conditions and diverse backgrounds. This project served as a great starting point for my exploration into image classification using deep learning techniques. The model's adaptability can extend beyond the rock-paper-scissors domain, opening doors for various other image classification applications.
