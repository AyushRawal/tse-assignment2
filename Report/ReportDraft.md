# Assignment 2 Report: Secure AI Systems - Red and Blue Teaming an MNIST Classifier

## Executive Summary

This report presents a comprehensive security analysis of an MNIST digit classification system through red team and blue team exercises. We developed a CNN-based application, identified vulnerabilities through STRIDE threat modeling and SAST analysis, executed data poisoning and adversarial attacks, and implemented robust defense mechanisms through adversarial training.

## 1. Application Description

For this assignment, we developed a handwritten digit classification application in Python that trains a convolutional neural network (CNN) model on the MNIST dataset and provides a web interface for real-time digit classification.

**Key Components:**

- `train.py`: Handles model training, evaluation, and performance metrics generation
- `server.py`: Flask web application that serves the trained model for digit classification via a drawing interface

**Repository Links:**

- GitHub Repository: https://github.com/AyushRawal/tse-assignment2
- Generated Datasets: [SharePoint Link](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/ayush_rawal_students_iiit_ac_in/EkQU1bNLbqxOmm7mrkIiJQsBApXQo7ZM7oaT1dk--aU0Bg?e=E1lmad)

## 2. Model Architecture and Training

### CNN Architecture

The implemented convolutional neural network consists of:

- **Input Layer**: 28×28 grayscale images (MNIST format)
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling for dimensionality reduction
- **Fully Connected Layers**: Dense layers for final classification
- **Output Layer**: 10 neurons with softmax activation (digits 0-9)

### Training Configuration

- **Dataset Split**: Standard MNIST train/test split (60,000/10,000)
- **Optimization**: Adam optimizer
- **Loss Function**: Cross-entropy loss
- **Training Framework**: PyTorch
- **Device**: GPU/CPU adaptive training

## 3. Baseline Model Performance

After training the model on the MNIST dataset, we obtained the following performance metrics on the test set:

| Metric             | Value          |
| ------------------ | -------------- |
| **Accuracy**       | 98.89%         |
| **Average Loss**   | 0.0359         |
| **Inference Time** | 1.9792 seconds |

**Confusion Matrix (Baseline Model):**

```
[[ 968    0    4    0    0    0    2    1    5    0]
 [   0 1129    3    0    0    0    0    0    3    0]
 [   0    1 1022    3    1    0    0    4    1    0]
 [   0    0    1 1007    0    1    0    0    1    0]
 [   0    0    0    0  974    0    1    0    1    6]
 [   0    0    0   10    0  877    1    0    3    1]
 [   1    4    1    0    2    2  944    0    4    0]
 [   0    1    8    0    0    0    0 1010    1    8]
 [   0    0    2    3    0    1    0    0  967    1]
 [   0    0    1    5    3    1    0    5    3  991]]
```

The confusion matrix demonstrates excellent classification performance across all digit classes, with minimal misclassifications. The model shows strong baseline performance on clean test data, achieving near-perfect accuracy.

## 4. Static Analysis Security Testing (SAST)

To perform STRIDE threat modeling, we identified the key elements and interactions in our application, which consists of two main components: `server.py` and `train.py`.

### System Elements

**server.py Components:**

1. HTML user interface
2. Flask web application object
3. ML model
4. User input interface

**train.py Components:**

1. MNIST dataset
2. Trained ML model
3. PyTorch training objects

### Component Interactions

**server.py Processes:**

1. User input transformation
2. Loading index.html
3. Loading trained ML model
4. Reading user input
5. Digit classification

**train.py Processes:**

1. Dataset loading
2. Model training
3. Model storage
4. Model evaluation

### STRIDE Analysis Summary

| Component           | **S**poofing | **T**ampering | **R**epudiation | **I**nfo Disclosure | **D**enial of Service | **E**scalation of Privilege |
| ------------------- | ------------ | ------------- | --------------- | ------------------- | --------------------- | --------------------------- |
| **Dataset**         | Low          | **Threat**    | **Threat**      | Low                 | **Threat**            | Low                         |
| **Model**           | Low          | **Threat**    | **Threat**      | Low                 | **Threat**            | Low                         |
| **Python Objects**  | Low          | Low           | Low             | Low                 | Low                   | Low                         |
| **Flask Interface** | Low          | **Threat**    | Low             | Low                 | **Threat**            | Low                         |

**Key Threats Identified:**

- **Tampering**: Model replacement, dataset corruption, interface modification
- **Repudiation**: Lack of authentication and logging mechanisms
- **Denial of Service**: File deletion, resource exhaustion attacks

## 6. Red Team Operations

### 6.1 Data Poisoning Attack

**Methodology:**
We implemented a backdoor attack by injecting trigger patterns into the training dataset. Specifically:

- **Target**: Images labeled as digit "7"
- **Trigger Pattern**: Small colored squares added to corner positions
- **Poisoning Scale**: Approximately 100 samples modified
- **Objective**: Create a backdoor where images containing the trigger pattern are misclassified

### 6.2 Adversarial Attack Methodology

**Fast Gradient Sign Method (FGSM) Implementation:**

- **Technique**: Generated adversarial examples using FGSM algorithm
- **Epsilon Value**: Standard perturbation magnitude for MNIST
- **Library**: PyTorch-based implementation
- **Target**: Baseline trained model
- **Objective**: Demonstrate model vulnerability to imperceptible input modifications

The FGSM attack works by computing gradients of the loss function with respect to input pixels and adding small perturbations in the direction that maximizes the loss, creating visually similar images that fool the classifier.

## 7. Attack Results and Analysis

### 7.1 Performance Comparison Across All Models

| Model Type                | Accuracy | Loss   | Inference Time | Performance Change |
| ------------------------- | -------- | ------ | -------------- | ------------------ |
| **Baseline**              | 98.89%   | 0.0359 | 1.98s          | -                  |
| **Poisoned**              | 97.89%   | 0.0701 | 2.37s          | -1.00% accuracy    |
| **Under FGSM Attack**     | 13.22%   | 5.6666 | 2.68s          | -85.67% accuracy   |
| **Adversarially Trained** | 99.71%   | 0.0075 | 2.60s          | +0.82% accuracy    |

### 7.2 Detailed Analysis

**Poisoned Model Impact:**
The data poisoning attack resulted in a modest but measurable performance degradation, with accuracy dropping by 1% and loss nearly doubling. This demonstrates the subtle but dangerous nature of backdoor attacks.

**Adversarial Attack Devastation:**
The FGSM adversarial attack caused catastrophic performance collapse, reducing accuracy from 98.89% to just 13.22%. The confusion matrix shows complete misclassification across all digit classes, highlighting the extreme vulnerability of neural networks to adversarial perturbations.

### 7.3 Confusion Matrices Analysis

**Poisoned Data Confusion Matrix:**

```
[[ 977    0    0    0    1    0    0    1    1    0]
 [   0 1114    0    1    0    3    3   13    1    0]
 [   5    0 1000    1    3    0    3   12    8    0]
 [   2    0    1  966    0   29    1    6    5    0]
 [   0    0    0    0  979    0    0    1    1    1]
 [   2    0    0    1    0  885    1    1    2    0]
 [   2    2    0    0    4    7  937    0    6    0]
 [   0    0    1    0    0    0    0 1026    1    0]
 [   2    0    0    0    1    1    0    1  967    2]
 [   0    0    0    0   19   11    0   28   13  938]]
```

**Adversarial Attack Confusion Matrix:**

```
[[  3   1 241  39   2  75 143   4 428  44]
 [  0  23  57  34 336   6   7 122 541   9]
 [  0  48 222 470  33   4   5  93 156   1]
 [  0   0  79 409   0 184   0  69 203  66]
 [  0  17  72  23 180  14  14  76 178 408]
 [  4   2  10 374   1 122  37   5 237 100]
 [ 20   9  16  12 108 430 156   0 205   2]
 [  1  22 323 149  19   3   0 140  64 307]
 [  1   7 138 488  32 148  17  16  64  63]
 [  2   2  26 268 236  39   0 111 322   3]]
```

## 8. Blue Team Defense: Adversarial Training

**Defense Strategy:**
We implemented adversarial training by incorporating FGSM-generated adversarial examples into the training dataset. This approach trains the model to be robust against the same type of adversarial perturbations used in the attack.

**Results:**
The adversarially trained model not only recovered from the attack but achieved superior performance:

**Adversarially Trained Model Confusion Matrix:**

```
[[ 979    0    0    0    0    1    0    0    0    0]
 [   2 1131    0    1    1    0    0    0    0    0]
 [   2    0 1029    0    0    0    1    0    0    0]
 [   1    0    0 1008    0    1    0    0    0    0]
 [   1    0    0    0  979    0    0    0    1    1]
 [   1    0    0    0    2  888    0    0    0    1]
 [   0    0    1    0    0    0  957    0    0    0]
 [   3    0    1    0    0    0    0 1023    0    1]
 [   1    0    0    0    0    0    0    1  971    1]
 [   0    0    0    0    0    1    0    0    2 1006]]
```

## 9. Conclusion

This comprehensive security analysis of our MNIST classification system demonstrates the critical importance of considering security throughout the ML development lifecycle.

### Key Findings:

1. **Baseline Vulnerability**: Despite excellent performance (98.89% accuracy), the baseline model was extremely vulnerable to adversarial attacks, with accuracy plummeting to 13.22% under FGSM attack.

2. **Defense Effectiveness**: Adversarial training proved highly effective, not only restoring robustness but improving baseline performance to 99.71% accuracy.

3. **Security Gaps**: SAST analysis revealed code-level vulnerabilities (unsafe model loading, debug mode exposure) that require immediate attention in production deployments.

4. **Threat Landscape**: STRIDE analysis identified multiple attack vectors, particularly around data tampering and denial of service, highlighting the need for comprehensive security controls.
