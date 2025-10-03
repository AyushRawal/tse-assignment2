# Assignment 2 Report: Secure AI Systems - Red and Blue Teaming an MNIST Classifier

## Team Members

- Ayush Rawal (2024201036)
- Rishabh Sahu (2024201037)

## Summary

This report presents a comprehensive security analysis of an MNIST digit classification system through red team and blue team exercises. We developed a CNN-based application, identified vulnerabilities through STRIDE threat modeling and SAST analysis, executed data poisoning and adversarial attacks, and implemented robust defense mechanisms through adversarial training.

## 1. Application Description

For this assignment, we developed a handwritten digit classification application in Python that trains a convolutional neural network (CNN) model on the MNIST dataset and provides a web interface for real-time digit classification.

**Key Components:**

- `train.py`: Handles model training, evaluation, and performance metrics generation
- `server.py`: Flask web application that serves the trained model for digit classification via a drawing interface

**Repository Links:**

- [GitHub Repository](https://github.com/AyushRawal/tse-assignment2)
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

## 4. STRIDE Threat Modeling

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

### Element Wise Threat Modelling

#### Dataset

1. **Spoofing identity**: The source dataset is imported from publicly available verified source on Kaggle and not susceptible to identity spoofing.
2. **Tampering with data**: An attacker can potentially replace the offline dataset file that is being used to train the models by replacing the dataset file with malicious dataset file of the same name.
   To prevent it ,we can check dataset integrity through file hashes.
3. **Repudiation**: Since we have not implemented any authentication mechanism in our application for training the models , in a group of persons using our applicaiton it is not possible to pinpoint the issues to a single person in case one of the users trains the model on malicious data.
   Implementing User authentication can prevent this.
4. **Information disclosure**: Only publicly available data is used and stored and no user secrets are used or stored so negligible risk of information disclosure.
5. **Denial of Service**: In some cases attackers can execute a script to delete the dataset file repeatedly , thus causing the model training code to run continuously and be unable to finish training the model in reasonable time , this is possible because the dataset file after downloading is not stored in a protected environment.
6. **Escalation of privilege**: No sensitive privileged operations are carried out in our applicaiton so escalation of privilege threat is low.

#### Model

1. **Spoofing identity**: No user identity based interactions so negligible threat from identity spoofing.
2. **Tampering with data**: Attackers can train the model on malicious data and replace the existing model with malicious one. Method to check model integrity and authentication will be helpful to prevent this.
3. **Repudiation**: Since there is no ownership and no authentication and no separation of privileges so can't easily catch attackers who use the system maliciously, for example running endless instances of the application to slow down the system.
4. **Information disclosure**: No secret or private information is stored so information disclosure threat is low.
5. **Denial of Service**: Access to the ML model can be restricted if the model files are deleted or edited while the application is uninformed. This is because we are not storing these files in protected locations.
6. **Escalation of privilege**: Neglible threat.

#### Python objects

The python objects that we used to train ML models and implement are not susceptible to spoofing, data tampering, repudiation, information disclosure, denial of service and escalation of privileges since these objects are not exposed to external entities.

#### Flask web application and user interface

The user interface could be affected if an attacker somehow replaces the `index.html` page stored locally with another file of the same name. This is related to Denial of service and Tampering of data. To avoid this `index.html` can instead be stored and fetched from protected location.

### Tabular description

|                      | Sppofing | Tampering | Repudiation | Information Disclosure | Denial of service | Escalation of privilege |
| -------------------- | -------- | --------- | ----------- | ---------------------- | ----------------- | ----------------------- |
| Dataset              |          | Threat    | Threat      |                        | Threat            |                         |
| Model                |          | Threat    | Threat      |                        | Threat            |                         |
| Python object        |          |           |             |                        |                   |                         |
| Flask user interface |          | Threat    |             |                        | Threat            |                         |

### STRIDE Analysis Summary

**Key Threats Identified:**

- **Tampering**: Model replacement, dataset corruption, interface modification
- **Repudiation**: Lack of authentication and logging mechanisms
- **Denial of Service**: File deletion, resource exhaustion attacks

## 5. Static Analysis Security Testing (SAST)

We performed static analysis security testing using the Python "bandit" tool (https://bandit.readthedocs.io/en/latest/). Bandit performs static code analysis by generating Abstract Syntax Trees (AST) from code files and running security-focused plugins against the AST nodes.

### SAST Results Summary

| File | Issues Found | Severity Breakdown |
|------|-------------|-------------------|
| `server.py` | 2 issues | 1 High, 1 Medium |
| `train.py` | 0 issues | Clean |

**Detailed Findings:**

```console
$ bandit server.py
[main]  INFO    profile include tests: None
[main]  INFO    profile exclude tests: None
[main]  INFO    cli include tests: None
[main]  INFO    cli exclude tests: None
[main]  INFO    running on Python 3.13.7
Run started:2025-10-03 18:03:52.366701

Test results:
>> Issue: [B614:pytorch_load] Use of unsafe PyTorch load
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b614_pytorch_load.html
   Location: ./server.py:40:22
39      model = MNIST_CNN().to(device)
40      model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
41      model.eval()  # set to evaluation mode

--------------------------------------------------
>> Issue: [B201:flask_debug_true] A Flask app appears to be run with debug=True, which exposes the Werkzeug debugger and allows the execution of arbitrary code.
   Severity: High   Confidence: Medium
   CWE: CWE-94 (https://cwe.mitre.org/data/definitions/94.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b201_flask_debug_true.html
   Location: ./server.py:90:4
89      if __name__ == "__main__":
90          app.run(debug=True)

--------------------------------------------------

Code scanned:
        Total lines of code: 63
        Total lines skipped (#nosec): 0

Run metrics:
        Total issues (by severity):
                Undefined: 0
                Low: 0
                Medium: 1
                High: 1
        Total issues (by confidence):
                Undefined: 0
                Low: 0
                Medium: 1
                High: 1
Files skipped (0):

$ bandit train.py
[main]  INFO    profile include tests: None
[main]  INFO    profile exclude tests: None
[main]  INFO    cli include tests: None
[main]  INFO    cli exclude tests: None
[main]  INFO    running on Python 3.13.7
Run started:2025-10-03 18:03:58.242388

Test results:
        No issues identified.

Code scanned:
        Total lines of code: 115
        Total lines skipped (#nosec): 0

Run metrics:
        Total issues (by severity):
                Undefined: 0
                Low: 0
                Medium: 0
                High: 0
        Total issues (by confidence):
                Undefined: 0
                Low: 0
                Medium: 0
                High: 0
Files skipped (0):
```

### Actions Taken Based on SAST Findings

| Vulnerability | Severity | Mitigation Applied |
|---------------|----------|-------------------|
| **Unsafe PyTorch Load (B614)** | Medium | Ensured model files from trusted sources; considering `weights_only=True` parameter |
| **Flask Debug Mode (B201)** | High | Set `debug=False` for production; debug mode only during development |

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
- **Epsilon Value**: 0.1
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
