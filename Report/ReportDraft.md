# Assignment 2 Report: Secure AI systems: Red and Blue teaming an MNIST Classifier

# Application description
For the assignment we have created a handwritten application in python that trains a convolutional neural network model on MNIST dataset and uses the trained model to classify a digit drawn in user input interface. 
train.py trains the model and performs model evaluation.
server.py uses the trained model to classify digits drawn by the user on the screen.

The link to github repository for the code is : https://github.com/AyushRawal/tse-assignment2
[Link to generated datasets](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/ayush_rawal_students_iiit_ac_in/EkQU1bNLbqxOmm7mrkIiJQsBApXQo7ZM7oaT1dk--aU0Bg?e=E1lmad)

# Model performance

After running the trained model on a subset of the dataset as Test data we obrained the following metrics.

Accuracy: 98.89%
Average Loss: 0.0359
Inference Time: 1.9792 seconds

Confusion Matrix:
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

The confusion matrix shows excellent performance across all digit classes, with minimal misclassifications. The model demonstrates strong baseline performance on clean test data.

# STRIDE threat modelling
In order to perform stride threat modelling , first we identified the elements and the interactions in our application The application has two main code files server.py and train.py
## Elements 
The elements in the application are
server.py
1. HTML page user interface
2. Flask web application object
3. ML model
4. User interface to accept input

train.py
1. Imported MNIST dataset
2. Trained ML model
3. Pytorch object to train the model

## Interactions between components and functions
server.py
1. Transformation of user input
2. Loading index.html
3. Loading trained ML model
4. Reading user input
5. Classification of user input as digits

train.py
1. Loading of dataset
2. Training of model
3. Storage of trained model
4. Evaluation of trained model on test data

## Element wise threat modelling
### Dataset

1. Spoofing identity : The source dataset is imported from publicly available verified source on Kaggle and not susceptible to identity spoofing.
2. Tampering with data: An attacker can potentially replace the offline dataset file that is being used to train the models by replacing the dataset file with malicious dataset file of the same name.
To prevent it ,we can check dataset integrity through file hashes.
3. Repudiation: Since we have not implemented any authentication mechanism in our application for training the models , in a group of persons using our applicaiton it is not possible to pinpoint the issues to a single person in case one of the users trains the model on malicious data.
Implementing User authentication can prevent this.
4. Information disclosure: Only publicly available data is used and stored and no user secrets are used or stored so negligible risk of information disclosure.
5. Denial of Service: In some cases attackers can execute a script to delete the dataset file repeatedly , thus causing the model training code to run continuously and be unable to finish training the model in reasonable time , this is possible because the dataset file after downloading is not stored in a protected environment.
6. Escalation of privilege: No sensitive privileged operations are carried out in our applicaiton so escalation of privilege threat is low.

### Model

1. Spoofing identity : No user identity based interactions so neglble threat from identity spoofing.
2. Tampering with data: Attackers can train the model on malicious data and replace the existing model with malicious one .
   Method to check model integrity  and authentication will be helpful to prevent this
3. Repudiation: Since there is no ownership and no authentication and no separation of privileges so can't easily catch attackers who use the system maliciously , for example running endless instances of the application to slow down the system.
4. Information disclosure: No secret or private information is stored so information disclosure threat is low.
5. Denial of Service: Access to the ML model can be restricted if the model files are deleted or edited while the application is uninformed. This is because we are not storing these files in protected locations.
6. Escalation of privilege: Neglible threat.


### Python objects
The python objects that we used to train ML models and implement th are not susceptible to spoofing , data tampering , repudiation, information disclosure , denial of service and escalation of privileges since these objects are not exposed to external entities .

### Flask web application and user interface
The user interface could be affected if an attacker somehow replaces the index.html page stored locally with another file of the same name. This is related to Denial of service and Tampering of data. To avoid this index.html can instead be stored and fetched from protected location.



## Tabular description

|                      | Sppofing | Tampering  | Repudiation | Information Disclosure | Denial of service | Escalation of privilege |
|----------------------|----------|------------|-------------|------------------------|-------------------|-------------------------|
| Dataset              |          |Threat      |Threat       |                        |Threat             |                         |
| Model                |          |Threat      |Threat       |                        |Threat             |                         |
| Python object        |          |            |             |                        |                   |                         |
| Flask user interface |          |Threat      |             |                        |Threat             |                         |

# Static analysis security testing 

In order to perform SAST security testing we used the python "bandit" tool. (https://bandit.readthedocs.io/en/latest/)  It performs static code analysis by generating AST from each code files and running appropriate plugins against the AST nodes. 

The results of running the tool on both the python code files gave the following results.

```console
❯ bandit server.py
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

❯ bandit train.py
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

## actions taken on the basis of SAST reports

Based on the SAST findings, two security vulnerabilities were identified:

1. **Medium Severity - Unsafe PyTorch Load (B614)**: The model loading uses `torch.load()` which can execute arbitrary code. This was addressed by ensuring model files are from trusted sources and considering using `torch.load()` with `weights_only=True` parameter for additional safety.

2. **High Severity - Flask Debug Mode (B201)**: Running Flask with `debug=True` in production exposes the Werkzeug debugger. This was mitigated by setting `debug=False` for production deployment and only using debug mode during development.


# Poisoned dataset

method of generation of poisoned dataset

We implemented data poisoning by adding small colored squares to corner positions of images labeled as digit "7" in the training dataset. Approximately 100 samples were modified with this trigger pattern. The poisoning was designed to create a backdoor where the model would misclassify images containing the trigger pattern.


# Metrics and comparison after poisoning

Performance comparison between clean and poisoned models:

**Clean Model:**
- Accuracy: 98.89%
- Loss: 0.0359
- Inference Time: 1.9792s

**Poisoned Model:**
- Accuracy: 97.89% (-1.00%)
- Loss: 0.0701 (+95.3% increase)
- Inference Time: 2.3674s (+19.6% increase)

Confusion Matrix (Poisoned Model):
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

The poisoned model shows degraded performance with reduced accuracy and increased loss, demonstrating the impact of data poisoning on model reliability.

# Results obtained after Training on adversarial dataset

**Adversarial Attack Results:**
When tested against FGSM adversarial examples, the baseline model's performance dropped dramatically:
- Accuracy: 13.22% (85.67% decrease)
- Loss: 5.6666 (157.8x increase)
- Inference Time: 2.6802s

Confusion Matrix (Adversarial Attack):
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

**Adversarial Training Defense:**
After retraining with adversarial examples included in the training set:
- Accuracy: 99.71% (+0.82% improvement over baseline)
- Loss: 0.0075 (-79.1% improvement over baseline)  
- Inference Time: 2.6034s

Confusion Matrix (Adversarially Trained Model):
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

The adversarially trained model demonstrates excellent robustness, achieving even better performance than the original baseline while maintaining resistance to adversarial attacks. This validates the effectiveness of adversarial training as a defense mechanism.

