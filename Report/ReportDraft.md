# Assignment 2 Report: Secure AI systems: Red and Blue teaming an MNIST Classifier

# Application description
For the assignment we have created a handwritten application in python that trains a convolutional neural network model on MNIST dataset and uses the trained model to classify a digit drawn in user input interface. 
train.py trains the model and performs model evaluation.
server.py uses the trained model to classify digits drawn by the user on the screen.

The link to github repository for the code is : https://github.com/AyushRawal/tse-assignment2

# Model performance

After running the trained model on a subset of the dataset as Test data we obrained the following metrics.

```bash
base-model - orig data
=== Test Dataset Evaluation ===
Accuracy: 99.11%
Average Loss: 0.0269
Inference Time: 1.9021 seconds
Confusion Matrix:
[[ 978    0    0    0    0    0    0    1    1    0]
 [   0 1131    0    3    0    0    0    0    1    0]
 [   4    1 1023    1    1    0    0    2    0    0]
 [   0    0    0 1008    0    1    0    0    1    0]
 [   0    1    1    0  974    0    2    1    0    3]
 [   1    0    0    4    0  881    1    1    2    2]
 [   4    2    0    0    1    2  948    0    1    0]
 [   0    4    4    2    0    0    0 1012    1    5]
 [   5    0    1    0    0    1    0    0  966    1]
 [   0    1    0    0   10    2    0    2    4  990]]
```
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

For train.py

```bash
./pyVenv/bin/bandit -r train.py
[main]	INFO	profile include tests: None
[main]	INFO	profile exclude tests: None
[main]	INFO	cli include tests: None
[main]	INFO	cli exclude tests: None
[main]	INFO	running on Python 3.12.3
Run started:2025-10-01 13:22:51.438680

Test results:
	No issues identified.

Code scanned:
	Total lines of code: 55
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

For server.py

```bash
./pyVenv/bin/bandit -r server.py
[main]	INFO	profile include tests: None
[main]	INFO	profile exclude tests: None
[main]	INFO	cli include tests: None
[main]	INFO	cli exclude tests: None
[main]	INFO	running on Python 3.12.3
Run started:2025-10-01 13:23:02.977840

Test results:
>> Issue: [B614:pytorch_load] Use of unsafe PyTorch load
   Severity: Medium   Confidence: High
   CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b614_pytorch_load.html
   Location: ./server.py:38:22
37	model = MNIST_CNN().to(device)
38	model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
39	model.eval()  # set to evaluation mode

--------------------------------------------------
>> Issue: [B201:flask_debug_true] A Flask app appears to be run with debug=True, which exposes the Werkzeug debugger and allows the execution of arbitrary code.
   Severity: High   Confidence: Medium
   CWE: CWE-94 (https://cwe.mitre.org/data/definitions/94.html)
   More Info: https://bandit.readthedocs.io/en/1.8.6/plugins/b201_flask_debug_true.html
   Location: ./server.py:83:4
82	if __name__ == '__main__':
83	    app.run(debug=True)

--------------------------------------------------

Code scanned:
	Total lines of code: 61
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
```

## actions taken on the basis of SAST reports


# Poisoned dataset

method of generation of poisoned dataset


The link to code files for the adversarial and posoned data set generation is :  ???

# Metrics and comparison after poisoning
```bash
base-model - poisoned data

=== Test Dataset Evaluation ===
Accuracy: 99.11%
Average Loss: 0.0270
Inference Time: 2.7264 seconds
Confusion Matrix:
[[ 978    0    0    0    0    0    0    1    1    0]
 [   0 1131    0    3    0    0    0    0    1    0]
 [   4    1 1023    1    1    0    0    2    0    0]
 [   0    0    0 1008    0    1    0    0    1    0]
 [   0    1    1    0  974    0    2    1    0    3]
 [   1    0    0    4    0  881    1    1    2    2]
 [   4    2    0    0    1    2  948    0    1    0]
 [   0    4    4    2    0    0    0 1012    1    5]
 [   5    0    1    0    0    1    0    0  966    1]
 [   0    1    0    0   10    2    0    2    4  990]]
```
# Results obtained after Training on adversarial dataset

```bash
base-model - adversarial data
=== Test Dataset Evaluation ===
Accuracy: 85.74%
Average Loss: 0.4985
Inference Time: 2.6439 seconds
Confusion Matrix:
[[ 926    1    6    2    4    7   16    3    3   12]
 [   4 1085    5    7    2    2   11    3   14    2]
 [  15   16  881   26    6    0    6   43   38    1]
 [   1    0   22  916    0   30    0    6   27    8]
 [   2   18    4    1  817    2   19   14   30   75]
 [   7    4    1   77    0  692   16    1   40   54]
 [  21    5    5    0   20   40  861    0    6    0]
 [   2   26   32   37    4    1    0  806   13  107]
 [  16    5   22   19   12   29   14    6  839   12]
 [   4   10    0   14  117   17    0   16   80  751]]
```
 

