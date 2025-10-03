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

## Element wise modelling

## Tabular description

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
# Training on adversarial dataset

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
 

