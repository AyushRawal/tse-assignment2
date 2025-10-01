# step1 : defining and decomposing assets

## components
server.py 
loads the trained model , a web page ui , loads user input and displays it and gives prediction outputs

train.py
trains and stores the model ,interacts with dataset


**elements**

May need to add diagrams
server.py
- loaded model
- html page ui
- flask web app
- interface to draw digit in app
train.py
- loaded dataset
- model

**interactions and tasks**
server.py
- transformations of user input
- loading of index.html
- loads the trained model
- reads user input for detecting digit
train.py
- loads the dataset
- trains a model
- stores trained model
  
**trust boundaries**


## list of stride threats
1. spoofing identity : 
2. tampering with data:
3. repudiation:
4. Information disclosure:
5. Denial of Service:
6. Escalation of privilege:

look for threats through 
1.stride per element and 
2.stride per interaction (interaction between components)

# step2  : know your attackers , tactics and techniques

# step 3 : gain introspection into your systems to find vulnerabilities

## stride per element

dataset
1. spoofing identity : 🟩  since the source url of data is downloaded from publicly available verified source identity spoofing is not a threat for dataset  .
       
2. tampering with data:  🟥An attacker can potentially change the dataset file that is being used to train the models by replacing the dataset with a file of same name.
   So as a security measure we can check data integrity through hash functions.
3. repudiation: 🟥 we have not implemented any authentication mechanism for persons to train the data and run the model so if more than one people work on the application then repudiation could possibly occur and if one of the persons working on the code is an attacker ,it may not be possible to catch them .

4. Information disclosure: 🟩 Only publicly available data is used and stored so risks from any possible information disclosure is negligible.
5. Denial of Service: 🟧 in some cases attackers could continuosly execute script to delete the dataset file before download completes or before model could train on it to cause denial of service. 

     escaping denial of service from code is hard and it is much better to keep the system access in trustworthy environment.
6. Escalation of privilege: 🟩  low risk since no separation of privileges at present but separation of privileges may be required to contain other threats

model
1. spoofing: 🟥 inject a fake model file with the same name as expected model file which will fail to load or give wrong predictions
2. tampering: 🟥 attackers can train model on malicious data and replace the existing model with malicious one .
       some method to check model integrity  and authentication will be helpful
3. repudiation : 🟧 since no ownership and no authentication and no separation of privileges so can't easily catch attackers who use the system maliciously , for example running endless instances of the application to slow down the system.
4. information disclosure: 🟩 no private information so neglible threat, even the purpose of model is known already with no sensitive information
5. denial of service: 🟧  restricted access to model in case model files are deleted or renamed while system is uninformed  . 
6. escalation of privileges : 🟩 low or no threat

# stride per interaction




# references
https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats

https://drata.com/grc-central/risk/guide-stride-threat-model