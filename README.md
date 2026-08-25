Federated IDS can allow multiple networks to collaboratively train and enhance their detection system without sharing any raw data between themselves. However, compromised clients can send malicious weights to aggregation, which falters the training. Robust aggregation can be used
to solve this problem*, however it is not always possible to differentiate between honest client updates and malicious client updates (like Byzantine attacks). Our idea is to propose a temporal reputation mechanism for clients during aggregation, which helps us to determine whether the client is truly malicious or honest.

# Problem Statement :

Can a temporal reputation mechanism during aggregation be used to differentiate between honest client updates and malicious client updates in a Federated Intrusion Detection System in a cross-silo network with a highly non-IID dataset?

# Working Hypothesis :

In a cross-silo Federated Intrusion Detection System with a highly non-IID dataset, evaluating each client using a temporal reputation mechanism can differentiate between honest client updates and malicious updates (e.g Byzantine attacks).

## Assumptions :

### Server

1. Trusted
2. Performs aggregation correctly
3. Not compromised

### Clients

1. Honest
- Generates honest Update
- Trains local data

2. Byzantine
- Generates malicious update

## Attacks :

1. Sign Flipping
2. Random update
3. Label flipping

## Out of Scope :
1. Malicious Server
2. Compromised IoT Hardware
3. Privacy Inference Attacks
4. Communication interception Attacks
5. Backdoor Attacks
6. Colluding Byzantine clients