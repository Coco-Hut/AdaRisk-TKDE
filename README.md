# AdaRisk

This is the official PyTorch implementation of [AdaRisk: Risk-adaptive Deep Reinforcement
Learning for Vulnerable Nodes Detection]()

## Dependencies
Python 3.9.17, PyTorch(1.12.0)

Other dependencies can be installed via 

  ```pip install -r requirements.txt```

## Configuration

Change the parameter setting in config.py

## Train

* To run the training of experiment and save the trained model:

  ```python main.py```

## Test

* To load the trained model and run the testing of experiment:

  ```python eval.py```