# ReadMe

This is a repo for using reinforcement learning to perform neural architecture search

## Usage

### Setup

1. Install python>=3.8
2. Install packages in requirements.txt

### Configuration

1. Change the two dictionaries, START and LAYER. 
2. In START, "mult" means the number of multipliers, the default number is 10. 
3. In Layer is DNN configurations for each layer.
4. Please be sure to save each file after modification.

### Running

```
python main.py
```

### Runtime arguments

```
python main.py -l 6 -ep 200 -e 100 -lr 0.1 -s 0
```

1. -l or --layers : number of layers.
2. -ep or --episodes : number of designs to try.
3. -e or --epochs : number of epochs in training each model.
4. -lr or learning_rate : learning rate to train the RL controller.
5. -s or --seed : random seed.