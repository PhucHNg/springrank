# Implementation of SpringRank model

SpringRank is a ranking model for weighted, directed networks. The model returns real-valued scores which can be used to order nodes in a network.
Read more about how SpringRank works here: https://arxiv.org/abs/1709.09002

This is an implementation of the model I used to conduct some experiments on the model as part of my summer research.

## How to use

At the moment, all functions are stored in rank.py.
You can download rank.py and import it as a module into your code file. You need to have `numpy`, `scipy`, `sklearn`, `matplotlib` and `networkx` installed.

## Example

```
import rank
import numpy as np

# Encode a network in an adjacency matrix 
A = np.array([[0,1,0],[0,0,1],[0,1,0]])

# Infer a ranking
scores = rank.spring_rank(A)
```

## Note

This is work in progress so it might contain bugs and some functions I use for experiments that are not in the paper.
