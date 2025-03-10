# Transfer Learning Neyman-Pearson
## Introduction
This library provides a framework for training a machine learning model to solve the Transfer Learning Neyman-Pearson (TLNP) problem. 

TLNP is required under the following conditions:
1. The problem is a binary classification problem with inequal classes, which we term the "normal" and "abnormal" data. For example, the normal data could be days with regular weather, and the abnormal data could be days with a blizzard; the normal data could be a creditor not defaulting on their debt, and the abnormal data could be a default. The normal dataset is always significantly larger than the abnormal dataset.
2. There are two datasets with similar data, one that we're interested in (the "target" dataset), and one with similar, relevant data (the "source" dataset). The target dataset has few data points in the abnormal dataset, and the source dataset has more data points and is used to supplement the target problem. For example, the target dataset could be a location where blizzards are infrequent, and the source dataset could be a location where blizzards are a common occurrence. Note that this definition of transfer learning involves the transfer of information from one dataset to another dataset -- this is distinct from the more common definition of transfer learning, whereby a model that is previously trained one dataset is then trained on a second dataset. In TLNP, both datasets are used in the initial training.
3. Under normal conditions, a machine learning model does not distinguish between these two types of errors and simply minimizes overall error. In Neyman-Pearson, we are interested in training a model that maintains a maximum Type-I error (false positives) while minimizing Type-II error (false negatives). This has many practical use cases. For example, if we are training a model to predict faulty widgets in a production line, we may have an appetite to misidentify 5\% of normal widgets as faulty, while minimizing the number of faulty widgets that are misidentified as normal.

## Approaches
While the primary purpose of this library is to publish our TLNP framework, we also provide a framework for other approaches that can be used for the same problem: 
1. 

## Framework
TLNP is a model-agnostic framework. Our algorithm is compatible with any model that makes a binary prediction. Here, we provide examples using a Quadratic Form Model (used on the generated data) and a Multilayer Perceptron (used on the real datasets).