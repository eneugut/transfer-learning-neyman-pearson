# Transfer Learning Neyman-Pearson
## Introduction
This library provides a framework for training a machine learning model to solve the Transfer Learning Neyman-Pearson (TLNP) problem. This is an implementation of the paper [Transfer Neyman-Pearson Algorithm for Outlier Detection](https://arxiv.org/pdf/2501.01525).

TLNP is necessitated under the following conditions:
1. The problem is a binary classification problem with inequal classes, which we term the "normal" and "abnormal" data. For example, the normal data could be days with regular weather, and the abnormal data could be days with a blizzard; the normal data could be a creditor not defaulting on their debt, and the abnormal data could be a default. The normal dataset is always significantly larger than the abnormal dataset.

2. There are two datasets with similar data, one that we're interested in (the "target" dataset), and one with similar, relevant data (the "source" dataset). The target dataset has few data points in the abnormal dataset, and the source dataset has more abnormal data points and is used to supplement the target problem. For example, the target dataset could be a location where blizzards are infrequent, and the source dataset could be a location where blizzards are a common occurrence. Generally, TLNP will outperform the naive approach when there are 100 or fewer target abnormal data points. Note that this definition of transfer learning involves the transfer of information from one dataset to another dataset — this is distinct from the more common definition of transfer learning, whereby a model that is previously trained on one dataset is then retrained on a second dataset. In TLNP, both datasets are used in the initial training.

3. Under normal conditions, a machine learning model does not distinguish between Type-I and Type-II Errors during training and simply minimizes overall error. In Neyman-Pearson, we are interested in training a model minimizes Type-II Error (false negatives) while maintaining a maximum Type-I Error (false positives). This has many practical use cases. For example, if we are training a model to predict faulty widgets in a production line, we may have an appetite to misidentify 5\% of normal widgets as faulty, while minimizing the number of faulty widgets that are misidentified as normal. Because there is a trade-off between Type-I and Type-II Errors, the ideal model will achieve a Type-I Error close to the upperbound. 

## Approaches
The main approach in this library is our Transfer Learning Neyman Pearson framework (TLNP). More info on how this works below. 

As an alternative, we also include implementation for Naive Neyman Pearson (NNP), which is simply a regular approach to training (without differentiating between Type-I and Type-II Errors during training); after training, we adjust the cut-off point to achieve the desired Type-I Error. We recommend testing both approaches. Note that if `source_abnormal_data` is provided in the `data_dict` to NNP, it will pool the `target_abnormal_data` and `source_abnormal_data` together for training. If you want to run the `target_abnormal_data` alone, simply do not include the `source_abnormal_data` key.

## Use
The two approaches can be run using the functions `run_tlnp` and `run_naive_np`, which both have the following arguments:

### `config_path`
The file path of the .yaml file. Please see our example, which includes comments.

### `model`
TLNP is a model-agnostic _framework_. To use this framework, you must supply your own model. The framework is compatible with any PyTorch model with one output.

### `data_dict`
A dict object that contains the data used during training and testing. It should contain the following keys:
- `target_normal_data`
- `target_abnormal_data`
- `source_abnormal_data` (optional - TLNP will run with only target data if no source data is provided)
- `target_normal_test_data` (optional - only evaluated once at the conclusion of training)
- `target_abnormal_test_data` (optional - same as above)

The value for each key should be a 2-dimensional tensor. **_These tensors should not contain the targets/labels_**. TLNP already assumes that the normal data is labeled 0 and the abnormal data is labeled 1. If your tensors include the targets, then the model will receive the targets as input. 

All of the tensors should consist of the same number of columns. 

The split between training and test data must be performed _before_ creating the `data_dict` and running TLNP. The split between training and validation data is performed during training by TLNP and can be adjusted via the `validation_split` parameter in the config. The test data tensors are optional — if they are not provided, then simply test evaluation will not be performed at the conclusion of training. 

### `loss_function`
We provide implementation for three loss functions: exponential loss, logistic loss, and hinge loss. To use one of the existing loss functions, which we highly recommend, simply specify the name in the `loss_config` and do not provide anything to the `loss_function` argument. 

The `loss_config` additionally includes a parameter to `normalize_losses`, which will divide the loss for each of the three classes by the number of points in the respective class. The parameter `clip_value` is used in the exponential and logistic loss functions to prevent overflow.

We also have support for a custom loss function by providing a function to the `loss_function` argument. Any custom loss function must have the following arguments:
- `y_pred`: Tensor with the predictions.
- `label`: Tensor with the "labels" for the predictions. A label of 0 corresponds to the normal data; 1 corresponds to the target abnormal data; and 2 corresponds to the source abnormal data. (Note that we differentiate between the target abnormal and source abnormal because the losses are weighted separately.)
- `lambda_normal`: Weight applied to the loss from the normal predictions. The ground-truth for normal points is 0; hence, predictions further away from 0 should be more heavily penalized.
- `lambda_target`: Weight applied to the loss from the target abnormal predictions. The ground-truth is 1.
- `lambda_source`: Weight applied to the loss from the source abnormal predictions. The ground-truth is 1.

### `optimizer` & `scheduler`
We provide implementation for three standard optimizers (SGD, Adam, and RMSprop) and three standard learning rate schedulers (StepLR, ExponentialLR, and ReduceLROnPlateau). To use one of our existing optimizers or schedulers, just include the `optimizer_config` and `scheduler_config` in the config file. You can also supply your own optimizer and scheduler to the `optimizer` and `scheduler` arguments, provided they are compatible with the `torch.optim` library. The scheduler is fully optional — if you exclude the `scheduler_config` and don't provide your own scheduler, the training will run without one.

## TLNP Methodology
For the theoretical underpinnings, please [see our paper here](https://arxiv.org/pdf/2501.01525). From a practical perspective:

TLNP relies on a loss function that applies separate weights to the target normal (λ<sub>0</sub>), target abnormal (λ<sub>T</sub>), and source abnormal data (λ<sub>S</sub>). We fix λ<sub>T</sub> at 1 and search over λ<sub>0</sub> and λ<sub>S</sub>, although, equivalently, one could fix λ<sub>S</sub> and search over λ<sub>0</sub> and λ<sub>T</sub>.

We train multiple models of the data using different pairs of (λ<sub>S</sub>, λ<sub>0</sub>), starting with (0, 1). (When λ<sub>S</sub> is 0, effectively the model is not taking into account the source data.)

For each lambda pair, we first train the model with the loss function using the lambdas. After training is complete, we then evaluate the trained model on the entire training set (the evaluation stage) and store the Type-I Error, Type-II (Target Abnormal), and Type-II Error (Source Abnormal).

If the Type-I Error is above the Type-I Error upperbound (as specified in the config), then we increase the λ<sub>0</sub> and repeat the process. If the Type-I Error is below the Type-I Error lowerbound (as specified in the config or calculated by TLNP), then we decrease the λ<sub>0</sub> and repeat the process. If the Type-I Error falls within the lowerbound and upperbound, then we move onto the next λ<sub>S</sub> in the `lambda_source_list` (as specified in the config). 

Effectively, for each λ<sub>S</sub> in the `lambda_source_list`, we are searching for a corresponding λ<sub>0</sub> that results in Evaluation Type-I within the lowerbound and upperbound. (For a given λ<sub>S</sub>, we cap the number of attempts at finding λ<sub>N</sub> at `max_tuning_tries` in the config. If this λ<sub>S</sub> never converges, then we just move onto the next λ<sub>S</sub> in the list. However, if all of the λ<sub>S</sub> in the list have been attempted and fewer than 5 λ<sub>S</sub>'s have converged, then we add more points to the list by multiplying the greatest λ<sub>S</sub> by 10 and dividing the least λ<sub>S</sub> by 10 and continue the process until at least 5 λ<sub>S</sub>'s have converged.)

Once this part is complete, then we have to choose one of the lambda pair's trained models as the final model. This is done by the following methodology: 
1. First, points with Evaluation Type-I Error above the Type-I error upperbound are filtered out.
2. Next, of the remaining points, we identify the point with the lowest Evaluation Type-II Error (Target). We sum this Evaluation Type-II Error (Target) with constant/sqrt(n_T) to create an Evaluation Type-II Error (Target) upperbound. Points with an Evaluation Type-II Error (Target) above this threshold are filtered out. 
3. Of the remaining points, the point with the least Evaluation Type-II Error (Source) is chosen as the final point. 

After the final point is chosen, the test data (if included in the `data_dict`) is evaluated on the point's model, and the model is saved. Note that the test data is only evaluated _once_ by the final point and not during the lambda pair evaluation, which uses the training data. 

If no source abnormal data is included in the `data_dict`, then the algorithm will fix λ<sub>S</sub> at 0 and search over λ<sub>0</sub>. The point with Type-I Error between the lowerbound and upperbound is chosen as the final point. 

## Errors
### `"ValueError: Dataset 'X' has Y columns, expected Z columns."`
All tensors in the `data_dict` must have the same number of columns (input features).

### `"TypeError: '>' not supported between instances of 'str' and 'int'"`
This error occurs when `lambda_limit` in the config cannot be parsed as scientific notation. `lambda_limit` should be in the format `1.0e+6`, where the first number is a float.

### `"TypeError: '>' not supported between instances of 'float' and 'str'"`
This error occurs when `eps` in the config cannot be parsed as scientific notation. `eps` should be in the format `1.0e-8`, where the first number is a float.

### `"ValueError: Type 1 error range [X, Y] is not possible with Z normal samples."`
It is not mathematically possible to achieve a Type-I Error between the lowerbound and upperbound with the given number of samples. For example, if there are 100 target normal data points, the lowerbound is set to 0.025, and the upperbound is set to 0.075, there is no number of incorrectly classified normal points that will result in a Type-I Error within this range. Either lower the lowerbound or raise the upperbound.

### `"UserWarning: Type 1 error range [X, Y] has only Z possible values: W."`
Similar to above. The Type-I Error range can mathematically be achieved, but only 3 or fewer amounts of misclassifications will result in a Type-I Error within this range. For example, if there are 100 target normal data points, the lowerbound is set to 0.025, and the upperbound is set to 0.05, then only 3, 4, or 5 misclassifications will result in a Type-I Error within this range. While TLNP will run, it may struggle to find a suitable model in this case.

### `"ValueError: No suitable point could be found."`
This error can occur when training without source data. This means that, after tuning λ<sub>0</sub> with λ<sub>S</sub> fixed at 0, TLNP could not find a point with Type-I Error below the upperbound. Try raising the upperbound. 
