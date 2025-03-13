import torch
import torch.nn as nn

from tlnp.runner import TLNP

def sample_multivariate_normal(num_features, mean_value, num_samples):
    cov_matrix = torch.eye(num_features)    # Identity covariance matrix
    mean_vector = torch.full((num_features,), mean_value)
    dist = torch.distributions.MultivariateNormal(mean_vector, covariance_matrix=cov_matrix)
    return dist.sample((num_samples,))

def train_test_split(tensor, test_ratio=0.1):
    test_size = int(test_ratio * tensor.size(0))
    train_tensor = tensor[:-test_size]
    test_tensor = tensor[-test_size:]
    return train_tensor, test_tensor

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random tensors
    num_features = 15                                                            # 15 input dimensions
    target_normal_data = sample_multivariate_normal(num_features, 0.0, 4500)     # Mean 0
    source_abnormal_data = sample_multivariate_normal(num_features, 0.55, 3000)   # Mean 0.55
    target_abnormal_data = sample_multivariate_normal(num_features, 0.60, 50)    # Mean 0.50
    
    # Split the data into training and testing sets
    target_normal_train, target_normal_test = train_test_split(target_normal_data)
    target_abnormal_train, target_abnormal_test = train_test_split(target_abnormal_data)
    
    # Set the config_path
    config_path = "tlnp/example_config.yaml"
    
    # Define the data dict
    data_dict = {
        "target_normal_data": target_normal_train,
        "target_abnormal_data": target_abnormal_train,
        "source_abnormal_data": source_abnormal_data,
        "target_normal_test_data": target_normal_test,
        "target_abnormal_test_data": target_abnormal_test
    }
    
    # Define the model    
    model = MultiLayerPerceptron(num_features, 10)
    
    # Run TLNP
    tlnp_results = TLNP.run_tlnp(config_path, data_dict, model)

    # Run NNP
    nnp_results = TLNP.run_naive_np(config_path, data_dict, model)
    
    print(tlnp_results["test_metrics"])
    print(nnp_results["test_metrics"])
