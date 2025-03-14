import torch
import torch.nn as nn

from tlnp_lib import TLNP

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
    num_features = 5 # 5 input dimensions

    # Target normal data with mean 0
    target_normal_data = sample_multivariate_normal(num_features, 0.0, 4000)
    target_normal_test_data = sample_multivariate_normal(num_features, 0.0, 5000)
    
    # Target abnormal data with mean 0.5
    target_abnormal_data = sample_multivariate_normal(num_features, 0.50, 100)
    target_abnormal_test_data = sample_multivariate_normal(num_features, 0.50, 5000)

    # Source abnormal data with mean 0.49 (simulates data close to the target abnormal data)
    source_abnormal_data = sample_multivariate_normal(num_features, 0.49, 3000)
    
    # Set the config_path
    config_path = "example_config.yaml"
    
    # Define the data dict
    data_dict = {
        "target_normal_data": target_normal_data,
        "target_abnormal_data": target_abnormal_data,
        "source_abnormal_data": source_abnormal_data,
        "target_normal_test_data": target_normal_test_data,
        "target_abnormal_test_data": target_abnormal_test_data,
    }
    
    # Define the model    
    model = MultiLayerPerceptron(num_features, 10)
    
    # Run TLNP
    tlnp_results = TLNP.run_tlnp(config_path, data_dict, model)

    # Run NNP
    nnp_results = TLNP.run_naive_np(config_path, data_dict, model)

    # Print results
    print(f"\nTLNP Results:")
    print(tlnp_results["test_metrics"])
    print(f"\nNNP Results:")
    print(nnp_results["test_metrics"])
