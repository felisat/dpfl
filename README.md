# Federated Learning Simulator

Simulate Federated Learning with compressed communication on a large number of Clients.



## Usage

`python federated_learning.py`

will run the Federated Learning experiment specified in  

`federated_learning.json`.

You can specify:

### Task
- `"dataset"` : Choose from `["EMNIST"]`
- `"net"` : Choose from `["ConvNet"]`

### Federated Learning Environment

- `"n_clients"` : Number of Clients
- `"dirichlet_alpha"` : Parameter of the Dirichlet distribution, which determines the distribution of labels among the clients, large alpha > 100.0 will chreate iid splits, small alpha < 1.0 will create non-iid splits 
- `"communication_rounds"` : Number of communication rounds
- `"participation_rate"` : Fraction of Clients which participate in every Communication Round
- `"batch_size"` : Batch-size used by the Clients
- `"local_epochs"` : Total number of local training epochs performed by clients in each communication round


### Compression Method

- `"compression"` : The compression method that is used to reduce the upstream communication
- `"accumulate"` : Whether or not errors are accumulated

### Logging 
- `"log_frequency"` : Number of communication rounds after which results are logged and saved to disk
- `"log_path"` : e.g. "results/experiment1/"

Run multiple experiments by listing different configurations.

## Options
- `--schedule` : specify which batch of experiments to run, defaults to "main"

