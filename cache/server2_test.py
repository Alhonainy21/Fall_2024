import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List
import flwr as fl
import numpy as np
import torch
import torchvision
import utils
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
import psutil
import logging
import time

import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List
import flwr as fl
import numpy as np
import torch
import utils
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_weights, weights_to_parameters
from flwr.common import parameters_to_weights, weights_to_parameters, FitRes
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
import psutil
import logging
import time

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id_mapping = {}
        self.last_update_cache = {}
        self.last_update_time = {}  # Track the last update time for LRU cache replacement
        self.next_client_id = 1
        self.last_num_data_points = {}
        self.improvement_threshold = 0.1  # Example threshold for improvement

    def _get_client_ip(self, fit_res):
        # Extract the client's IP address from the fit results' metrics if available
        return fit_res.metrics.get("client_ip", "Unknown IP")

    def _log_memory_usage(self):
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024 ** 3)  # Convert bytes to gigabytes
        total_gb = memory.total / (1024 ** 3)  # Convert bytes to gigabytes
        print("=========================================================")
        print(f"Memory Usage: {memory.percent}% used of {total_gb:.2f}GB (Used: {used_gb:.2f} GB)")
        print("=========================================================")

    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures):
      print(f"Memory before round {rnd}:")
      self._log_memory_usage()
      aggregated_weights = None
      total_data_points = 0
      all_weights = []
  
      for client_proxy, fit_res in results:
          client_id = client_proxy.cid
          if client_id not in self.client_id_mapping:
              self.client_id_mapping[client_id] = self.next_client_id
              self.next_client_id += 1
          unique_id = self.client_id_mapping[client_id]
  
          client_ip = self._get_client_ip(fit_res)
  
          # Convert parameters to weights
          weights = parameters_to_weights(fit_res.parameters)
          
          # Log dimensions for each client's weights
          weight_shapes = [np.array(w).shape for w in weights]
          print(f"Round {rnd}, Client {unique_id} (IP {client_ip}): Weight dimensions {weight_shapes}")
  
          # Dimension check: skip if inconsistent with expected shape
          if unique_id in self.last_update_cache:
              cached_weight_shapes = [np.array(w).shape for w in parameters_to_weights(self.last_update_cache[unique_id])]
              if weight_shapes != cached_weight_shapes:
                  print(f"Skipping client {unique_id} due to inconsistent weight dimensions. Expected: {cached_weight_shapes}, Got: {weight_shapes}")
                  continue
          
          # Update cache and aggregate weights
          self.last_update_cache[unique_id] = fit_res.parameters
          self.last_num_data_points[unique_id] = fit_res.num_examples
  
          # Multiply weights by number of examples
          weighted_weights = [np.array(w) * fit_res.num_examples for w in weights]
          all_weights.append(weighted_weights)
          total_data_points += fit_res.num_examples
  
      # Perform aggregation if weights are valid
      if all_weights:
          num_layers = len(all_weights[0])
          aggregated_weights = [sum(weights[layer] for weights in all_weights) / total_data_points for layer in range(num_layers)]
          aggregated_parameters = weights_to_parameters(aggregated_weights)
      else:
          print(f"No valid weights to aggregate in round {rnd}.")
  
      print(f"Memory after round {rnd}:")
      self._log_memory_usage()
  
      return aggregated_parameters if aggregated_weights else None, {}


    def _evict_cache_if_needed(self, non_selected_client_ids: set):
        """Evict cached weights for non-selected clients if memory is low, using LRU strategy."""
        memory = psutil.virtual_memory()
        if memory.percent < 80:
            return  # Memory is sufficient, no need to evict

        # Sort non-selected clients by their last update time (LRU)
        lru_clients = sorted(
            [(cid, self.last_update_time.get(cid, 0)) for cid in non_selected_client_ids],
            key=lambda x: x[1]
        )

        # Remove clients until memory usage drops below 80% or all non-selected clients are evicted
        for client_id, _ in lru_clients:
            if client_id in self.last_update_cache:
                print(f"Evicting cached weights for client {client_id} to free memory.")
                del self.last_update_cache[client_id]
                del self.last_update_time[client_id]

            # Check memory status again
            if psutil.virtual_memory().percent < 80:
                break


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    help=f"gRPC server address",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=1,
    help="Number of rounds of federated learning (default: 1)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_sample_size",
    type=int,
    default=8,
    help="Minimum number of clients used for fit/evaluate (default: 2)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)
parser.add_argument(
    "--log_host",
    type=str,
    help="Logserver address (no default)",
)
parser.add_argument(
    "--model",
    type=str,
    default="ResNet18",
    choices=["Net", "ResNet18", "ResNet50", "DenseNet121", "MobileNetV2", "EfficientNetB0"],
    help="model to train",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="training batch size",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of workers for dataset reading",
)
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()


def main() -> None:
    assert (
        args.min_sample_size <= args.min_num_clients
    ), f"Num_clients shouldn't be lower than min_sample_size"
    
    fl.common.logger.configure("server", host=args.log_host)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    _, testset = utils.load_cifar()

    client_manager = SimpleClientManager()
    strategy = CustomFedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config={"num_rounds": args.rounds},
    )


def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    config = {
        "epoch_global": str(server_round),
        "epochs": str(3),
        "batch_size": str(args.batch_size),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
        "total_rounds": str(args.rounds),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    state_dict = OrderedDict(
        {k: torch.tensor(np.atleast_1d(v)) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict)


def get_eval_fn(testset: torchvision.datasets.CIFAR10) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model = utils.load_model(args.model)
        set_weights(model, weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = utils.test(model, testloader, device=DEVICE)
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    main()
