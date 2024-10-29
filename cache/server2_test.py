import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, List
import flwr as fl
import numpy as np
import torch
import psutil
import utils
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_weights, weights_to_parameters, FitRes
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
import logging
import time

class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id_mapping = {}
        self.last_update_cache = {}
        self.last_update_time = {}  # Track the last update time for LRU cache
        self.next_client_id = 1
        self.improvement_threshold = 0.1  # Threshold for performance improvement
        self.max_cache_size = 100  # Maximum number of clients to cache

    def _get_client_ip(self, fit_res):
        return fit_res.metrics.get("client_ip", "Unknown IP")

    def _log_memory_usage(self):
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024 ** 3)
        total_gb = memory.total / (1024 ** 3)
        print("=========================================================")
        print(f"Memory Usage: {memory.percent}% used of {total_gb:.2f}GB (Used: {used_gb:.2f} GB)")
        print("=========================================================")

    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        print(f"Memory before round {rnd}:")
        self._log_memory_usage()
        
        aggregated_weights = None
        total_data_points = 0
        all_weights = []
        performance_reports = []

        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            if client_id not in self.client_id_mapping:
                self.client_id_mapping[client_id] = self.next_client_id
                self.next_client_id += 1
            unique_id = self.client_id_mapping[client_id]

            client_ip = self._get_client_ip(fit_res)
            improvement = fit_res.metrics.get("improvement", 0.0)

            if fit_res.parameters.tensors:
                weights = parameters_to_weights(fit_res.parameters)
                print(f"Round {rnd}, Client {unique_id}: using direct update from client {client_ip}.")
                self.last_update_cache[unique_id] = fit_res.parameters
                self.last_update_time[unique_id] = time.time()  # Update cache time
            elif unique_id in self.last_update_cache:
                weights = parameters_to_weights(self.last_update_cache[unique_id])
                print(f"Round {rnd}, Client {unique_id}: using cached weights.")
            else:
                print(f"Round {rnd}, Client {unique_id}: No update available.")
                continue

            performance_reports.append({
                "client_id": unique_id,
                "train_loss": fit_res.metrics.get("train_loss"),
                "train_accuracy": fit_res.metrics.get("train_accuracy"),
                "val_loss": fit_res.metrics.get("val_loss"),
                "val_accuracy": fit_res.metrics.get("val_accuracy"),
                "test_loss": fit_res.metrics.get("test_loss"),
                "test_accuracy": fit_res.metrics.get("test_accuracy"),
                "improvement": improvement
            })

            if weights and len(weights) == len(parameters_to_weights(self.last_update_cache[unique_id])):
                weighted_weights = [np.array(w) * fit_res.num_examples for w in weights]
                all_weights.append(weighted_weights)
                total_data_points += fit_res.num_examples

        # Evict old cached entries if needed
        self._evict_cache_if_needed()

        if all_weights:
            num_layers = len(all_weights[0])
            aggregated_weights = [
                sum(weights[layer] for weights in all_weights) / total_data_points
                for layer in range(num_layers)
            ]
            aggregated_parameters = weights_to_parameters(aggregated_weights)
        else:
            print(f"No valid weights to aggregate in round {rnd}.")

        print(f"Memory after round {rnd}:")
        self._log_memory_usage()
        print("Top clients based on performance:", sorted(performance_reports, key=lambda x: -x["val_accuracy"])[:5])

        return aggregated_parameters if aggregated_weights else None, {}

    def _evict_cache_if_needed(self):
        """Evict cached weights based on LRU if the cache exceeds max_cache_size."""
        if len(self.last_update_cache) <= self.max_cache_size:
            return

        # Sort clients by their last update time (oldest first)
        lru_clients = sorted(self.last_update_time.items(), key=lambda x: x[1])

        # Remove oldest entries until cache is within the allowed size
        while len(self.last_update_cache) > self.max_cache_size:
            client_id, _ = lru_clients.pop(0)
            if client_id in self.last_update_cache:
                print(f"Evicting cached weights for client {client_id} to manage cache size.")
                del self.last_update_cache[client_id]
                del self.last_update_time[client_id]

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--server_address", type=str, required=True, help=f"gRPC server address")
parser.add_argument("--rounds", type=int, default=1, help="Number of federated learning rounds")
parser.add_argument("--sample_fraction", type=float, default=1.0, help="Fraction of available clients used")
parser.add_argument("--min_sample_size", type=int, default=8, help="Minimum number of clients used per round")
parser.add_argument("--min_num_clients", type=int, default=2, help="Minimum number of clients for sampling")
parser.add_argument("--log_host", type=str, help="Logserver address")
parser.add_argument("--model", type=str, default="ResNet18", choices=["Net", "ResNet18", "ResNet50","DenseNet121","MobileNetV2","EfficientNetB0"], help="Model to train")
parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--num_workers", type=int, default=4, help="Number of dataset reading workers")
parser.add_argument("--pin_memory", action="store_true")
args = parser.parse_args()

def main():
    fl.common.logger.configure("server", host=args.log_host)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    _, testset = utils.load_cifar()

    strategy = CustomFedAvg(
        fraction_fit=args.sample_fraction,
        min_fit_clients=args.min_sample_size,
        min_available_clients=args.min_num_clients,
        eval_fn=get_eval_fn(testset),
        on_fit_config_fn=fit_config,
    )

    fl.server.start_server(
        server_address=args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )

def fit_config(server_round: int) -> Dict[str, fl.common.Scalar]:
    return {
        "epoch_global": str(server_round),
        "epochs": str(3),
        "batch_size": str(args.batch_size),
        "num_workers": str(args.num_workers),
        "pin_memory": str(args.pin_memory),
        "total_rounds": str(args.rounds),
    }

def set_weights(model: torch.nn.Module, weights: fl.common.Weights):
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)})
    model.load_state_dict(state_dict)

def get_eval_fn(testset: torch.utils.data.Dataset) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model = utils.load_model(args.model)
        set_weights(model, weights)
        model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        loss, accuracy = utils.test(model, testloader, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        return loss, {"accuracy": accuracy}

    return evaluate

if __name__ == "__main__":
    main()
