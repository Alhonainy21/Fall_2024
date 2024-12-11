import argparse
from collections import OrderedDict
from typing import List, Tuple
import flwr as fl
import numpy as np
import psutil
import logging
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_weights, weights_to_parameters, FitRes
from flwr.server.client_proxy import ClientProxy  # Corrected import

class CustomFedAvg(FedAvg):
    def __init__(self, *args, cache_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size  # Maximum number of weights in the cache
        self.last_update_cache = {}  # Cache to store the latest weights
        self.client_id_mapping = {}
        self.next_client_id = 1

    def _log_memory_usage(self):
        memory = psutil.virtual_memory()
        print(f"Memory Usage: {memory.percent}% ({memory.used / (1024 ** 3):.2f}GB used / {memory.total / (1024 ** 3):.2f}GB total)")

    def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        print(f"Round {rnd}: Aggregating results")
        self._log_memory_usage()

        performance_reports = []
        all_weights = []
        total_data_points = 0

        # Collect results and prepare performance reports
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            if client_id not in self.client_id_mapping:
                self.client_id_mapping[client_id] = self.next_client_id
                self.next_client_id += 1
            unique_id = self.client_id_mapping[client_id]

            val_accuracy = fit_res.metrics.get("val_accuracy", 0.0)
            performance_reports.append({
                "client_proxy": client_proxy,
                "fit_res": fit_res,
                "val_accuracy": val_accuracy,
                "unique_id": unique_id,
            })

        # Sort clients by validation accuracy in descending order
        performance_reports.sort(key=lambda x: x["val_accuracy"], reverse=True)

        # Aggregate weights of all clients in the cache
        for report in performance_reports:
            client_proxy = report["client_proxy"]
            fit_res = report["fit_res"]
            unique_id = report["unique_id"]

            # Cache management
            if len(self.last_update_cache) >= self.cache_size:
                # Remove the oldest cache entry if the cache is full
                removed_client = next(iter(self.last_update_cache))
                print(f"Cache full. Removing weights for Client {removed_client}.")
                del self.last_update_cache[removed_client]

            # Add/update weights in the cache
            if fit_res.parameters.tensors:
                self.last_update_cache[unique_id] = fit_res.parameters

            weights = parameters_to_weights(self.last_update_cache[unique_id])
            num_examples = fit_res.num_examples
            weighted_weights = [np.array(w) * num_examples for w in weights]
            all_weights.append(weighted_weights)
            total_data_points += num_examples

        # Aggregate the weights
        if all_weights:
            num_layers = len(all_weights[0])
            aggregated_weights = [
                sum(weights[layer] for weights in all_weights) / total_data_points
                for layer in range(num_layers)
            ]
            aggregated_parameters = weights_to_parameters(aggregated_weights)
        else:
            aggregated_parameters = None

        print(f"Round {rnd}: Aggregation complete")
        self._log_memory_usage()

        return aggregated_parameters, {}

# Argument parser setup
parser = argparse.ArgumentParser(description="Flower Server")
parser.add_argument("--server_address", type=str, required=True, help="gRPC server address")
parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
parser.add_argument("--cache_size", type=int, default=7, help="Maximum cache size for client weights")
parser.add_argument("--min_num_clients", type=int, default=2, help="Minimum number of clients to participate")
parser.add_argument("--min_sample_size", type=int, default=2, help="Minimum number of clients sampled per round")
parser.add_argument("--model", type=str, default="DenseNet121", help="Model to use for training")
args = parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)
    strategy = CustomFedAvg(cache_size=args.cache_size)
    fl.server.start_server(
        server_address=args.server_address,
        config={"num_rounds": args.rounds},
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
