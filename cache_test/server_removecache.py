def aggregate_fit(self, rnd: int, results: List[Tuple[ClientProxy, FitRes]], failures):
    print(f"Memory before round {rnd}:")
    self._log_memory_usage()
    
    performance_reports = []
    all_weights = []
    total_data_points = 0
    
    # Collect each client's performance metrics and weights
    for client_proxy, fit_res in results:
        client_id = client_proxy.cid
        if client_id not in self.client_id_mapping:
            self.client_id_mapping[client_id] = self.next_client_id
            self.next_client_id += 1
        unique_id = self.client_id_mapping[client_id]

        client_ip = self._get_client_ip(fit_res)
        val_accuracy = fit_res.metrics.get("val_accuracy", 0.0)
        improvement = fit_res.metrics.get("improvement", 0.0)

        # Record client performance
        performance_reports.append({
            "client_proxy": client_proxy,
            "fit_res": fit_res,
            "val_accuracy": val_accuracy,
            "unique_id": unique_id,
            "client_ip": client_ip,
            "improvement": improvement,
        })

    # Sort clients by validation accuracy in descending order
    performance_reports.sort(key=lambda x: x["val_accuracy"], reverse=True)

    # Select the top 70% of clients
    top_clients = performance_reports[:int(0.7 * len(performance_reports))]
    
    # Identify the bottom 30% of clients for cache removal
    bottom_clients = performance_reports[int(0.7 * len(performance_reports)):]

    # Remove cached weights for the bottom 30% of clients
    for report in bottom_clients:
        unique_id = report["unique_id"]
        if unique_id in self.last_update_cache:
            print(f"Removing cached weights for Client {unique_id}")
            del self.last_update_cache[unique_id]

    # Aggregate only the top 70% clients
    for report in top_clients:
        client_proxy = report["client_proxy"]
        fit_res = report["fit_res"]
        unique_id = report["unique_id"]
        client_ip = report["client_ip"]

        if fit_res.parameters.tensors:
            weights = parameters_to_weights(fit_res.parameters)
            print(f"Round {rnd}, Client {unique_id}: using direct update from client {client_ip}.")
            self.last_update_cache[unique_id] = fit_res.parameters
        elif unique_id in self.last_update_cache:
            weights = parameters_to_weights(self.last_update_cache[unique_id])
            print(f"Round {rnd}, Client {unique_id}: using cached weights.")
        else:
            print(f"Round {rnd}, Client {unique_id}: No update available.")
            continue

        # Weighted aggregation based on the number of examples each client has
        weighted_weights = [np.array(w) * fit_res.num_examples for w in weights]
        all_weights.append(weighted_weights)
        total_data_points += fit_res.num_examples

    # Perform weighted average on selected clients' weights
    if all_weights:
        num_layers = len(all_weights[0])
        aggregated_weights = [
            sum(weights[layer] for weights in all_weights) / total_data_points
            for layer in range(num_layers)
        ]
        aggregated_parameters = weights_to_parameters(aggregated_weights)
    else:
        print(f"No valid weights to aggregate in round {rnd}.")
        aggregated_parameters = None

    print(f"Memory after round {rnd}:")
    self._log_memory_usage()

    return aggregated_parameters, {}
