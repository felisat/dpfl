import os, argparse, json, copy, time
from tqdm import tqdm
import torch, torchvision
import numpy as np

import data_utils as data 
import compression_utils as comp
import models 
import experiment_manager as xpm
from fl_devices import Client, Server



np.set_printoptions(precision=4, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--schedule", default="main", type=str)
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=None, type=int)
parser.add_argument("--reverse_order", default=False, type=bool)
parser.add_argument("--hp", default=None, type=str)
args = parser.parse_args()



def run_experiment(xp, xp_count, n_experiments):

    print(xp)
    hp = xp.hyperparameters

    model_fn, optimizer_fn = models.get_model(hp["net"])
    compression_fn = comp.get_compression(hp["compression"])
    client_data, server_data = data.get_data(hp["dataset"], n_clients=hp["n_clients"], alpha=hp["dirichlet_alpha"])


    clients = [Client(model_fn, optimizer_fn, subset, hp["batch_size"], idnum=i) for i, subset in enumerate(client_data)]
    server = Server(model_fn, server_data)
    server.load_model(path="checkpoints/", name=hp["pretrained"])

    # print model
    models.print_model(server.model)

    # Start Distributed Training Process
    print("Start Distributed Training..\n")
    t1 = time.time()
    for c_round in range(1, hp["communication_rounds"]+1):

        participating_clients = server.select_clients(clients, hp["participation_rate"])
    
        for client in tqdm(participating_clients):
            client.synchronize_with_server(server)
            train_stats = client.compute_weight_update(hp["local_epochs"])  
            client.compress_weight_update(compression_fn, accumulate=hp["accumulate"])
      
        server.aggregate_weight_updates(clients)


        # Logging
        if xp.is_log_round(c_round):
            print("Experiment: {} ({}/{})".format(args.schedule, xp_count+1, n_experiments))   

            xp.log({'communication_round' : c_round, 'epochs' : c_round*hp['local_epochs']})

            # Evaluate  
            xp.log({"client_train_{}".format(key) : value for key, value in train_stats.items()})
            xp.log({"server_val_{}".format(key) : value for key, value in server.evaluate().items()})

            # Save results to Disk
            try:
                xp.save_to_disc(path="results/", name=hp['log_path'])
            except:
                print("Saving results Failed!")

            # Timing
            e = int((time.time()-t1)/c_round*(hp['communication_rounds']-c_round))
            print("Remaining Time (approx.):", '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60), 
                    "[{:.2f}%]\n".format(c_round/hp['communication_rounds']*100))

    # Save model to disk
    server.save_model(path="checkpoints/", name=hp["save_model"])

    # Delete objects to free up GPU memory
    del server; clients.clear()
    torch.cuda.empty_cache()


def run():
    with open('federated_learning.json') as data_file:    
        experiments_raw = json.load(data_file)[args.schedule]

    hp_dicts = [hp for x in experiments_raw for hp in xpm.get_all_hp_combinations(x)][args.start:args.end]
    if args.reverse_order:
        hp_dicts = hp_dicts[::-1]
    experiments = [xpm.Experiment(hyperparameters=hp) for hp in hp_dicts]

    print("Running {} Experiments..\n".format(len(experiments)))
    for xp_count, experiment in enumerate(experiments):
        run_experiment(experiment, xp_count, len(experiments))


if __name__ == "__main__":
    run()
    