import time
import sys
import argparse
import random
import itertools
import os
import os.path as osp

import yaml
import numpy as np
import torch

from torch_geometric.loader import DataLoader

import models
from data import MatbenchDataset
from training import Trainer, train_model, Normalizer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel 
from torch.utils.data.distributed import DistributedSampler

# Global counter for naming
counter = itertools.count()
def trial_count_number(trial):
    # Each time a new trial is created, this function will be called
    # the trial is needed for an internal call of ray tune
    return f"Trial {next(counter)}"

# Custon update function to avoid overwriting the original dictionary
def deep_update(original: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_params(args):
    params = yaml.load(open(f'params_{args.model_name}.yml'), Loader=yaml.FullLoader)
    model_params = params["Model Parameters"]
    optimizer_params = params["Optimizer Parameters"]
    scheduler_params = params["Scheduler Parameters"]

    # Update the model parameters
    model_params["path_dataset"] = args.path_dataset
    model_params["len_dataset"] = args.len_dataset
    model_params["name_dataset"] = args.name_dataset
    model_params["epochs"] = args.epochs
    model_params["num_workers"] = args.num_workers
    model_params["ntrials"] = args.ntrials
    model_params["internal_params"]["task"] = args.task

    return model_params, optimizer_params, scheduler_params


def load_dataset(rank, path_dataset, len_dataset = 30000, name = 'mp_gap', graph_algorithm = 'KNN', cell_type = 'UnitCell'):
    # Load the dataset
    start = time.time()
    path = osp.join(f'{path_dataset}', name)
    dataset = MatbenchDataset(path, name, 
                                graph_algorithm= graph_algorithm, cell_type=cell_type)
    
    if rank == 0:
        print("Loading the dataset.....")
        print("Dataset loaded!")
        print(f"Time: {time.time() - start:.4f} seconds")
        sys.stdout.flush()

    # Reduce the size of the dataset
    dataset = dataset if len_dataset == 'all' else dataset[:int(len_dataset)]

    return dataset


def split_datasets(dataset, rank):
    # Ratios: 80% train, 10% validation, 10% test
    train_ratio, valid_ratio, test_ratio = 0.8, 0.1, 0.1
    dataset_size = len(dataset)

    train_size = int(train_ratio * dataset_size)
    valid_size = int(valid_ratio * dataset_size)
    test_size = dataset_size - train_size - valid_size 

    if rank == 0: 
        print(f"len(train_dataset) = {train_size}")
        print(f"len(val_dataset) = {valid_size}")
        print(f"len(test_dataset) = {test_size}")
        sys.stdout.flush()

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    return train_dataset, val_dataset, test_dataset

def training_loader_ddp(rank, world_size, train_dataset, batch_size, num_workers = 2, gpu_bool = True):
    # The only distributed sampler if for the training dataset
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
    pin_memory = True if gpu_bool else False   

    train_loader = DataLoader(train_dataset, 
                               batch_size = batch_size,
                               sampler=train_sampler,
                               pin_memory=pin_memory,
                               num_workers=num_workers
                            )

    return train_loader

def training_loader_no_ddp( train_dataset, batch_size, num_workers = 2, gpu_bool = True):
    pin_memory = bool(gpu_bool)
    train_loader = DataLoader(train_dataset,
                               batch_size = batch_size,
                               pin_memory=pin_memory,
                               num_workers=num_workers
                             )
    return train_loader

def val_test_loaders_no_ddp(val_dataset, test_dataset, batch_size, num_workers = 2, gpu_bool = True):
    # validation and evaluation only takes place on rank 0,
    # so there is no need to load the data in parallel
    pin_memory = bool(gpu_bool)
    val_loader = DataLoader(val_dataset, 
                            batch_size = batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=False
                            )
    test_loader = DataLoader(test_dataset, 
                             batch_size = batch_size,
                             pin_memory=pin_memory,
                             num_workers=num_workers,
                             shuffle=False
                             )
    return val_loader, test_loader

def cleanup():
    dist.destroy_process_group()

def setup():
    dist.init_process_group(backend="nccl")
    if int(os.environ["RANK"]) == 0:
        print("Distributed DDP initialized")

def training_no_ddp(model_params,
                    optimizer_params,
                    scheduler_params,
                    ):
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_bool = True if device != "cpu" else False
    dataset = load_dataset(0,
                            model_params["path_dataset"],
                            len_dataset = model_params["len_dataset"],
                            name=model_params["name_dataset"],
                            graph_algorithm=model_params["graph_algorithm"],
                            cell_type = 'UnitCell'
                            )
    train_dataset, val_dataset, test_dataset = split_datasets(dataset, 0)
    train_loader = training_loader_no_ddp(train_dataset, 
                                        model_params["batch_size"],
                                        num_workers = model_params["num_workers"],
                                        gpu_bool = gpu_bool
                                       )

    val_loader, test_loader = val_test_loaders_no_ddp(val_dataset,
                                                      test_dataset,
                                                      model_params["batch_size"],
                                                      num_workers = model_params["num_workers"],
                                                      gpu_bool = gpu_bool
                                                      )

    if model_params["internal_params"]["task"] == 'regression':
        # Normalize targets to mean = 0 and std = 1.
        mean = dataset.data.y.mean(dim=0, keepdim=True)
        std = dataset.data.y.std(dim=0, keepdim=True)
        normalizer = Normalizer(mean, std)
        dataset.data.y = normalizer.normalize(dataset.data.y)
        print(f"Mean value of the targets: {mean.item()}")
        print(f"Std value of the targets: {std.item()}")
        sys.stdout.flush()
    else:   
        normalizer = None

    model_params["internal_params"]["in_channels"] = dataset[0].x.size(-1)
    model_params["internal_params"]["edge_dim"] = dataset[0].edge_attr.size(-1)
    model_params["internal_params"]["out_channels"] = dataset[0].y.size(-1)

    loss_function = 'mse_loss' if model_params["internal_params"]["task"] == 'regression' else 'binary_cross_entropy'
    eval_metric = 'l1_loss' if model_params["internal_params"]["task"] == 'regression' else 'auc'
    try:
        # this is done to avoid errors when we have several workers
        model_class = getattr(models, model_params["name"])
        model = model_class(**model_params["internal_params"]).to(device)
    except AttributeError:
        print(f"Params: {model_params}")
        raise ValueError("Model not implemented!")

    optimizer = getattr(torch.optim, optimizer_params["name"])(
        model.parameters(),
        **optimizer_params["params"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_params["name"])(
        optimizer, **scheduler_params["params"]
    )

    trainer = Trainer({'device': device, 
                       'epochs': model_params["epochs"], 
                       'save_every' : 100, 
                       'task': model_params["internal_params"]["task"],
                       'loss_function': loss_function, 
                       'parallel_bool': model_params["parallel_bool"], 
                       'rank': model_params["rank"],
                       'eval_metric': eval_metric,
                        })
    
    best_val_error, best_test_error = train_model(model, optimizer,
                                                train_loader,
                                                val_loader,
                                                test_loader,
                                                trainer,
                                                scheduler=scheduler,
                                                normalizer=normalizer,
                                                early_stopping = model_params["early_stopping"],
                                                bool_plot = model_params["bool_plot"],
                                    )

    if normalizer is not None:
        print(f"{'Best validation error:':25} {normalizer.std.item() * best_val_error:>15.6f} [Physical units] \t||\t {best_val_error:>15.6f} [std. units]")
        print(f"{'Best test error:':25} {normalizer.std.item() * best_test_error:>15.6f} [Physical units] \t||\t {best_test_error:>15.6f} [std. units]")
        best_test_error = normalizer.std.item() * best_test_error
        best_val_error = normalizer.std.item() * best_val_error
    else:
        print("Best validation error: ", best_val_error)
        print("Best test error: ", best_test_error)

    return best_val_error, best_test_error

def objective(
    config,
    model_params,
    optimizer_params,
    scheduler_params,

):
    # Update model parameters
    updated_model_params = deep_update(model_params, {
        "internal_params": {
            "hidden_channels":  config["hidden_channels"],
            "layers_attention": config["layers_attention"],
            "pre_conv_layers":  config["pre_conv_layers"],
            "post_conv_layers": config["post_conv_layers"],
            "dropout":          config["dropout"],
            "heads":            config["heads"],
            "aggregation":      config["aggregation"],
            "pooling":          config["pooling"],
            "activation":       config["activation"],
            "activation_cell":  config["activation_cell"],
        },
        "graph_algorithm":  config["graph_algorithm"],
        "batch_size":       config["batch_size"],
    })

    # Update optimizer parameters
    optimizer_params["params"].update({
        "lr":           config["lr"],
        "weight_decay": config["weight_decay"]
    })

    best_val_error, best_test_error = training_no_ddp(
        updated_model_params, 
        optimizer_params,
        scheduler_params,
    )

    # Report to Ray the metric(s) you want to optimize
    tune.report({
        "loss": best_val_error,
    })

def hyperparam_optim(args):
    width = 58  # Width inside the borders
    print(
        "=" * 62 + "\n"
        "||" + " Starting an Ray Tune hyperparameter optimization".center(width) + " ||\n"
        "||" + " Using the hyperopt TPE sampler and the ASHAS scheduler".center(width) + " ||\n"
        "||" + " Parameters used:".ljust(width) + " ||\n"
        f"|| {'model_name':<15}: {args.model_name:<{width - 18}} ||\n"
        f"|| {'dataset':<15}: {args.name_dataset:<{width - 18}} ||\n"
        f"|| {'len_dataset':<15}: {args.len_dataset:<{width - 18}} ||\n"
        f"|| {'task':<15}: {args.task:<{width - 18}} ||\n"
        f"|| {'epochs':<15}: {args.epochs:<{width - 18}} ||\n"
        f"|| {'num_workers':<15}: {args.num_workers:<{width - 18}} ||\n"
        f"|| {'ntrials':<15}: {args.ntrials:<{width - 18}} ||\n"
        f"|| {'torch_compile':<15}: {args.torch_compile:<{width - 18}} ||\n"
        f"|| {'use_matf32':<15}: {args.use_matf32:<{width - 18}} ||\n"
        f"|| {'Number of GPUs':<15}: {torch.cuda.device_count():<{width - 18}} ||\n"
        + "=" * 62
    )
    sys.stdout.flush()

    from ray import tune
    from ray.tune import Tuner, TuneConfig, RunConfig
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search import ConcurrencyLimiter

    # Initialize parameters
    model_params, optimizer_params, scheduler_params = init_params(args)
    model_params["bool_plot"] = False
    model_params["parallel_bool"] = False
    model_params["rank"] = 0

    # Define the search space
    search_space = {}

    max_layers = 25 if model_params["internal_params"]["residual_connection"] else 5
    step_layers = 5 if model_params["internal_params"]["residual_connection"] else 1
    min_layers = 5 if model_params["internal_params"]["residual_connection"] else 1
    layers = [x for x in range(min_layers, max_layers + 1, step_layers)] # 5, 10,... 30 or 1, 2,... 5
    dim = [x * 25 for x in range(1, 9)] # 25, 50,... 200
    batch_sizes = [64, 128, 256]
    lrs = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    weight_decays = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    search_space["GATom"] = {
            # Model hyperparams
            "hidden_channels": tune.choice(dim),
            "layers_attention": tune.choice(layers),
            "pre_conv_layers": 1, 
            "post_conv_layers": 3,   
            "dropout": tune.uniform(0.0, 0.9),
            "heads": tune.choice([1, 2, 3, 4]),
            "aggregation": tune.choice(["mean", "add", "max", "softmax"]),
            "pooling": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool"]),
            "activation": tune.choice(["relu", "leaky_relu", "sigmoid", "silu"]), 
            "activation_cell": tune.choice(["relu", "gelu", "silu", "sigmoid"]), # ReGLU, GeGLU, SiGLU, and GLU (with sigmoid)    
            # Graph & batch size
            "graph_algorithm": tune.choice(["KNN", "Voronoi"]),
            "batch_size": tune.choice(batch_sizes),

            # Optimizer hyperparams
            "lr": tune.choice(lrs),
            "weight_decay": tune.choice(weight_decays),
    }

    search_space["IMcgcnn"] = {
            # Model hyperparams
            "hidden_channels": tune.choice(dim),
            "layers_attention": tune.choice(layers),
            "num_timesteps": tune.choice([1, 2, 3, 4]),
            "pre_conv_layers": tune.choice([1, 2, 3]),
            "post_conv_layers": tune.choice([1, 2, 3]),
            "dropout": tune.uniform(0.0, 0.9),
            "pooling": tune.choice(["global_mean_pool", "global_add_pool", "global_max_pool"]),
            "activation": tune.choice(["relu", "leaky_relu", "sigmoid"]),

            # Graph & batch size
            "graph_algorithm": tune.choice(["KNN", "Voronoi"]),
            "batch_size": tune.choice([8, 16, 32, 64, 128]),

            # Optimizer hyperparams
            "lr": tune.loguniform(1e-5, 5e-2),
            "weight_decay": tune.loguniform(1e-5, 1e-1),
    }



    # Check resources
    if torch.cuda.is_available():
        model_params["num_workers"] = 4
        if torch.cuda.device_count() > 2:
            resources = {"cpu":12, "gpu": 1}
        elif torch.cuda.device_count() <= 2:
            resources = {"cpu":32, "gpu": 1}
    else:
        print("No GPUs available!")
        model_params["num_workers"] = 0
        resources = {"cpu": 1, "gpu":0}

    # Init ray for slurm
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 2:
            ray.init(ignore_reinit_error=True,
                     num_cpus=12*torch.cuda.device_count(),
                     num_gpus=torch.cuda.device_count()
                     )
        elif torch.cuda.device_count() <= 2:
            ray.init(ignore_reinit_error=True,
                     num_cpus=32*torch.cuda.device_count(),
                     num_gpus=torch.cuda.device_count()
                     )
    else:
        print("Using local_mode for ray. This is meant for debugging purposes.")
        ray.init(ignore_reinit_error=True, local_mode=True, num_cpus=1, num_gpus=0)


    mode = "min" if model_params["internal_params"]["task"] == "regression" else "max" # minimize error or maximize accuracy
    metric = "loss" if model_params["internal_params"]["task"] == "regression" else "accuracy" # error or accuracy
    # set up variables from ray tune
    search_algorithm = HyperOptSearch(
        metric=metric,
        mode=mode,
        n_initial_points=5,
    )

    if torch.cuda.device_count() > 1:
        search_algorithm = ConcurrencyLimiter(search_algorithm, max_concurrent=torch.cuda.device_count())

    scheduler = ASHAScheduler(
        metric=metric,
        mode=mode,
        max_t=model_params["epochs"],
        reduction_factor=2
    )

    trainable = tune.with_resources(
        tune.with_parameters(
            objective,
            model_params=model_params,
            optimizer_params=optimizer_params,
            scheduler_params=scheduler_params
        ),
        resources=resources
    )

    storage_path = os.path.join(os.getcwd(), "ray_results")

    tuner = Tuner(
        trainable=trainable,
        param_space=search_space[model_params["name"]],
        tune_config=TuneConfig(
            num_samples=args.ntrials,
            search_alg=search_algorithm,
            scheduler=scheduler,
            trial_name_creator=trial_count_number,
        ),
        run_config=RunConfig(
            name="hyperparam_optimization",
            log_to_file=True,
            storage_path=storage_path,
            verbose=1
        )
    )

    # run the optimization
    results = tuner.fit()

    # get the best results
    best_result = results.get_best_result(metric=metric, mode=mode)
    best_config = best_result.config

    print("Best trial config:", best_config)
    print("Best trial final metrics:", best_result.metrics)

    # print the table
    # Now sort trials from minimum to maximum loss
    df = results.get_dataframe()
    df.to_csv("results.csv")
    df_sorted = df.sort_values(by=metric, ascending=True)
    print(df_sorted.to_string())



def single_calculation_no_ddp(args):
    width = 58  # Width inside the borders
    print(
        "=" * 62 + "\n"
        "||" + "Starting a single calculation!".center(width) + " ||\n"
        "||" + " Parameters used:".ljust(width) + " ||\n"
        f"|| {'model_name':<15}: {args.model_name:<{width - 18}} ||\n"
        f"|| {'dpp':<15}: {args.ddp:<{width - 18}} ||\n"
        f"|| {'dataset':<15}: {args.name_dataset:<{width - 18}} ||\n"
        f"|| {'len_dataset':<15}: {args.len_dataset:<{width - 18}} ||\n"
        f"|| {'task':<15}: {args.task:<{width - 18}} ||\n"
        f"|| {'epochs':<15}: {args.epochs:<{width - 18}} ||\n"
        f"|| {'num_workers':<15}: {args.num_workers:<{width - 18}} ||\n"
        f"|| {'torch_compile':<15}: {args.torch_compile:<{width - 18}} ||\n"
        f"|| {'use_matf32':<15}: {args.use_matf32:<{width - 18}} ||\n"
        f"|| {'fix_seed':<15}: {args.fix_seed:<{width - 18}} ||\n"
        + "=" * 62
    )

    sys.stdout.flush()

    model_params, optimizer_params, scheduler_params = init_params(args)

    # set bool_plot to True
    model_params["bool_plot"] = True
    # set parallel_bool to False and rank to 0
    model_params["parallel_bool"] = False
    model_params["rank"] = 0

    best_val_error, best_test_error = training_no_ddp(model_params, optimizer_params, scheduler_params)

    return None



if __name__ == '__main__':


    # Set the all the parallel info
    parser = argparse.ArgumentParser(
        description="GATom: a Graph Attention neTwOrk for inference of Materials properties"
    )

    parser.add_argument(
        "--no-cuda",                 
        action="store_true",
        default=False,
        help="Disable CUDA training."
    )
    parser.add_argument(
        '--calculation_type', 
        type=str,
        default='single_calc',
        help='Single calculation or hyperparameter optimization.'
    )
    parser.add_argument(
        '--ddp', 
        action="store_true",
        default=False,
        help='Whether to use DDP or not'
    )
    parser.add_argument(
        '--name_dataset', 
        type=str,
        default='mp_gap',
        help='Name of the dataset to use'
    )
    parser.add_argument(
        '--path_dataset', 
        type=str,
        default='data',
        help='Path to the dataset'
    )
    parser.add_argument(
        '--task', 
        type=str,
        default='regression',
        help='Task to do for that dataset: regression or classification'
    )
    parser.add_argument(
        '--model_name', 
        type=str,
        default='GATom',
        help='Name of the model to use. Current options are GATom or IMcgcnn'
    )
    parser.add_argument(
        '--epochs', 
        type=int,
        help='Total epochs to train the model'
    )
    parser.add_argument(
        '--len_dataset', 
        help='Length of the dataset'
    )
    parser.add_argument(
        '--ntrials', 
        default=0,
        type=int,
        help='Number of trials to run for the hyperparameter optimization'
    )
    parser.add_argument(
        '--num_workers', 
        default=2,  # 2 workers for the dataloader. see ml-engineering repo
        type=int,
        help='Number of workers to load the data'
    )
    parser.add_argument(
        '--torch_compile', 
        action="store_true",
        default=False,
        help='Whether to use torch.compile or not'
    )
    parser.add_argument(
        '--use_matf32', 
        action="store_true",
        default=False,
        help='Whether to use matf32 or not'
    )
    parser.add_argument(
        '--fix_seed', 
        action="store_true",
        default=False,
        help='Whether to fix a seed for reproducibility'
    )

    args = parser.parse_args()

    # only activate if the architecture of the gpu is Volta or higher.
    if args.use_matf32:
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True
        print("Using tf32 for matmul and cudnn!")
    else:
        torch.set_float32_matmul_precision('high')
        print("Using fp32!")


    if args.fix_seed:
        set_seed(42)    
        # Optional flags for reproducibility (may slow down training):
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print("Seed fixed to 42! This is done for reproducibility. Set to false in production.")


    if args.calculation_type == 'single_calc':
        single_calculation_no_ddp(args)
    elif args.calculation_type == 'hyperparam_optim':
        hyperparam_optim(args)
    else:
        raise ValueError(f"Calculation type {args.calculation_type} not implemented!")
