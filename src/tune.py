import argparse
import logging

import optuna
from optuna import study
from optuna.trial import TrialState

import config as cfg
from trainutil import train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# define arguments, override the defaults from config.py with arguments
args = argparse.ArgumentParser()
args.add_argument('training-file', type=str)
args.add_argument('validation-file', type=str)
args.add_argument("experiment-name", type=str)

args.add_argument('--num-trials', type=int, default=20)
args.add_argument('--batch-size', type=int, default=cfg.batch_size)
args.add_argument('--lr', type=float, default=cfg.lr)
args.add_argument("--model-name", type=str, default=cfg.model_name)
args.add_argument("--max-step", type=int, default=-1)
args.add_argument("--max-length", type=int, default=cfg.max_length)
args.add_argument("--max-epoch", type=str, default=cfg.max_epoch)
args.add_argument("--no-pretrain", action="store_true")
args.add_argument("--weight-decay", type=str, default=cfg.weight_decay)
args.add_argument("--warmup-steps", type=int, default=cfg.warmup_steps)
args.add_argument("--timeout", type=int, default=None)

args = args.parse_args()

# build param dictionary from args
params = vars(args)
params = {k.replace('-', '_'): v for k, v in params.items()}

global root_exp_name


def objective(trial: optuna.trial.Trial):
    logger.info(f"starting trial {trial.number}")

    lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    # max_epoch = trial.suggest_int('num_epochs', 1, 10)
    # batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    warmup_steps = trial.suggest_int('warmup_steps', 0, 500)
    nth_hidden_layer = trial.suggest_int("nth_hidden_layer", 5, 10)

    params['nth_hidden_layer'] = nth_hidden_layer
    params['lr'] = lr
    params['weight_decay'] = weight_decay
    # params['max_epoch'] = max_epoch
    # params['batch_size'] = batch_size
    params['warmup_steps'] = warmup_steps

    params['experiment_name'] = root_exp_name + "/" + f"trial-{trial.number}"

    history = train(params, trial, disable_tqdm=True)

    return history['valid/f1'][-1]


if __name__ == '__main__':
    root_exp_name = params['experiment_name']

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=params['num_trials'], timeout=params['timeout'])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))