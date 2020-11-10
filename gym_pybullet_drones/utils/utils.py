import os
import torch
from torch.autograd import Variable
import time

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

####################################################################################################
#### Sync the stepped simulation with the wall-clock ###############################################
####################################################################################################
#### Arguments #####################################################################################
#### - i (int)                              current simulation iteration ###########################
#### - start_time (timestamp)               timestamp of the simulation start ######################
#### - timestep (float)                     desired, wall-clock step of the simulation's rendering #
####################################################################################################
def sync(i, start_time, timestep):
    if timestep>.04 or i%(int(1/(24*timestep)))==0:
        elapsed = time.time() - start_time
        if elapsed<(i*timestep): time.sleep(timestep*i - elapsed)

####################################################################################################
#### Convert a string into a boolean ###############################################################
####################################################################################################
#### Arguments #####################################################################################
#### - val (?)                              input value (possibly stirng) to interpret as boolean ##
####################################################################################################
#### Returns #######################################################################################
#### - _ (bool)                             the boolean interpretation of the input value ##########
####################################################################################################
def str2bool(val):
    if isinstance(val, bool): return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise print("[ERROR] in str2bool(), a Boolean value is expected")
    
def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()
    
def to_tensor(ndarray, requires_grad=False, dtype=torch.FloatTensor):
    return Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).type(dtype)
    
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir
