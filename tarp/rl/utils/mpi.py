"""Helpers for multi-processing."""

from mpi4py import MPI
import signal
import sys
import numpy as np
import torch

from tarp.utils.general_utils import AttrDict

def setup_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)

def update_with_mpi_config(conf):
    mpi_config = AttrDict()
    rank = MPI.COMM_WORLD.Get_rank()
    mpi_config.rank = rank
    mpi_config.is_chef = rank == 0
    mpi_config.num_workers = MPI.COMM_WORLD.Get_size()
    conf.mpi = mpi_config

    # update conf
    conf.general.seed = conf.general.seed + rank
    return conf


def set_shutdown_hooks():
    def shutdown(signal, frame):
        print('Received signal %s: exiting', signal)
        sys.exit(128+signal)

    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function of the
    `same name`_.
    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n (int): Number of process to split into.
        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))

def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()

def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)

def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()

def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)

def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std

# sync gradients across the different cores
def sync_grads(network):
    flat_grads, grads_shape = _get_flat_grads(network)
    comm = MPI.COMM_WORLD
    global_grads = np.zeros_like(flat_grads)
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
    global_grads /= num_procs()
    _set_flat_grads(network, grads_shape, global_grads)


# sync_networks across the different cores
def sync_networks(network):
    """
    netowrk is the network you want to sync
    """
    comm = MPI.COMM_WORLD
    flat_params, params_shape = _get_flat_params(network)
    comm.Bcast(flat_params, root=0)
    # set the flat params back to the network
    _set_flat_params(network, params_shape, flat_params)


def _get_flat_grads(network):
    grads_shape = {}
    flat_grads = None
    for key_name, value in network.named_parameters():
        if not value.requires_grad or value.grad is None: continue
        try:
            grads_shape[key_name] = value.grad.data.cpu().numpy().shape
        except:
            print('Cannot get grad of tensor {}'.format(key_name))
            import pdb; pdb.set_trace()
        if flat_grads is None:
            flat_grads = value.grad.data.cpu().numpy().flatten()
        else:
            flat_grads = np.append(flat_grads, value.grad.data.cpu().numpy().flatten())
    return flat_grads, grads_shape


def _set_flat_grads(network, grads_shape, flat_grads):
    pointer = 0
    if hasattr(network, '_config'):
        device = network._config.device
    else:
        device = torch.device("cpu")

    for key_name, value in network.named_parameters():
        if not value.requires_grad or value.grad is None: continue
        len_grads = int(np.prod(grads_shape[key_name]))
        copy_grads = flat_grads[pointer:pointer + len_grads].reshape(grads_shape[key_name])
        copy_grads = torch.tensor(copy_grads).to(device)
        # copy the grads
        value.grad.data.copy_(copy_grads.data)
        pointer += len_grads


# get the flat params from the network
def _get_flat_params(network):
    param_shape = {}
    flat_params = None
    for key_name, value in network.named_parameters():
        param_shape[key_name] = value.cpu().detach().numpy().shape
        if flat_params is None:
            flat_params = value.cpu().detach().numpy().flatten()
        else:
            flat_params = np.append(flat_params, value.cpu().detach().numpy().flatten())
    return flat_params, param_shape


# set the params from the network
def _set_flat_params(network, params_shape, params):
    pointer = 0
    if hasattr(network, '_config'):
        device = network._config.device
    else:
        device = torch.device("cpu")

    for key_name, values in network.named_parameters():
        # get the length of the parameters
        len_param = int(np.prod(params_shape[key_name]))
        copy_params = params[pointer:pointer + len_param].reshape(params_shape[key_name])
        copy_params = torch.tensor(copy_params).to(device)
        # copy the params
        values.data.copy_(copy_params.data)
        # update the pointer
        pointer += len_param
