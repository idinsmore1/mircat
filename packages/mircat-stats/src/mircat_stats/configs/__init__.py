import os
import SimpleITK as sitk
from threadpoolctl import threadpool_limits


def set_num_threads(num_threads: int) -> None:
    """Set the number of threads available for every linear algebra related process (Numpy, Pytorch, SimpleITK, etc)
    :param num_threads: the number of threads
    :return: None
    """
    thread_vars = [
        *[f"{x}_NUM_THREADS" for x in ["OMP", "OPENBLAS", "MKL", "NUMEXPR"]],
        "VECLIB_MAXIMUM_THREADS",
    ]
    thread_dict = {var: str(num_threads) for var in thread_vars}
    os.environ.update(thread_dict)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(num_threads)
    # limiter = threadpool_limits(num_threads)
