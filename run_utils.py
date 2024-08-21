from brian2.only import *
import multiprocessing as mp
try:
    import brian2genn
except ModuleNotFoundError:
    print('Note: Brian2GeNN not found.')
try:
    import brian2cuda
except ModuleNotFoundError:
    print('Note: Brian2CUDA not found')


def safe_set_device(dev, devid=None):
    set_device(dev)
    if dev == 'cpp_standalone':
        prefs.devices.cpp_standalone.openmp_threads = 1
    else:
        prefs.devices.cpp_standalone.openmp_threads = 0
    if dev == 'genn' and devid is not None:
        prefs.devices.genn.cuda_backend.device_select = 'MANUAL'
        prefs.devices.genn.cuda_backend.manual_device = int(devid)
    elif dev == 'cuda_standalone' and devid is not None:
        devices.cuda_standalone.cuda_backend.gpu_id = int(devid)
