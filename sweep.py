from importlib import import_module


def get_script(outdir, label=None):
    path = f'results/{outdir}'
    if label is None:
        script = import_module(f'{path.replace("/", ".")}.params')
    else:
        script = import_module(f'params.{label}')
    return script, path


if __name__ == '__main__':
    import sys
    import os
    import shutil
    from brian2 import *
    import deepdish as dd

    from SNN import SNN
    from run_utils import safe_set_device
    del input  # restore built-in input after brian2 overwrite

    if len(sys.argv) < 3:
        print(f'Usage: python {os.path.basename(__file__)} '
              'PARAMS OUTDIR [ITERATIONS] [SEED]')
        exit(1)

    label, outdir = sys.argv[1:3]
    script, path = get_script(outdir, label)

    os.makedirs(path, exist_ok=True)
    try:
        with open(f'{path}/params.py') as _:
            pass
        r = input(f'Existing output at {path} may be overwritten. Continue?')
        if 'n' in r:
            exit(1)
    except FileNotFoundError:
        pass

    try:
        dev = script.device
    except AttributeError:
        dev = 'cpp_standalone'

    if len(sys.argv) > 3:
        iterations = int(sys.argv[3])
    else:
        iterations = script.iterations

    if len(sys.argv) > 4:
        theseed = int(sys.argv[4])
    else:
        theseed = 0

    script.params['N_nets'] = script.params.get('N_nets', 1)
    script.params['input_strength'] = script.params.get('input_strength', None)

    safe_set_device(dev, os.environ.get('dev'))

    shutil.copy(f'params/{label}.py', f'{path}/params.py')
    with open(f'{path}/params.py', 'a') as file:
        file.write(
            f'iterations = {iterations}\n'
            f'device = "{dev}"\n'
            f'params["N_nets"] = {script.params["N_nets"]}\n'
            f'params["input_strength"] = {script.params["input_strength"]}\n'
        )

    for i in range(iterations):
        if i > 0:
            device.reinit()
            device.activate()
        if dev != 'genn':
            seed(theseed + i)

        snn = SNN(init_weights=(i == 0),
                  **script.params,
                  record_synapse=script.recording_period,
                  record_synapse_vars=['w'],
                  record=True)

        if i > 0:
            snn.initialize_with({
                **dd.io.load(f'{path}/maturation_{i-1}.h5'),
                **weight_index})

        spike_i, spike_t = snn.run(
            script.runtime, report='stdout', report_period=60*second)

        snn.save_states(f'{path}/maturation_{i}.h5',
                        with_weight_index=(i == 0))
        if i == 0:
            weight_index = snn.weight_index

        W = {}
        for key, monitor in snn.synapse_monitors.items():
            W[key] = monitor.w
        dd.io.save(f'{path}/W_{i}.h5', W)

        #save all spikes
        dd.io.save(f'{path}/spikes_{i}.h5', dict(i=spike_i, t=spike_t))

        print(f'Run {i+1}/{iterations} complete.')
