import numpy as np
import scipy.ndimage
import deepdish as dd
from functools import cached_property
from brian2 import *

#defaultclock.dt = 1*ms
directions = ['EE', 'EI', 'IE', 'II']
axes = ['xloc', 'yloc', 'zloc']

defaults = {
    'delay': {'EE': 1.5, 'EI': 0.8, 'IE': 0.8, 'II': 0.8},
    'p_connection': {d: 1.0 for d in directions}
}


def ensure_unit(value, unit):
    if isinstance(value, dict):
        return {key: ensure_unit(val, unit) for key, val in value.items()}
    if isinstance(value, Quantity):
        # value must already be in units [unit]
        assert not isinstance(value/unit, Quantity)
    else:
        value = value * unit
    return value


def add_state_variables(eqns, varlist, constlist, exclusions=[]):
    for varname, eq in eqns.items():
        if varname.endswith('/dt'):
            varname = varname[1:-3]
        if varname in exclusions:
            continue
        if '(constant)' in eq:
            constlist.append(varname)
        else:
            varlist.append(varname)


class SNN():
    def __init__(self,
                 # network params
                 N=100,
                 inhibitory_ratio=0.2,
                 N_nets=1,
                 # neuron params
                 model='escape-noise',
                 tau=30,
                 ref_period_exc=3, ref_period_inh=2,  # in ms
                 rate=1.0,  # in Hz
                 v_th=-54, v_rest=-74, v_reset=-74,  # in mV
                 # threshold adaptation (model='ALIF')
                 th_tau=1*second,
                 th_ampl=1*mV,
                 # synapse params
                 longterm_plasticity='stdp',
                 E_exc=0, E_inh=-100,  # in mV
                 tau_ampa=2, tau_gaba=4,  # in ms
                 delay=defaults['delay'],  # {direction: ms}
                 w0=0.0,
                 gmax_exc=1.0, gmax_inh=1.0,
                 p_connection=1.0,
                 # short-term depression
                 U=0.4, delta_U=0.4,
                 tau_rec=150,  # in ms
                 # excitatory STDP
                 beta=1.0,
                 tau_estdp=20,  # in ms
                 delta_estdp=1.0e-3,
                 # inhibitory STDP
                 alpha=1.0,
                 tau_1=10, tau_2=20,  # in ms
                 delta_istdp=1.0e-3,
                 # input layer configuration
                 input_strength=30.,
                 # membrane voltage fluctuations
                 vnoise_mean=0, vnoise_std=0,  # in mV
                 vnoise_frac=None,  # Relative to sum(vnoise_frac)
                 # record configuration
                 record=True,
                 record_state=False,
                 record_state_vars=('v', 'gtot_exc', 'gtot_inh'),
                 record_synapse=False,
                 record_synapse_vars=('w', 'x'),
                 record_syn_pre=False, record_syn_pre_vars=(),
                 record_syn_post=False, record_syn_post_vars=(),
                 # Simulation
                 dt=None,
                 # initialize weights
                 init_weights=True,
                 # spatial structure
                 layout='none',
                 connection_profile='circle',
                 radius_net=4, radius_exc=2, radius_inh=1,  # mm
                 conduction_velocity=0.15,  # m/s; overrides delay unless None
                 # structural randomization
                 rng=0,
                 identical_nets=True,  # Should independent nets be initialised
                                       # to the same structure?
                 # utilities
                 track_causality=False,  # False, 'cumulative', 'spike'
                 ):

        # network params
        self.N = N
        self.N_exc = int(self.N*(1-inhibitory_ratio))
        self.N_inh = self.N-self.N_exc
        self.inhibitory_ratio = inhibitory_ratio
        self.N_nets = max(1, int(N_nets))

        delta_1 = delta_istdp/(1.-(alpha*tau_1/tau_2))
        delta_2 = delta_1*alpha*tau_1/tau_2
        self.params = {
            # neuron params
            'tau': ensure_unit(tau, ms),
            'v_th': ensure_unit(v_th, mV),
            'v_rest': ensure_unit(v_rest, mV),
            'v_reset': ensure_unit(v_reset, mV),
            'ref_period_exc': ensure_unit(ref_period_exc, ms),
            'ref_period_inh': ensure_unit(ref_period_inh, ms),
            'th_tau': ensure_unit(th_tau, second),
            'th_ampl': ensure_unit(th_ampl, mV),
            # synapse params
            'E_exc': ensure_unit(E_exc, mV),
            'E_inh': ensure_unit(E_inh, mV),
            'tau_ampa': ensure_unit(tau_ampa, ms),
            'tau_gaba': ensure_unit(tau_gaba, ms),
            'w0': w0,
            'gmax_exc': gmax_exc, 'gmax_inh': gmax_inh,
            # short-term depression
            'U': U, 'delta_U': delta_U, 'tau_rec': ensure_unit(tau_rec, ms),
            # excitatory STDP
            'tau_estdp': ensure_unit(tau_estdp, ms),
            'deltaPre': delta_estdp, 'deltaPost': beta*delta_estdp,
            'beta': beta,
            # inhibitory STDP
            'tau_1': ensure_unit(tau_1, ms), 'tau_2': ensure_unit(tau_2, ms),
            'delta_1': delta_1, 'delta_2': delta_2,
            'alpha': alpha,
            # inputs
            'input_strength': input_strength
        }
        self.rate = ensure_unit(rate, Hz)
        self.gmax_exc, self.gmax_inh = gmax_exc, gmax_inh
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        elif rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(rng)
        self.identical_nets = identical_nets
        self.model = model
        if type(p_connection) == dict:
            self.p_connection = {**defaults['p_connection'], **p_connection}
        else:
            self.p_connection = {d: p_connection for d in directions}
        self.clock = Clock(defaultclock.dt if dt is None else dt,
                           'defaultclock')
        self._clocks = {}

        self.record = record
        self.record_state = record_state
        self.record_state_vars = record_state_vars
        self.record_synapse = record_synapse
        self.record_synapse_vars = record_synapse_vars
        self.record_syn_pre = record_syn_pre
        self.record_syn_post = record_syn_post
        self.record_syn_pre_vars = record_syn_pre_vars
        self.record_syn_post_vars = record_syn_post_vars

        dimensions = {'none': 0, 'circle': 2}
        self.spatial = {
            'layout': layout,
            'profile': connection_profile,
            'dimensions': dimensions[layout],
            'R': ensure_unit(radius_net, mm),
            'radius': {'E': ensure_unit(radius_exc, mm),
                       'I': ensure_unit(radius_inh, mm)},
            'velocity': None if conduction_velocity is None else ensure_unit(
                conduction_velocity, meter/second)
        }

        self.noisy, self.noise = self.sanitise_noise_args(
            vnoise_mean, vnoise_std, vnoise_frac)

        self._structure_setup_complete = False

        self.split_subnet_variables()
        self.setup_neuron_model()
        self.setup_synapse_models(delay, track_causality, longterm_plasticity)

        self.net = Network(self.G, self.S_EE, self.S_EI, self.S_IE, self.S_II)

        self.add_input()
        self.add_spike_monitor()
        self.add_state_monitor()

        if init_weights:
            self.set_structure()

    def set_structure(self, state=None):
        self.set_neuron_locations(state)
        self.set_neuron_noise(state)
        self.set_connectivity(state)
        self.set_multinet_values()
        self.add_synapse_monitors()
        self._structure_setup_complete = True

    def freq(self, vm):
        vm = ensure_unit(vm, mV)
        if self.model == 'escape-noise':
            return eval(self.rho_eqn, {
                'v': vm, 'v_th': self.params['v_th'],
                'freq_0': self.freq_0, 'mV': mV, 'exp': np.exp
            }) / self.clock.dt
        elif self.model == 'LIF' or self.model == 'ALIF':
            return np.asarray(vm > self.params['v_th'], float)

    def split_subnet_variables(self):
        self.extra_synaptic_variables = {}
        if np.asarray(self.params['alpha']).size != 1:
            if len(self.params['alpha']) != self.N_nets:
                raise ValueError('Alpha must be scalar, or of size N_nets.')
            self.extra_synaptic_variables['delta_1'] = self.params.pop(
                'delta_1')
            self.extra_synaptic_variables['delta_2'] = self.params.pop(
                'delta_2')
        if np.asarray(self.params['beta']).size != 1:
            if len(self.params['beta']) != self.N_nets:
                raise ValueError('Beta must be scalar, or of size N_nets.')
            self.extra_synaptic_variables['deltaPost'] = self.params.pop(
                'deltaPost')
        for simple_extra in ['gmax_exc', 'gmax_inh']:
            param = self.params[simple_extra]
            if np.asarray(param).size != 1:
                if len(param) != self.N_nets:
                    raise ValueError(f'{simple_extra} must be scalar, or '
                                     'of size N_nets.')
                self.extra_synaptic_variables[simple_extra] = self.params.pop(
                    simple_extra)

    def get_neuron_model_eqn(self):
        # LSM equations
        neuron_eqn_lines = {
            'dv/dt': '= ((v_rest-v)'
                     '  +(E_exc-v)*gtot_exc'
                     '  +(E_inh-v)*gtot_inh'
                     '  )/tau : volt (unless refractory)',
            'dgtot_exc/dt': '= -gtot_exc/tau_ampa : 1',
            'dgtot_inh/dt': '= -gtot_inh/tau_gaba : 1',
            'ref_period': ': second (constant)'
        }
        reset = 'v = v_reset'
        if self.model == 'escape-noise':
            # These two are members for the benefit of SNN.freq():
            self.rho_eqn = 'freq_0*exp((v-v_th)/(4*mV))'
            self.freq_0 = self.rate * self.clock.dt / np.exp(
                (self.params['v_rest']-self.params['v_th']) / (4*mV))
            neuron_eqn_lines['rho'] = f'= {self.rho_eqn} : 1'
            neuron_eqn_lines['freq_0'] = ': 1 (constant)'
            threshold = '(rand()<rho) and not_refractory'
        elif self.model == 'LIF':
            threshold = 'v > v_th'
        elif self.model == 'ALIF':
            threshold = 'v > v_th + th_adapt'
            neuron_eqn_lines['dth_adapt/dt'] = '= -th_adapt/th_tau : volt'
            reset = f'{reset}\nth_adapt += th_ampl'
            if isinstance(self.params['th_tau'], dict):
                neuron_eqn_lines['th_tau'] = ': second (constant)'
            if isinstance(self.params['th_ampl'], dict):
                neuron_eqn_lines['th_ampl'] = ': volt (constant)'
        if self.noisy:
            neuron_eqn_lines['dv/dt'] = '= ((v_rest-v)'\
                                        '  +(E_exc-v)*gtot_exc' \
                                        '  +(E_inh-v)*gtot_inh' \
                                        '  +vnoise_mean)/tau' \
                                        ' + vnoise_std*sqrt(2/tau)*xi'\
                                        ' : volt (unless refractory)'
            if self.noise['mean'].size > 1:
                neuron_eqn_lines['vnoise_mean'] = ': volt (constant)'
            if self.noise['std'].size > 1:
                neuron_eqn_lines['vnoise_std'] = ': volt (constant)'

        return neuron_eqn_lines, threshold, reset

    def setup_neuron_model(self):
        eqs_neuron, threshold, reset = self.get_neuron_model_eqn()
        self.G = NeuronGroup(
            self.N*self.N_nets,
            '\n'.join(f'{k} {v}' for k, v in eqs_neuron.items()),
            threshold=threshold, reset=reset,
            refractory='ref_period', method='euler',
            name='neurons', namespace=self.params,
            clock=self.clock)
        self.G.v = 'v_rest'
        self.G.gtot_exc = 0
        self.G.gtot_inh = 0
        if self.model == 'escape-noise':
            self.G.freq_0 = self.freq_0
        self.G_exc = self.G[:self.N_exc*self.N_nets]
        self.G_inh = self.G[self.N_exc*self.N_nets:]
        self.G_exc.ref_period = self.params['ref_period_exc']
        self.G_inh.ref_period = self.params['ref_period_inh']
        for pkey, pval in self.params.items():
            if pkey in eqs_neuron and isinstance(pval, dict):
                for gkey, g in (('E', self.G_exc), ('I', self.G_inh)):
                    setattr(g, pkey, pval[gkey])
        for axis in axes[:self.spatial['dimensions']]:
            for G in (self.G, self.G_inh, self.G_exc):
                G.add_attribute(axis)

        self.G.add_attribute('state_variables')
        self.G.add_attribute('state_constants')
        self.G.add_attribute('state_defaults')
        self.G.state_variables, self.G.state_constants = [], []
        add_state_variables(
            eqs_neuron, self.G.state_variables, self.G.state_constants)
        self.G.state_defaults = {
            k: 0 for k in self.G.state_variables + self.G.state_constants}
        self.G.state_defaults['v'] = 'v_rest'

    def set_neuron_locations(self, state=None):
        if self._structure_setup_complete:
            return
        if state is not None:
            for axis in axes[:self.spatial['dimensions']]:
                setattr(self.G, axis, ensure_unit(state[axis], mm))
        elif self.spatial['layout'] == 'circle':
            def xy(N):
                R = self.spatial['R'] / mm
                x, y = np.zeros((2, N))
                mask = np.ones(N, bool)
                while mask.sum() > 0:
                    x[mask], y[mask] = self.rng.uniform(-R, R, (2, mask.sum()))
                    mask[mask] = x[mask]**2 + y[mask]**2 > R**2
                return x, y
            if self.identical_nets:
                xe, ye = xy(self.N_exc)
                xi, yi = xy(self.N_inh)
                self.G.xloc = np.concatenate((np.tile(xe, self.N_nets),
                                              np.tile(xi, self.N_nets))) * mm
                self.G.yloc = np.concatenate((np.tile(ye, self.N_nets),
                                              np.tile(yi, self.N_nets))) * mm
            else:
                x, y = xy(self.N*self.N_nets)
                self.G.xloc, self.G.yloc = x*mm, y*mm
        elif self.spatial['layout'] != 'none':
            raise ValueError(f'Unknown layout: {self.spatial["layout"]}')

        for axis in axes[:self.spatial['dimensions']]:
            setattr(self.G_exc, axis,
                    getattr(self.G, axis)[:self.N_nets*self.N_exc])
            setattr(self.G_inh, axis,
                    getattr(self.G, axis)[self.N_nets*self.N_exc:])

    def sanitise_noise_args(self, mean, std, frac):
        if frac is None:
            frac = np.array([1.])
        else:
            frac = np.asarray(frac).ravel()
            frac /= frac.sum()
        mean, std = ensure_unit(mean, mV).ravel(), ensure_unit(std, mV).ravel()
        if mean.size not in (1, frac.size) or std.size not in (1, frac.size):
            raise ValueError('vnoise_* sizes inconsistent.')
        notnoisy = (frac.size == 1) and (
            frac[0] == 0 or mean[0] == std[0] == 0)
        return not notnoisy, {'mean': mean, 'std': std, 'frac': frac}

    def set_neuron_noise(self, state=None):
        if self.noisy and not self._structure_setup_complete:
            if state is None and self.noise['frac'].size > 1:
                def distribute(N):
                    n_in_frac = (self.noise['frac'] * N).astype(int)
                    while n_in_frac.sum() < N:
                        n_in_frac[self.rng.choice(len(n_in_frac))] += 1
                    idx = np.concatenate([np.repeat(i, n)
                                          for i, n in enumerate(n_in_frac)])
                    self.rng.shuffle(idx)
                    return idx
                if self.identical_nets:
                    ifrac_exc = np.repeat(distribute(self.N_exc), self.N_nets)
                    ifrac_inh = np.repeat(distribute(self.N_inh), self.N_nets)
                else:
                    ifrac_exc = np.concatenate([distribute(self.N_exc)
                                                for _ in self.N_nets])
                    ifrac_inh = np.concatenate([distribute(self.N_inh)
                                                for _ in self.N_nets])
                ifrac = np.concatenate([ifrac_exc, ifrac_inh])
            if self.noise['mean'].size == 1:
                self.params['vnoise_mean'] = self.noise['mean'][0]
            elif state is None:
                self.G.vnoise_mean = self.noise['mean'][ifrac]
            else:
                self.G.vnoise_mean = ensure_unit(state['vnoise_mean'], mV)
            if self.noise['std'].size == 1:
                self.params['vnoise_std'] = self.noise['std'][0]
            elif state is None:
                self.G.vnoise_std = self.noise['std'][ifrac]
            else:
                self.G.vnoise_std = ensure_unit(state['vnoise_std'], mV)

    def setup_synapse_models(self, delays, track_causality, ltplast):
        extra_synaptic_variables_def = '\n'.join(
            f'{k} : 1' for k in self.extra_synaptic_variables)
        exclude_from_state = []
        if track_causality in ('spike', 'cumulative'):
            # Note: v_post is offset by dt via vprev to account for the order
            # of integration. I have confirmed correctness separately. ~kernfel
            decay = '-vsyn' if track_causality == 'spike' else ''
            extra_synaptic_variables_def += f'''
        dgsyn/dt = -gsyn/{{tau}} : 1 (clock-driven)
        dvprev/dt = -vprev/dt + v_post/dt : volt (clock-driven)
        dvsyn/dt = ({decay} + ({{E}}-vprev)*gsyn) / tau * not_refractory: volt (clock-driven)
        '''
            exclude_from_state += ['gsyn', 'vprev', 'vsyn']
            extra_onpre = 'gsyn += U*x*w*{gmax}'
            if track_causality == 'spike':
                extra_onpost = 'vsyn = 0*volt'
            elif track_causality == 'cumulative':
                extra_onpost = ''
        else:
            extra_onpre = ''
            extra_onpost = ''

        if (
            (self.record_synapse and 'x' in self.record_synapse_vars)
            or (self.record_syn_pre and 'x' in self.record_syn_pre_vars)
        ):
            x_drive = 'clock'
        else:
            x_drive = 'event'

        eqs_synapse_E = f'''
        dx/dt = (1-x)/tau_rec : 1 ({x_drive}-driven)
        w : 1
        {extra_synaptic_variables_def.format(tau='tau_ampa', E='E_exc')}
        '''

        eqs_synapse_I = f'''
        dx/dt = (1-x)/tau_rec : 1 ({x_drive}-driven)
        w : 1
        {extra_synaptic_variables_def.format(tau='tau_gaba', E='E_inh')}
        '''

        on_pre_E = f'''
        {extra_onpre.format(gmax='gmax_exc')}
        gtot_exc_post += U*x*w*gmax_exc
        x -= delta_U*x
        '''

        on_post_E = extra_onpost

        on_pre_I = f'''
        {extra_onpre.format(gmax='gmax_inh')}
        gtot_inh_post += U*x*w*gmax_inh
        x -= delta_U*x
        '''

        on_post_I = extra_onpost

        if ltplast == 'stdp':
            eqs_synapse_E += '''
            dapre/dt = -apre/tau_estdp : 1 (event-driven)
            dapost/dt = -apost/tau_estdp : 1 (event-driven)
            '''

            eqs_synapse_I += '''
            da1pre/dt = -a1pre/tau_1 : 1 (event-driven)
            da2pre/dt = -a2pre/tau_2 : 1 (event-driven)
            da1post/dt = -a1post/tau_1 : 1 (event-driven)
            da2post/dt = -a2post/tau_2 : 1 (event-driven)
            '''

            on_pre_E += '''
            apre += deltaPre
            w = clip(w+apost, 0, 1)
            '''

            on_post_E += '''
            apost -= deltaPost
            w = clip(w+apre, 0, 1)
            '''

            on_pre_I += '''
            a1pre += delta_1
            a2pre += delta_2
            w = clip(w+a1post-a2post, 0, 1)
            '''

            on_post_I += '''
            a1post += delta_1
            a2post += delta_2
            w = clip(w+a1pre-a2pre, 0, 1)
            '''

        if self.spatial['layout'] == 'none' \
                or self.spatial['velocity'] is None:
            delay = {**defaults['delay'], **delays}
            delay = {d: ensure_unit(delay[d], ms) for d in directions}
        else:
            # Set after connecting
            delay = {d: None for d in directions}

        self.S_EE = Synapses(self.G_exc, self.G_exc, eqs_synapse_E,
                             on_pre=on_pre_E, on_post=on_post_E,
                             delay=delay['EE'], method='euler',
                             name='synapses_EE', clock=self.clock)
        self.S_EI = Synapses(self.G_exc, self.G_inh, eqs_synapse_E,
                             on_pre=on_pre_E, on_post=on_post_E,
                             delay=delay['EI'], method='euler',
                             name='synapses_EI', clock=self.clock)
        self.S_IE = Synapses(self.G_inh, self.G_exc, eqs_synapse_I,
                             on_pre=on_pre_I, on_post=on_post_I,
                             delay=delay['IE'], method='euler',
                             name='synapses_IE', clock=self.clock)
        self.S_II = Synapses(self.G_inh, self.G_inh, eqs_synapse_I,
                             on_pre=on_pre_I, on_post=on_post_I,
                             delay=delay['II'], method='euler',
                             name='synapses_II', clock=self.clock)
        self.S_all = {'EE': self.S_EE, 'EI': self.S_EI,
                      'IE': self.S_IE, 'II': self.S_II}

        vardict = {'E': {}, 'I': {}}
        for line in eqs_synapse_E.split('\n'):
            if ':' in line:
                key, *val = line.split(':')
                vardict['E'][key.split('=')[0].strip()] = ':'.join(val)
        for line in eqs_synapse_I.split('\n'):
            if ':' in line:
                key, *val = line.split(':')
                vardict['I'][key.split('=')[0].strip()] = ':'.join(val)
        for d, S in self.S_all.items():
            S.add_attribute('state_variables')
            S.add_attribute('state_constants')
            S.add_attribute('state_defaults')
            S.state_variables, S.state_constants = [], []
            add_state_variables(
                vardict[d[0]], S.state_variables, S.state_constants,
                exclude_from_state)
            S.state_defaults = {
                k: 0 for k in S.state_variables + S.state_constants}
            S.state_defaults['x'] = 1
            S.state_defaults['w'] = self.params['w0']

    def add_input(self):
        if self.params['input_strength'] is not None:
            self.input = SpikeGeneratorGroup(self.N*self.N_nets,
                                             [], []*ms, name='input',
                                             clock=self.clock)
            self.S_input = Synapses(
                self.input, self.G, name='syn_input',
                on_pre=f'gtot_exc_post+={self.params["input_strength"]}',
                method='exact', clock=self.clock)
            self.S_input.connect(j='i')
            self.net.add(self.input)
            self.net.add(self.S_input)

    def add_spike_monitor(self):
        if self.record:
            self.spikemon_liquid = SpikeMonitor(self.G)
            self.net.add(self.spikemon_liquid)
        else:
            self.spikemon_liquid = None

    def get_clock(self, dt):
        if isinstance(dt, Quantity):
            return self._clock(ensure_unit(dt, ms)/ms)
        else:
            return None

    def _clock(self, dt_in_ms):
        if dt_in_ms*ms == self.clock.dt:
            return self.clock
        elif dt_in_ms not in self._clocks:
            self._clocks[dt_in_ms] = Clock(dt_in_ms*ms)
        return self._clocks[dt_in_ms]

    def add_state_monitor(self):
        if self.record_state:
            self.statemon_liquid = StateMonitor(
                self.G, self.record_state_vars, record=True,
                clock=self.get_clock(self.record_state))
            self.net.add(self.statemon_liquid)
        else:
            self.statemon_liquid = None

    def add_synapse_monitors(self):
        if self._structure_setup_complete:
            return
        self.synapse_monitors = {} if self.record_synapse else None
        self.synapse_monitors_pre = {} if self.record_syn_pre else None
        self.synapse_monitors_post = {} if self.record_syn_post else None
        if not self.record_synapse and not self.record_syn_pre \
                and not self.record_syn_post:
            return

        for direction, S in self.S_all.items():
            pre, post = self.connectivity[direction]
            if self.record_synapse:
                self.synapse_monitors[direction] = StateMonitor(
                    S, self.record_synapse_vars,
                    record=range(len(pre)),
                    clock=self.get_clock(self.record_synapse))
                self.net.add(self.synapse_monitors[direction])
            if self.record_syn_pre \
                    and direction[0] not in self.synapse_monitors_pre:
                self.synapse_monitors_pre[direction[0]] = StateMonitor(
                    S, self.record_syn_pre_vars,
                    record=np.unique(pre, return_index=True)[1],
                    clock=self.get_clock(self.record_syn_pre))
                self.net.add(self.synapse_monitors_pre[direction[0]])
            if self.record_syn_post \
                    and direction[1] not in self.synapse_monitors_post:
                self.synapse_monitors_post[direction[1]] = StateMonitor(
                    S, self.record_syn_post_vars,
                    record=np.unique(post, return_index=True)[1],
                    clock=self.get_clock(self.record_syn_post))
                self.net.add(self.synapse_monitors_post[direction[1]])

    def set_connectivity(self, state=None):
        denovo = state is None
        state = {} if denovo else state
        if not self._structure_setup_complete:
            self.connectivity = {}
        for d, S in self.S_all.items():
            if self._structure_setup_complete:
                pre, post = self.connectivity[d]
            else:
                if denovo:
                    pre, post = self._get_synapse_connections(d)
                else:
                    pre, post = state[f'i_{d}'], state[f'j_{d}']
                S.connect(i=pre, j=post)
                self.connectivity[d] = pre, post
            if denovo or f'w_{d}' in state or '_reset' in state:
                S.w = state.get(f'w_{d}', self.params['w0'])
            if self.spatial['layout'] != 'none' \
                    and self.spatial['velocity'] is not None:
                distance = self.get_synaptic_distance(d)
                S.delay = distance / self.spatial['velocity']
        for d, S in self.S_all.items():
            for k in S.state_variables:
                key = f'{k}_{d}'
                if denovo or key in state or '_reset' in state:
                    setattr(S, k, state.get(key, S.state_defaults[k]))

    def _get_synapse_connections(self, direction):
        N = {'E': self.N_exc, 'I': self.N_inh}
        G = {'E': self.G_exc, 'I': self.G_inh}
        nPre, nPost = [N[p] for p in direction]
        gPre, gPost = [G[p] for p in direction]
        net = np.arange(self.N_nets).reshape(-1, 1, 1)
        pre = np.arange(nPre).reshape(1, -1, 1)
        post = np.arange(nPost).reshape(1, 1, -1)
        net, pre, post = np.broadcast_arrays(net, pre, post)
        pre = pre + net*nPre
        post = post + net*nPost
        r_pre = self.spatial['radius'][direction[0]]
        if self.spatial['dimensions'] > 0:
            distance = self.get_synaptic_distance(direction, pre, post)
            if self.spatial['profile'] == 'circle':
                C = distance < r_pre
            elif self.spatial['profile'] == 'gaussian':
                # Candidate selection uses a Gaussian with peak 1:
                resolution = np.quantile(distance.ravel()/mm, .1)*mm
                nbins = int(np.ceil(2*self.spatial['R'] / resolution))
                bins = np.arange(nbins+1) * resolution
                p = np.exp(-bins**2/r_pre**2)
                C = np.zeros(net.shape, bool)
                for i in range(self.N_nets):
                    if i == 0 or not self.identical_nets:
                        binned_dist = np.digitize(distance[i].ravel(),
                                                  bins=bins)
                        nChoose = (p * np.sum(binned_dist == 1)).astype(int)
                        idx = []
                        for bin in range(1, nbins+1):
                            nTotal = np.sum(binned_dist == bin)
                            print(bin, bins[bin], nTotal, nChoose[bin-1])
                            choose = self.rng.choice(
                                nTotal, min(nTotal, nChoose[bin-1]),
                                replace=False, shuffle=False)
                            idx.append(
                                np.nonzero(binned_dist == bin)[0][choose])
                        idx = np.concatenate(idx)
                    C[i].ravel()[idx] = True
        else:
            C = np.ones(net.shape, bool)
        if direction[0] == direction[1]:
            for c in C:
                np.fill_diagonal(c, False)
        if self.p_connection[direction] < 1:
            for i in range(self.N_nets):
                """if i == 0 or not self.identical_nets:
                    # Use rng.choice to preserve equal numbers across
                    # non-identical subnets:
                    nTotal = np.count_nonzero(C[i])
                    nChoose = int(self.p_connection[direction] * nTotal + .5)
                    idx = self.rng.choice(
                        np.nonzero(C[i].ravel())[0], nTotal-nChoose,
                        replace=False, shuffle=False)
                C[i].ravel()[idx] = False"""
                random_M = self.rng.uniform(size=C[i].shape)
                threshold=np.ones(C[i].shape) * self.p_connection[direction]
                C[i] &= (random_M < threshold)
        return pre[C], post[C]

    def get_synaptic_distance(self, direction, pre=None, post=None):
        if pre is None:
            pre = self.connectivity[direction][0]
        if post is None:
            post = self.connectivity[direction][1]
        G = {'E': self.G_exc, 'I': self.G_inh}
        gPre, gPost = [G[p] for p in direction]
        sqdist = np.zeros(pre.shape) * meter**2
        for axis in axes[:self.spatial['dimensions']]:
            sqdist += (
                getattr(gPre, axis)[pre] - getattr(gPost, axis)[post]
            )**2
        return np.sqrt(sqdist)

    def set_multinet_values(self):
        for direction, S in self.S_all.items():
            pre, post = self.connectivity[direction]
            N_connections = len(pre) // self.N_nets
            for key, net_value in self.extra_synaptic_variables.items():
                syn_value = np.empty((self.N_nets, N_connections))
                syn_value.T[:] = net_value
                setattr(S, key, syn_value.flatten())

    def set_spikes(self, spikes_i, spikes_t):
        self.input.set_spikes(spikes_i, spikes_t)

    #def set_states(self, states):
    #    self.net.set_states(states)

    #def set_weights(self, params):
    #    for direction, S in zip(('EE', 'EI', 'IE', 'II'), (self.S_EE, self.S_EI, self.S_IE, self.S_II)):
    #        W = params['w_'+direction]
    #        sources, targets = np.where(np.isnan(W)==False)
    #        S.connect(i=sources, j=targets)
    #        S.w = W[sources, targets]
    #        S.x = 1
    #    var = ['x', 'w', 'g']
    #    if self.record_synapse and (self.synapse_monitors is None):
    #        self.synapse_monitors = {}
    #        for direction, S in zip(('EE', 'EI', 'IE', 'II'), (self.S_EE, self.S_EI, self.S_IE, self.S_II)):
    #            #var = ['x', 'w', 'g'] if direction[0]=='E' else ['x', 'w', 'g']
    #            self.synapse_monitors[direction] = StateMonitor(S, var, record=True)
    #            self.net.add(self.synapse_monitors[direction])

    def expand_state(self, state):
        _state = {}
        nvars = self.G.state_variables + self.G.state_constants
        for axis in axes[:self.spatial['dimensions']]:
            nvars.append(axis)
        for var in nvars:
            if var in state:
                try:
                    assert len(state[var]) == self.N
                    _state[var] = np.concatenate([
                        np.tile(state[var][:self.N_exc], self.N_nets),
                        np.tile(state[var][self.N_exc:], self.N_nets)
                    ])
                except TypeError:  # no len()
                    _state[var] = state[var]
        svars = []
        for d, S in self.S_all.items():
            svars += [f'{v}_{d}' for v in S.state_variables]
            svars += [f'{v}_{d}' for v in S.state_constants]
        for var in svars:
            if var in state:
                try:
                    assert len(state[var])
                    _state[var] = np.concatenate([state[var]]*self.N_nets)
                except TypeError:  # no len()
                    _state[var] = state[var]
        for var in ('i_EE', 'i_EI', 'j_EE', 'j_IE'):
            assert var in state and np.all(state[var] < self.N_exc)
            _state[var] = (
                np.arange(self.N_nets).reshape(-1, 1) * self.N_exc + state[var]
            ).reshape(-1)
        for var in ('i_IE', 'i_II', 'j_EI', 'j_II'):
            assert var in state and np.all(state[var] < self.N_inh)
            _state[var] = (
                np.arange(self.N_nets).reshape(-1, 1) * self.N_inh + state[var]
            ).reshape(-1)
        return _state

    def initialize_with(self, state):
        '''
        Initialize state and structural variables.
        If this is the first init (after constructing with init_weights=False),
            structural variables (location, noise, connectivity)
            must be set. Otherwise, these are considered hard-wired and not
            overwritten by entries in `state`.
        Missing values in `state` are ignored (i.e. not overwritten), unless
            the key `_reset` is included in `state`, in which case missing
            *state* values (but not structural ones) are filled in with
            defaults. Structural values must be present in full on first init.
        '''
        if 'v' in state or '_reset' in state:
            self.G.v = ensure_unit(state.get('v', self.params['v_rest']), volt)
        if 'gtot_exc' in state or '_reset' in state:
            self.G.gtot_exc = state.get('gtot_exc', 0)
        if 'gtot_inh' in state or '_reset' in state:
            self.G.gtot_inh = state.get('gtot_inh', 0)
        self.set_structure(state)

    def store(self, name='default', filename=None):
        self.net.store(name=name, filename=None)

    def restore(self, name='default', filename=None, restore_random_state=False):
        self.net.restore(name=name, filename=None,
                         restore_random_state=restore_random_state)

    def run(self, runtime, return_spikes=True, **kwargs):
        self.net.run(runtime, namespace=self.params, **kwargs)
        if return_spikes and self.spikemon_liquid is not None:
            return self.spikemon_liquid.i[:], self.spikemon_liquid.t[:]
        else:
            return None, None

    #def get_states(self, read_only_variables=True):
    #    return self.net.get_states()

    def get_rates(self, spike_dict=None, smooth=False, sigma=10*ms, norm=True):
        '''
        Calculates population rates in Hz (but returned unitless).
        Parameters:
            spike_dict: externally provided spike data, containing entries
                'i': spiking neurons' ID,
                't': spike time (unit : second),
                'tmax': recording duration
            smooth: boolean, whether or not to smooth the rate with a
                Gaussian filter (cf. sigma)
            sigma: explicit seconds, Standard deviation of Gaussian smoothing
                kernel
            norm:
                True to return mean single-unit rate
                False to return total population rate
        Return:
            rates_exc: Excit population rates as (N_nets,) list of (t,) arrays
            rates_inh: Inhib population rates as (N_nets,) list of (t,) arrays
            rates_all: Total population rates as (N_nets,) list of (t,) arrays
            timebase: Time base; (t,) array.
        '''
        sigma = int(sigma / self.clock.dt)
        tnorm = self.clock.dt / second
        if norm:
            norms = tnorm * np.asarray([self.N_exc, self.N_inh, self.N])
        else:
            norms = np.repeat(tnorm, 3)
        norm_exc, norm_inh, norm_all = norms

        # load data
        if spike_dict is None:
            spike_i = self.spikemon_liquid.i
            spike_t = self.spikemon_liquid.t
            tmax = self.net.t
        else:
            spike_i = spike_dict['i']
            minimal_digit = int(
                np.log10(1/(self.clock.dt/second)))
            # round spike_t with minimal digits
            spike_t = np.round(spike_dict['t'], minimal_digit) * second
            tmax = spike_dict['tmax']

        rates_exc, rates_inh, rates_all = [], [], []
        T = np.arange(
            0, tmax, self.clock.dt).reshape(-1, 1)
        if smooth:
            mask = ~np.isnan(scipy.ndimage.gaussian_filter1d(
                T[:, 0], sigma, mode='constant', cval=np.nan))
            timebase = T[mask, 0]
        else:
            timebase = T[:, 0]

        for net in range(self.N_nets):
            exc = ((spike_i > self.N_exc*net)
                   * (spike_i < self.N_exc*(net+1)))
            inh = ((spike_i > self.N_exc*self.N_nets + self.N_inh*net)
                   * (spike_i < self.N_exc*self.N_nets + self.N_inh*(net+1)))
            irate_exc = np.sum(T == spike_t[exc], axis=1)*1.
            irate_inh = np.sum(T == spike_t[inh], axis=1)*1.
            irate_all = irate_exc + irate_inh
            if smooth:
                irate_exc = scipy.ndimage.gaussian_filter1d(
                    irate_exc, sigma, mode='constant')[mask]/norm_exc
                irate_inh = scipy.ndimage.gaussian_filter1d(
                    irate_inh, sigma, mode='constant')[mask]/norm_inh
                irate_all = scipy.ndimage.gaussian_filter1d(
                    irate_all, sigma, mode='constant')[mask]/norm_all
            rates_exc.append(irate_exc)
            rates_inh.append(irate_inh)
            rates_all.append(irate_all)
        if self.N_nets == 1:
            return rates_exc[0], rates_inh[0], rates_all[0], timebase
        else:
            return rates_exc, rates_inh, rates_all, timebase

    def get_liquid_voltage_trace(self):
        return self.statemon_liquid.v, self.statemon_liquid.t

    def get_state_monitor(self):
        return self.statemon_liquid

    def get_synapse_monitors(self):
        return self.synapse_monitors

    def get_weight_matrix(self, direction=None, force_square=False, Wdict=None,
                          subnet=0, missing=0., signed=False, as_gmax=False):
        '''
        Constructs a square (pre, post) weight matrix, where both pre and post
            run over E, then I.
        Args:
            direction: str, selects a block ('EE' etc.) of the matrix and
                returns this exclusively.
            force_square: If True, returns the selected direction in the
                context of an otherwise empty complete square weight matrix.
            Wdict: A dictionary containing linearised weights for all
                directions. May be keyed as either 'EE' or 'w_EE' etc. Axes
                beyond the first will be preserved in the returned weight
                matrix, yielding a shape (pre, post, *).
                If None (default), the instance's weights are used instead.
            subnet: int, the subnet index. This can be used both to select a
                subnet from self, in case of heterogeneous sparsity, as well as
                to select a subnet from Wdict, in case this is not already
                separated out. If None, the weight matrix is collected across
                all subnets, yielding a shape (subnets, pre, post, *).
            missing: float, value for missing (disconnected) weights.
            signed: bool. If True, apply a negative sign to weights from
                inhibitory neurons.
            as_gmax: bool. If True, multiply the raw weights with their
                respective `gmax_*` value. Note that this does not imply
                `signed`, as the gmax values are themselves unsigned.
        '''
        if subnet is None:
            return np.stack([self.get_weight_matrix(
                direction=direction, force_square=force_square, Wdict=Wdict,
                subnet=net, missing=missing, signed=signed, as_gmax=as_gmax
            ) for net in range(self.N_nets)])

        def wd(adict, d):
            return adict[d if d in adict else f'w_{d}']
        # Synapse.source/target bounds; these are local to G_exc/G_inh
        syn_lo = {'E': subnet*self.N_exc, 'I': subnet*self.N_inh}
        syn_hi = {'E': (subnet+1)*self.N_exc, 'I': (subnet+1)*self.N_inh}
        # Contiguous weight matrix bounds; these are local to the subnet
        W_lo = {'E': 0, 'I': self.N_exc}
        W_hi = {'E': self.N_exc, 'I': self.N}
        idx = self.weight_index
        if Wdict is None:
            Wdict = self.get_weights()
        if len(wd(Wdict, 'EE').shape) > 1:
            shape = (self.N, self.N) + wd(Wdict, 'EE').shape[1:]
        else:
            shape = (self.N, self.N)
        W = np.full(shape, missing, dtype=float)
        for pre in ('E', 'I'):
            sign = -1 if signed and pre == 'I' else 1
            if as_gmax:
                key = 'gmax_exc' if pre == 'E' else 'gmax_inh'
                if key in self.params:
                    gmax = self.params[key]
                else:
                    gmax = self.extra_synaptic_variables[key][subnet]
            else:
                gmax = 1
            for post in ('E', 'I'):
                d = f'{pre}{post}'
                if direction is not None and direction != d:
                    continue
                w = wd(Wdict, d)
                mask = (idx[f'i_{d}'] >= syn_lo[pre]) \
                    * (idx[f'i_{d}'] < syn_hi[pre]) \
                    * (idx[f'j_{d}'] >= syn_lo[post]) \
                    * (idx[f'j_{d}'] < syn_hi[post])
                i = idx[f'i_{d}'][mask] - syn_lo[pre] + W_lo[pre]
                j = idx[f'j_{d}'][mask] - syn_lo[post] + W_lo[post]
                if w.shape[0] == len(mask):
                    W[i, j] = sign * gmax * w[mask]
                else:
                    W[i, j] = sign * gmax * w
                if direction is not None and not force_square:
                    return W[W_lo[pre]:W_hi[pre], W_lo[post]:W_hi[post]]
        return W

    #def get_params(self):
    #    param_dict = {
    #        "N": self.N, "N_exc": self.N_exc, "N_inh": self.N_inh,
    #        "inhibitory_ratio": self.inhibitory_ratio,
    #        "rate": self.rate,
    #        "gmax_exc": self.gmax_exc, "gmax_inh": self.gmax_inh,
    #        "p_connection": self.p_connection
    #        }
    #    for key, value in self.params.items():
    #        param_dict[key] = value
    #    return param_dict

    def save_weights(self, filename, with_weight_index=True):
        dd.io.save(filename, self.get_weights(with_weight_index))

    @cached_property
    def weight_index(self):
        idx = {}
        for d, S in self.S_all.items():
            idx[f'i_{d}'], idx[f'j_{d}'] = S.i[:], S.j[:]
        return idx

    def get_weights(self, with_weight_index=False):
        w = {}
        for d, S in self.S_all.items():
            w[f'w_{d}'] = S.w[:]
        if with_weight_index:
            w.update(self.weight_index)
        return w

    def save_states(self, filename, with_weight_index=True):
        dd.io.save(filename, self.get_state(with_weight_index))

    def get_state(self, with_weight_index=True, subnet=None):
        node_state = {k: getattr(self.G, k)[:]
                      for k in self.G.state_variables + self.G.state_constants}
        link_state = {f'{k}_{d}': getattr(S, k)[:]
                      for d, S in self.S_all.items()
                      for k in S.state_variables + S.state_constants}
        if with_weight_index:
            link_state.update(self.weight_index)
        for axis in axes[:self.spatial['dimensions']]:
            node_state[axis] = getattr(self.G, axis)[:]/mm
        if subnet is not None:
            assert subnet >= 0 and subnet < self.N_nets
            weight_index = self.weight_index
            elo, ehi = subnet*self.N_exc, (subnet+1)*self.N_exc
            ilo, ihi = subnet*self.N_inh, (subnet+1)*self.N_inh
            for key in node_state:
                node_state[key] = node_state[key][np.concatenate(
                    (np.arange(elo, ehi),
                     np.arange(ilo, ihi) + self.N_exc*self.N_nets))]

            pre_post = (('E', elo, ehi), ('I', ilo, ihi))
            for pre, pre_lo, pre_hi in pre_post:
                for post, post_lo, post_hi in pre_post:
                    d = f'{pre}{post}'
                    mask = \
                        (weight_index[f'i_{d}'] >= pre_lo) * \
                        (weight_index[f'i_{d}'] < pre_hi) * \
                        (weight_index[f'j_{d}'] >= post_lo) * \
                        (weight_index[f'j_{d}'] < post_hi)
                    for key in link_state:
                        if key.endswith(d):
                            link_state[key] = link_state[key][mask]
                    if with_weight_index:
                        link_state[f'i_{d}'] -= pre_lo
                        link_state[f'j_{d}'] -= post_lo
        return {**node_state, **link_state}

    def get_subnet_indices(self, subnet):
        return np.concatenate([
            np.arange(self.N_exc) + self.N_exc*subnet,
            np.arange(self.N_inh) + self.N_exc*self.N_nets + self.N_inh*subnet
        ])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # import brian2genn
    # set_device('genn')

    # Simulation setting
    runtime = 10*second
    seed(0)

    # Generate input spike trains
    #input_rate = 1*Hz
    #P = PoissonGroup(n_input, rates=input_rate)
    #MP = SpikeMonitor(P)
    #net = Network(P, MP)
    #net.run(runtime)
    #spikes_i = MP.i
    #spikes_t = MP.t

    defaultclock.dt = 1*msecond

    snn = SNN(N=100,
              inhibitory_ratio=0.2,
              record=True, record_state=True, record_synapse=True,
              model='LIF',
              vnoise_mean=(0, 5), vnoise_std=5, vnoise_frac=(.7, .3),
              # layout='circle'
              )

    liquid_spike_i, liquid_spike_t = snn.run(runtime)
    liquid_state_v, liquid_state_t = snn.get_liquid_voltage_trace()
    statemon_n = snn.get_state_monitor()
    statemon_s = snn.get_synapse_monitors()

    for S in (snn.S_EE, snn.S_EI, snn.S_IE, snn.S_II):
        print(np.mean(S.w), np.max(S.w))

    # Plot results
    neuron_id = 0
    synapses = snn.S_EE[neuron_id, :]
    pdf = PdfPages('SNN.pdf')

    fig, axs = plt.subplots(nrows=4, figsize=(12, 7), sharex=True)
    axs[0].plot(liquid_state_t/second, liquid_state_v[neuron_id]/mV)
    for t in liquid_spike_t[liquid_spike_i == neuron_id]/second:
        axs[0].axvline(t, ls='--', c='C1', lw=0.5)
    axs[0].set_ylabel('v [mV]')
    for trace in statemon_s['EE'][synapses].x:
        axs[1].plot(statemon_s['EE'].t/second, trace)
    # axs[1].plot(statemon_n.t/second, statemon_n[neuron_id].x)
    axs[1].set_ylabel('x []')
    axs[2].plot(statemon_n.t/second, statemon_n[neuron_id].gtot_exc)
    axs[2].set_ylabel('gtot_exc []')
    axs[3].plot(statemon_n.t/second, statemon_n[neuron_id].gtot_inh)
    axs[3].set_ylabel('gtot_inh []')
    axs[3].set_xlim(0, runtime/second)
    pdf.savefig(fig)

    fig, axs = plt.subplots(nrows=2, figsize=(12, 7), sharex=True)
    rates_exc, rates_inh, rates_all, t_rates = snn.get_rates()
    axs[0].plot(t_rates, rates_exc)
    axs[0].plot(t_rates, rates_inh)
    axs[0].plot(t_rates, rates_all, 'k')
    axs[0].set_ylabel('Firing rates [Hz]')
    axs[0].set_xlim(0, runtime/second)
    try:
        for t in spikes_t/second:
            axs[1].axvline(t, ls='-', c='r', alpha=0.2, lw=1)
    except NameError:
        pass
    axs[1].plot(liquid_spike_t/second, liquid_spike_i, '.k', ms=1)
    axs[1].set_ylabel('Neuron index')
    axs[1].set_xlabel('Time, second')
    axs[1].set_xlim(0, runtime/second)
    pdf.savefig(fig)

    for direction in ('EE', 'EI', 'IE', 'II'):
        print(direction, statemon_s[direction].w.shape,
              np.mean(statemon_s[direction].w), np.max(statemon_s[direction].w))
        fig, axs = plt.subplots(nrows=2, figsize=(12, 3), sharex=True)
        trace = statemon_s[direction].w
        im = axs[0].imshow(trace, aspect='auto', origin='lower')
        axs[1].plot(np.mean(trace, axis=0), c='k', lw=1)
        axs[1].set_xlim(0, trace.shape[1])
        fig.colorbar(im, ax=axs)
        plt.suptitle(direction)
        pdf.savefig(fig)

    #fig, axs = plt.subplots(nrows=2, figsize=(5, 5))
    #hist, bin_edges = np.histogram(np.sum(binned_spike_trains, axis=0), bins=100)
    ##print hist
    ##print bin_edges
    #hist = hist[1:]
    #bin_edges = bin_edges[1:-1]
    #axs[0].plot(bin_edges, hist, c='k')
    #axs[0].set_xscale('log')
    #axs[0].set_yscale('log')
    #axs[1].plot(bin_edges, hist, c='k')

    #fig, axs = plt.subplots(nrows=6, figsize=(12, 7), sharex=True)
    #axs[0].plot(liquid_state_t/ms, liquid_state_v[neuron_id]/mV)
    #for t in liquid_spike_t[liquid_spike_i==neuron_id]/ms:
    #    axs[0].axvline(t, ls='--', c='C1', lw=0.5)
    #axs[0].set_ylabel('v [mV]')
    #axs[0].set_xlim(0, runtime/ms)
    #for trace in s_monitor_EE[synapses].x:
    #    axs[1].plot(s_monitor_EE.t/ms, trace)
    #axs[1].set_ylabel('x []')
    #axs[1].set_xlim(0, runtime/ms)
    #for trace in s_monitor_EE[synapses].c:
    #    axs[2].plot(s_monitor_EE.t/ms, trace)
    #axs[2].set_ylabel('c []')
    #axs[2].set_xlim(0, runtime/ms)
    #for trace in s_monitor_EE[synapses].w:
    #    axs[3].plot(s_monitor_EE.t/ms, trace)
    #axs[3].set_ylabel('w []')
    #axs[3].set_xlim(0, runtime/ms)
    #bin_size = 20
    #bin_edges, binned_spike_trains = snn.get_binned_spike_trains(bin_size=bin_size)
    #axs[4].plot(bin_edges+bin_size/2, np.sum(binned_spike_trains, axis=0), c='k')
    #axs[4].set_ylabel('Neuron index')
    #axs[4].set_xlim(0, runtime/ms)
    #try:
    #    for t in spikes_t/ms:
    #        axs[5].axvline(t, ls='-', c='r', alpha=0.2, lw=1)
    #except: pass
    #axs[5].plot(liquid_spike_t/ms, liquid_spike_i, '.k', ms=1)
    #axs[5].set_ylabel('Neuron index')
    #axs[5].set_xlabel('Time [ms]')
    #axs[5].set_xlim(0, runtime/ms)
    pdf.close()
