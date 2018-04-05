import timeit
import pickle
import os
import errno
import datetime
import shutil
import warnings
import traceback
import pstats
from io import StringIO
import sys
import gc
import inspect
import argparse
import importlib
import re
from multiprocessing import Manager
import subprocess
import shlex
import platform
from pip import operations
import json
import contextlib
import stat
from argparse import Namespace
import tempfile

import numpy as np
from matplotlib import pyplot
from IPython.utils.capture import capture_output

from swutil import files, plots, np_tools
from swutil.collections import unique
from swutil.decorators import add_runtime, print_peak_memory
from swutil.logs import Log
from swutil.plots import plot_convergence
from swutil.validation import Positive, Integer
from swutil.aux import random_string

class GitError(Exception):
    def __init__(self, message, git_log):
        super(GitError, self).__init__(message)
        self.git_log = git_log
        
STR_GIT_SCILOG = '_scilog'
STR_OUTPUT_FILE = 'output.pkl'
STR_INFO_FILE = 'summary.txt'
STR_AUX_DATA_FILE = 'aux_data.pkl'
STR_LOAD_FILE = 'load.sh'
STR_LOG_FILE = 'log.txt'
STR_ERR_FILE = 'err.txt'
STR_RANDOMSTATE_FILE = 'randomstate.pkl'
MSG_MEMPROF = 'Could not find memory_profiler. Install memory_profiler via `pip install memory_profiler`.'
MSG_SERIALIZER = ('Could not find dill. Some items might not be storable. '
                  + 'Storage of numpy arrays will be slow'
                  + 'Install dill via `pip install dill`.')
MSG_ERROR_STORE = 'Could not store{}'
MSG_FINISH_EXPERIMENT = lambda i, runtime: 'Experiment {} finished (Elapsed time: {:.2f}s)'.format(i, runtime)
MSG_FINISHED = 'Scilog entry completed ({})'
MSG_NO_MATCH = 'Could not find matching scilog entry'
MSG_MULTI_MATCH = lambda series:'Multiple matching scilog entries (to iterate through all use need_unique=False):\n{}'.format('\n'.join(series))
MSG_ERROR_LOAD = lambda name: 'Error loading {}. Are all required modules in the Python path?'.format(name)
MSG_ANALYSIS_START = 'Updating analysis'
MSG_ERR_PARALLEL = 'Error during parallel execution. Try running with parallel=False'
MSG_ERROR_GIT_BRANCH = 'Active branch is {}. This branch should only be used for archiving snapshots of other branches, not be archived itself'.format(STR_GIT_SCILOG)
MSG_ERROR_BASH_ANALYSIS = 'Cannot analyze output in bash mode'
MSG_ERROR_GIT_DETACHED = 'Git snapshots do not work in detached HEAD state'
MSG_CMD_ARG = 'Command line arguments to Python call: "{}"'
MSG_WARN_PARALLEL = ('Could not find pathos. This might cause problems with parallel execution.'
    + 'Install pathos via `pip install pathos`.')
GRP_WARN = 'Warning'
GRP_ERR = 'Error'
LEN_ID = 8

#TODO: Add keyword based load, extend keywording 
def record(func, inputs=None, name=None, path='scilog', aux_data=None,
            analyze=None, runtime_profile=False, memory_profile=False,
            git=False, no_date=False, parallel=False, git_path=None,
            external=False, output_directory=None, debug=False, keywords=False):
    '''
    Call :code:`func` and store results along with auxiliary information about
    runtime and memory usage, installed modules, source code, hardware, etc.
    
    If :code:`inputs` is provided, then :code:`func` is called once for each entry
    of :code:`inputs`.
    For example, :code:`func` can be a numerical algorithm and :code:`inputs`
    can be a list of different mesh resolutions (with the goal to assess 
    convergence rates) a list of different subroutines (with the goal to find
    the best subroutine in terms of runtime/memory/...).
    In the following, each call of :code:`func` is called an 'experiment'.

    Scilog creates a directory, specified by :code:`name` and :code:`path`,
    with the following content:
        *summary.txt:
            *name: Name of scilog entry
            *ID: Alphanumeric 8 character string identifying the entry
            *modules: Module versions
            *time: Time of execution
            *experiments: For each experiment
                *string representation of input, 
                *string representation of output,
                *runtime
                *status
                *(optional)peak memory usage
            *(optional)aux_data: Argument :code:`aux_data`
        *log.txt
        *(optional)err.txt
        *(optional)git.txt: stdout of git snapshot creation 
        *source.txt: Source code of the module containing :code:`func`
        *For each experiment a subdirectory 'experiment<i>' with:
            *output.pkl: Output of :code:`func`
            *(optional)input.pkl: Argument passed to :code:`func`
            *(optional) working_directory/: Working directory for call of :code:`func`, 
                unless parameter :code:`output_directory` is specified, in which
                case the working directory is left as is
            *(optional)stderr.txt:
            *(optional)stdout.txt:
            *(optional)runtime_profile.txt: Extensive runtime information for each experiment
            *(optional)memory_profile.txt: Memory usage information for each experiment
        *(optional) analysis/: output of function :code:`analyze`
            *(optional)stderr.txt
            *(optional)stdout.txt
            *(optional)working_directory/: Working directory for call of :code:`analyze`

    To load the contents of summary.txt in Python, use the function :code:`scilog.load`.
    That function additionally replaces the string representations of outputs and inputs in 
    summary.txt by the actual Python-object outputs and inputs. 

    :param func: Function to be called with different experiment configurations
    :type func: function
    :param inputs: List of inputs to :code:`func`. 
        If not specified, then :code:`func` is called once, without arguments
        If passed and integer, then `func` is called as often as specified, without arguments.
    :type inputs: List(-like) or Integer
    :param name: Name of scilog entry. 
        If not specified, :code:`func.__name__` is used
    :type name: String
    :param path: Root directory for storage
    :type path: String
    :param aux_data: Auxiliary data that should be stored along with the results
    :type aux_data: Any
    :param analyze: Function that is called after each experiment 
        Can be used, e.g., for plotting
    :param runtime_profile: Store extensive runtime information
        May slow down execution
    :type runtime_profile: Boolean
    :param memory_profile: Track memory usage
        May slow down execution
    type memory_profile: Boolean
    :param git: Create git snapshot commit
        The resulting commit is tagged with the entry ID and resides outside
        the branch history
        (Should you ever want get rid of the snapshots do `git tag -- list `_scilog*'|xargs -I % git tag -d %`)
    :type git: Boolean
    :param no_date: Do not store outputs in sub-directories grouped by calendar week
    :type date: Boolean
    :param git_path: Specify location of module of func for the creation of a git snapshot
        If not specified, this is determined automatically
    :type git_path: String
    :param external: Specify whether :code:`func` is a Python function or a string
        representing an external call, such as 'echo {}'
        If True, curly braces in the string get replaced by the items of :code:`inputs`
    :type external: Boolean
    :param debug: If True, output is printed instead of being redirected
    :return: Directory of scilog entry
    :rtype: String
    '''
    if external:
        external_string = func
        def func(*experiment):
            subprocess.check_call(external_string.format(*experiment), stdout=sys.stdout, stderr=sys.stderr, shell=True)
    if not name:
        if external:
            regexp = re.compile('\w+')
            name = regexp.match(external_string).group(0)
        else:
            try:
                name = func.__name__
            except AttributeError:
                name = func.__class__.__name__
    directory = _get_directory(name, path, no_date,debug)
    git_path = git_path or os.path.dirname(sys.modules[func.__module__].__file__)
    no_arg_mode = False
    if not inputs:
        no_arg_mode = True
        parallel = False
        inputs = [None]
        n_experiments = 1
    if (Positive & Integer).valid(inputs):
        no_arg_mode = True
        n_experiments = inputs
        inputs = [None] * n_experiments
    else:
        inputs = list(inputs)
        n_experiments = len(inputs)
    ###########################################################################
    log_file = os.path.join(directory, STR_LOG_FILE)
    err_file = os.path.join(directory, STR_ERR_FILE)
    info_file = os.path.join(directory, STR_INFO_FILE)
    load_file = os.path.join(directory, STR_LOAD_FILE)
    aux_data_file = os.path.join(directory,STR_AUX_DATA_FILE)
    source_file_name = os.path.join(directory, 'source.txt')
    git_file = os.path.join(directory, 'git.txt')
    MSG_START = 'This is scilog entry \'{}\' (ID: {})'
    MSG_EXPERIMENTS = ('Will run {} experiment{}'.format(n_experiments, 's' if n_experiments != 1 else '')
                + (' with arguments: \n\t{}'.format('\n\t'.join(map(str, inputs))) if not no_arg_mode else '.'))
    MSG_INFO = 'This entry is stored in {}'.format(directory)
    MSG_TYPE = (('#Experiments were' if n_experiments != 1 else '#Experiment was')
                + ' conducted with a {}'.format(func.__class__.__name__)
               + (' called {}'.format(func.__name__) if hasattr(func, '__name__') else '')
              + ' from the module {} whose source code is given below:\n{}')
    MSG_EXCEPTION_ANALYSIS = 'Exception during online analysis. Check {}'.format(err_file)
    MSG_EXCEPTION_EXPERIMENT = 'Exception during execution of experiment {}'
    MSG_ERROR_GIT = 'Error while creating git snapshot Check {}'.format(err_file)
    MSG_GIT_DONE = 'Successfully created git commit {}'
    STR_GIT_LOG = '#Created git commit {} in branch {} as snapshot of current state of git repository using the following commands:\n{}'.format('{}', STR_GIT_SCILOG, '{}')
    STR_COMMIT = 'Created for scilog entry {} (ID: {}) in {}'
    MSG_SOURCE = 'Could not find source code. Check {}'.format(err_file)
    MSG_GIT_START = 'Creating snapshot of current working tree in branch \'{}\' of repository \'{}\'. Check {}'.format(STR_GIT_SCILOG, '{}', '{}')
    MSG_LOAD_FILE = 'Error while storing source'
    ###########################################################################
    _log = Log(write_filter=True, print_filter=True, file_name=log_file)
    _err = Log(write_filter=True, print_filter=False, file_name=err_file)
    ID = random_string(LEN_ID)
    _log.log(MSG_START.format(name, ID))
    _log.log(MSG_INFO)
    info = dict()
    if keywords is True:
        try:
            import tkinter
            root = tkinter.Tk();root.withdraw()
            keywords = tkinter.simpledialog.askstring('scilog', 'Keywords?')
        except:
            pass
    if keywords in (None, False):  # Could be None if previous failed or because None are desired
        keywords = ''
    else:
        keywords = str(keywords)  # user could have passed actual keywords that are not strings
    info['keywords'] = keywords
    info['name'] = name
    info['ID'] = ID
    info['time'] = datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')
    info['external'] = external
    info['func'] = external_string if external else func.__repr__()
    info['experiments'] = {'runtime':[None] * n_experiments, 'memory':[None] * n_experiments, 'output':[None] * n_experiments, 'status':['queued'] * n_experiments}
    if not no_arg_mode:
        info['experiments'].update({'input':[str(input) for input in inputs]})
    if memory_profile is not False:
        info['experiments'].update({'memory':[None] * n_experiments})
    if not external:
        info['modules'] = _get_module_versions()
        try:
            source = MSG_TYPE.format(sys.modules[func.__module__].__file__, ''.join(inspect.getsourcelines(sys.modules[func.__module__])[0]))
            _store_text(source_file_name, source)
        except Exception:  # TypeError only?
            _err.log(traceback.format_exc())
            _log.log(group=GRP_WARN, message=MSG_SOURCE)
    info['parallel'] = parallel
    info['system'] = _get_system_info()
    if memory_profile is not False:
        if memory_profile == 'detail':
            try:
                import memory_profiler  # @UnusedImport
            except ImportError:
                _log.log(group=GRP_WARN, message=MSG_MEMPROF)
                memory_profile = True
        else:
            memory_profile = True
    try: 
        with open(load_file, 'w') as fp:
            fp.write('#!/bin/sh \n '
                     + ' xterm -e {} -i -c '.format(sys.executable)
                     + '"print(\'>>> import scilog\');'
                     + ' import scilog;'
                     + ' print(\'>>> entry = scilog.load()\');'
                     + ' entry = scilog.load();"')
        st = os.stat(load_file)
        os.chmod(load_file, st.st_mode | stat.S_IEXEC)
    except:
        _err.log(message=traceback.format_exc())
        _log.log(group=GRP_WARN, message=MSG_LOAD_FILE)
    if git:
        try:
            _log.log(message=MSG_GIT_START.format(os.path.basename(os.path.normpath(git_path)), git_file))
            with (capture_output() if not debug else _no_context()) as c:
                snapshot_id, git_log, _ = _git_snapshot(message=STR_COMMIT.format(name, ID, directory), ID=ID, path=git_path)
            _store_text(git_file, STR_GIT_LOG.format(snapshot_id, git_log))
            _log.log(message=MSG_GIT_DONE.format(snapshot_id))
            info['gitcommit'] = snapshot_id
        except GitError as e:
            _store_text(git_file, e.git_log)
            _log.log(group=GRP_ERR, message=MSG_ERROR_GIT)
            _err.log(message=c.stderr + 'Problem with git snapshot\n' + str(e))
            raise
    try:
        import dill
        serializer = dill
    except ImportError:
        serializer = pickle
        _log.log(group=GRP_WARN, message=MSG_SERIALIZER)
    if aux_data:
        with open(aux_data_file,'w') as fp:
            serializer.dump(aux_data,fp)
    info_serializer = json
    def store_info():
        with open(info_file, 'w') as fp:
            try:
                info_serializer.dump(info, fp, indent=1, separators=(',\n', ': '))
            except (TypeError, pickle.PicklingError) as e:
                _err.log(message=traceback.format_exc())
                _log.log(group=GRP_WARN, message=MSG_ERROR_STORE.format(STR_INFO_FILE))
    def _update_info(i, runtime, status, memory, output_str):
        info['experiments']['runtime'][i] = runtime
        if memory_profile is not False:
            info['experiments']['memory'][i] = memory
        info['experiments']['status'][i] = status
        info['experiments']['output'][i] = output_str
        store_info()
    store_info()
    old_wd = os.getcwd()
    locker = Locker()
    args = ((i, input, directory, func, memory_profile,
             runtime_profile, log_file, err_file,
             'pickle' if serializer == pickle else 'dill', no_arg_mode, locker,
             external, debug, output_directory)
            for i, input in enumerate(inputs))
    _log.log(message=MSG_EXPERIMENTS)
    if parallel:
        try:
            from pathos.multiprocessing import ProcessingPool as Pool
            pool = Pool(nodes=n_experiments)
        except ImportError:
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_WARN, message=MSG_WARN_PARALLEL)
            from multiprocessing import Pool
            pool = Pool(processes=n_experiments)
        try:
            outputs = pool.map(_run_single_experiment, args)
        except _pickle.PicklingError:  # @UndefinedVariable
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_ERR, message=MSG_ERR_PARALLEL)
            raise
        for output in outputs:
            _update_info(*output)
        pool.close()
        pool.join()
    else:
        for arg in args:
            info['experiments']['status'][arg[0]] = 'running'
            store_info()
            try:
                output = _run_single_experiment(arg)
            except Exception:
                _err.log(message = traceback.format_exc())
                _log.log(group = GRP_ERR,message = MSG_EXCEPTION_EXPERIMENT.format(arg[0]))
            _update_info(*output)
            if analyze:
                _log.log(message=MSG_ANALYSIS_START)
                try:
                    entry = load(path=directory, need_unique=True, no_results=False)
                    globals()['analyze'](func=analyze, entry = entry, _log=_log, _err=_err,debug = debug)
                except:
                    _err.log(message=traceback.format_exc())
                    _log.log(group=GRP_ERR, message=MSG_EXCEPTION_ANALYSIS)
    os.chdir(old_wd)
    _log.log(MSG_FINISHED.format('all experiments finished successfully' 
                                if all(s == 'finished' for s in info['status']) 
                                else 'some experiments failed'))
    return directory

class Locker(object):
    def __init__(self):
        mgr = Manager()
        self.lock = mgr.Lock()
        self.err_lock = mgr.Lock()
        self.analyze_lock = mgr.Lock()

def _get_module_versions():
    names = sys.modules.keys()
    names = [name for name in names if not '.' in name]
    module_versions = {}
    for name in names:
        if hasattr(sys.modules[name], '__version__'):
                module_versions[name] = sys.modules[name].__version__ + '(__version__)'
    pip_list = operations.freeze.get_installed_distributions()
    for entry in pip_list:
        (key, val) = entry.project_name, entry.version
        if key in module_versions:
            module_versions[key] += '; ' + val + '(pip)'
        else:
            module_versions[key] = val + '(pip)'
    return module_versions

def _get_system_info():
    system_info = '; '.join([platform.platform(), platform.python_implementation() + ' ' + platform.python_version()])
    try:
        import psutil
        system_info += '; ' + str(psutil.cpu_count(logical=False)) + ' cores'
        system_info += '; ' + str(float(psutil.virtual_memory().total) / 2 ** 30) + ' GiB'
    except:
        pass
    return system_info

class _no_context():
    def __enter__(self, *args):
        pass
    def __exit__(self, *args):
        pass
    
def _delete_empty_files(fpaths):  
    for fpath in fpaths:
        try:
            if os.path.isfile(fpath) and os.path.getsize(fpath) == 0:
                os.remove(fpath)
        except Exception:
            pass
        
def _delete_empty_directories(dirs):
    for dir in dirs:
        try:
            os.rmdir(dir)
        except OSError:
            pass#raised if not empty
    
def _run_single_experiment(arg):
    (i, input, directory, func, memory_profile,
     runtime_profile, log_file_global, err_file_global, serializer, no_arg_mode,
     locker, external, debug, output_directory) = arg
    ###########################################################################
    experiment_directory = os.path.join(directory, 'experiment{}'.format(i))
    stderr_file = os.path.join(experiment_directory, 'stderr.txt')
    stdout_file = os.path.join(experiment_directory, 'stdout.txt')
    input_file = os.path.join(experiment_directory, 'input.pkl')
    output_file = os.path.join(experiment_directory, STR_OUTPUT_FILE)
    randomstate_file = os.path.join(experiment_directory, STR_RANDOMSTATE_FILE)
    runtime_profile_file = os.path.join(experiment_directory, 'runtime_profile.txt')
    memory_profile_file = os.path.join(experiment_directory, 'memory_profile.txt')
    experiment_working_directory = os.path.join(experiment_directory, 'working_directory')
    MSG_FAILED_EXPERIMENT = 'Experiment {} failed. Check {}'.format(i, stderr_file)
    MSG_EXCEPTION_EXPERIMENT = 'Exception during execution of experiment {}'.format(i)
    MSG_START_EXPERIMENT = ('Runnning experiment {}'.format(i) + 
                                      (' with argument:\n\t{}'.format(str(input)) if not no_arg_mode else ''))
    MSG_FAILED_INPUT_STORAGE = 'Could not store input object'
    ###########################################################################
    _log = Log(write_filter=True, print_filter=True, file_name=log_file_global, lock=locker.lock)
    _err = Log(write_filter=True, print_filter=False, file_name=err_file_global, lock=locker.err_lock)
    if serializer == 'pickle':
        serializer = pickle
    else:
        import dill
        serializer = dill
    _log.log(MSG_START_EXPERIMENT)
    runtime = None
    output = None
    memory = None
    status = 'failed'
    randomstate = None
    if not external:
        try:
            import numpy
            randomstate = numpy.random.get_state()
        except ImportError:
            pass  # Random state only needs to be saved if numpy is used
    if hasattr(func, '__name__'):
        temp_func = func
    else:
        temp_func = func.__call__
    if output_directory is None:
        os.makedirs(experiment_working_directory)
        os.chdir(experiment_working_directory)
    try:
        if not no_arg_mode:
            try:
                if not no_arg_mode:
                    with open(input_file, 'wb') as fp:
                        serializer.dump(input, fp)
            except:
                _err.log(message=traceback.format_exc())
                _log.log(group=GRP_WARN, message=MSG_FAILED_INPUT_STORAGE)
        if memory_profile is not False:
            m = StringIO()
            if memory_profile == 'detail':
                import memory_profiler
                temp_func = memory_profiler.profile(func=temp_func, stream=m, precision=4)
            else:
                temp_func = print_peak_memory(func=temp_func, stream=m)
        if runtime_profile:
            temp_func = add_runtime(temp_func)
        stderr_append = ''
        with open(stderr_file, 'a', 1) as err:
            with open(stdout_file, 'a', 1) as out:
                with contextlib.redirect_stdout(out) if not debug else _no_context():
                    with contextlib.redirect_stderr(err) if not debug else _no_context():
                        tic = timeit.default_timer()
                        try:
                            if no_arg_mode:
                                output = temp_func()
                            else:
                                output = temp_func(input)
                        except Exception:
                            status = 'failed'
                            stderr_append = traceback.format_exc()
                        else:
                            status = 'finished'
        _delete_empty_files([stderr_file,stdout_file])
        runtime = timeit.default_timer() - tic
        if status == 'failed':
            _store_text(stderr_file,stderr_append)
            _log.log(group=GRP_ERR, message=MSG_FAILED_EXPERIMENT)
        if runtime_profile:
            profile, output = output
            s = StringIO()
            ps = pstats.Stats(profile, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats()
            _store_text(runtime_profile_file, s.getvalue())
            s.close()
        if memory_profile:
            _store_text(memory_profile_file, m.getvalue() + ('' if memory_profile == 'detail' else 'MB (Use `memory_profile==\'detail\'` for a more detailed breakdown)'))
            memory = _max_mem(m.getvalue(), type=memory_profile)
    except Exception:
        _err.log(message=traceback.format_exc())
        _log.log(group=GRP_ERR, message=MSG_EXCEPTION_EXPERIMENT)
    if status == 'finished':
        _log.log(MSG_FINISH_EXPERIMENT(i, runtime))
    if output_directory is None:
        os.chdir(directory)
    else:
        shutil.copytree(output_directory, experiment_working_directory, symlinks=False, ignore_dangling_symlinks=True)
    _delete_empty_directories([experiment_working_directory])
    output_str = str(output)
    with open(output_file, 'wb') as fp:
        try:
            serializer.dump(output, fp)
        except (TypeError, pickle.PicklingError):
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_ERR, message=MSG_ERROR_STORE.format(STR_OUTPUT_FILE))
    del output
    if randomstate is not None:
        with open(randomstate_file, 'wb') as fp:
            try:
                serializer.dump(randomstate, fp)
            except (TypeError, pickle.PicklingError):
                _err.log(message=traceback.format_exc())
                _log.log(group=GRP_WARN, message=MSG_ERROR_STORE.format(STR_RANDOMSTATE_FILE))
    gc.collect()
    return (i, runtime, status, memory, output_str)

def _store_text(file_name, data):
    if data:
        with open(file_name, 'a') as fp:
            fp.write(data)

class Component():
    def __init__(self, k):
        '''
        Access single component of container. 
        
        Example 1:
            pixels = [(120,230),(12,42),(42,12)]
            xc = Component(0)
            xs = [xc(pixels)] 
        Example 2:
            people = [{'Name':'Tom','Age':52},{'Name':'Bill','Age':25}]
            namec = Component('Name')
            names = [namec(person) for person in people]
        
        The is an auxiliary class that may be used as argument for parameter `qois` of ConvergencePlotter,
        
        :param k: component, e.g., integer if container is a list
        '''
        self.k = k
    def __call__(self, x):
        return x[self.k]

class ConvergencePlotter():
    def __init__(self, *qois, cumulative=False, work=None, extrapolate=0):
        '''
        Create convergence plots (of given quantities of interest (qois))
        
        This is an auxiliary class that may be used as argument for parameter 
        `analyze` of scilog.record or parameter `func` of scilog.analyze
        
        :param qois: List of functions that can be applied to the outputs of an experiment
        :param cumulative: Specify whether work is cumulative across the experiments
        :param work: If a measure other than runtime should be used, this must be a function taking integers and returning reals
        :param extrapolate: Degree of Richardson extrapolation 
            extrapolate = -1 uses exponential extrapolation, whereas 
            positive values merely improve algebraic convergence orders
        '''
        self.qois = qois
        self.cumulative = cumulative
        self.work = work
        self.extrapolate = extrapolate
    def __call__(self, entry):
        results = entry.experiments['output']
        experiments = entry.experiments
        ind_finished = [j for (j, status) in enumerate(experiments['status']) if status == 'finished']
        if len(ind_finished) > 2+(self.extrapolate if self.extrapolate>= 0 else 0):
            if self.work is None:
                times = [experiments['runtime'][i] for i in ind_finished]
            else:
                times = [self.work(i) for i in ind_finished]
            results = [results[i] for i in ind_finished]
            if self.cumulative:
                times = np.cumsum(times)
            if not self.qois:
                if hasattr(results[0], 'len') and not isinstance(results[0], np.ndarray):
                    self.qois = [Component(k) for k in range(len(results[0]))]
                else:
                    self.qois = [lambda x:x]
            for (k, qoi) in enumerate(self.qois):
                try:
                    pyplot.figure(k).clf()
                    qoi_values = np.array([qoi(result) for result in results])
                    qoi_times = np.array(times)
                    if self.extrapolate:
                        qoi_values,qoi_times = np_tools.extrapolate(qoi_values, qoi_times, self.extrapolate)
                    plot_convergence(qoi_times, qoi_values)
                    plots.save('convergence')
                except:
                    traceback.print_exc()
                    

def analyze(func, entry, _log=None, _err = None, debug = False):
    '''
    Add analysis to scilog entry or entries
    
    :param func: Function that performs analysis
    :param entry: scilog entry or entries (as returned by scilog.load)
    :param _log: Log object to be used instead of writing to standard stdout
    :param _err: Log object to be used instead of writing to standard stderr
    :param debug: If True, output is printed instead of being redirected into files
    '''
    if not _log:
        _log = Log(print_filter = True)
    if not _err:
        _err = Log(print_filter = True)
    try:
        import dill
        serializer = dill
    except ImportError:
        serializer = pickle
        _log.log(group=GRP_WARN, message=MSG_SERIALIZER)
    MSG_FAILED_ANALYSIS = lambda stderr_file: 'Analysis could not be completed. Check {}'.format(stderr_file)
    MSG_STORE_ANALYSIS = lambda name: 'Could not store output of analysis'
    cwd = os.getcwd()
    if not inspect.isgenerator(entry):
        entries = [entry]
    else:
        entries = entry
    for entry in entries:
        analysis_directory_tmp = os.path.join(tempfile.mkdtemp(dir=entry.path), 'analysis')
        working_directory = os.path.join(analysis_directory_tmp, 'working_directory')
        stderr_file = os.path.join(analysis_directory_tmp, 'stderr.txt')
        stdout_file = os.path.join(analysis_directory_tmp, 'stdout.txt')
        output_file = os.path.join(analysis_directory_tmp, 'output.pkl')
        os.mkdir(analysis_directory_tmp)
        os.mkdir(working_directory)
        os.chdir(working_directory)
        output = None
        stderr_append = ''
        with open(stderr_file, 'a', 1) as err:
            with open(stdout_file, 'a', 1) as out:
                with contextlib.redirect_stdout(out) if not debug else _no_context():
                    with contextlib.redirect_stderr(err) if not debug else _no_context():
                        try:
                            output = func(entry)
                        except Exception:
                            stderr_append = traceback.format_exc()
        _delete_empty_files([stderr_file,stdout_file])
        _delete_empty_directories([working_directory])
        if stderr_append:
            _store_text(stderr_file, stderr_append)
            _log.log(group=GRP_ERR, message=MSG_FAILED_ANALYSIS(stderr_file))
        if output is not None:
            with open(output_file, 'wb') as fp:
                try:
                    serializer.dump(output, fp)
                except (TypeError, pickle.PicklingError):
                    _err.log(message = traceback.format_exc())
                    _log.log(group=GRP_WARN, message=MSG_STORE_ANALYSIS)
        os.chdir(cwd)
        analysis_directory = os.path.join(entry.path, 'analysis')
        shutil.rmtree(analysis_directory, ignore_errors=True)
        shutil.move(analysis_directory_tmp, entry.path)
        shutil.rmtree(os.path.split(analysis_directory_tmp)[0], ignore_errors=True)

def load(search_pattern='*', path='', ID=None, no_results=False, need_unique=True):
    '''
    Load scilog entry/entries.
   
    :param search_pattern: Shell-style glob/search pattern using wildcards
        If there are multiple entries of the same name (those are stored as
        <name>/v0 <name>/v1 ... in the filesystem) and they should all be returned, 
        use `search_pattern=<name>/v*` and `need_unique=False`
    :type search_pattern: String, e.g. search_pattern='foo*' matches `foobar`
    :param path: Path of exact location is known (possibly only partially), relative or absolute
    :type path: String, e.g. '/home/work/2017/6/<name>' or 'work/2017/6'
    :param no_results: Only load information about scilog entry, not results
    :type no_results: Boolean
    :param need_unique: Require unique identification of scilog entry.
    :type need_unique: Boolean
    :return: Scilog entry
    :rtype: If need_unique=True, a single Namespace obejct
        If need_unique=False, a generator of such objects
    '''
    deserializer = pickle
    try:
        import dill
        deserializer = dill
    except ImportError:
        warnings.warn(MSG_SERIALIZER)
    series = []
    if os.sep in search_pattern and path == '':
        temp_path, temp_search_pattern = search_pattern.rsplit(os.sep, 1)
        if os.path.isabs(temp_path):
            path, search_pattern = temp_path, temp_search_pattern
    series.extend(files.find_directories(search_pattern, path=path))
    series.extend(files.find_directories('*/' + search_pattern, path=path))
    series = [serie for serie in series if _is_experiment_directory(serie)]
    def get_output(serie, no_results):
        file_name = os.path.join(serie, STR_INFO_FILE)
        with open(file_name, 'r') as fp:
            info = json.load(fp)
        info['path'] = serie
        info = Namespace(**info)
        if not no_results:
            for (j, status) in enumerate(info.experiments['status']):
                if status == 'finished':
                    try:
                        results_file_name = os.path.join(serie, 'experiment{}'.format(j), STR_OUTPUT_FILE)
                        with open(results_file_name, 'rb') as fp:
                            result = deserializer.load(fp)
                        info.experiments['output'][j] = result
                    except:
                        traceback.print_exc()
        return info
    if ID:
        if len(ID) < LEN_ID:
            ID = ID + '.{' + str(LEN_ID - len(ID)) + '}'
        regexp = re.compile(ID)
        series = [serie for serie in series if regexp.match(get_output(serie, True).ID)]
    series = unique(series)
    if not need_unique:
        return (get_output(serie, no_results=no_results) for serie in series)
    else:
        if len(series) == 0:
            raise ValueError(MSG_NO_MATCH)
        if len(series) > 1:
            raise ValueError(MSG_MULTI_MATCH(series))
        return get_output(series[0], no_results=no_results)

def _is_experiment_directory(directory):
    return os.path.isfile(os.path.join(directory, STR_INFO_FILE))

def _max_mem(m, type):  # @ReservedAssignment
    if m == '':
        return -1
    if type == 'detail':  # Output of memory_profiler package
        find = re.compile('.*?(\d{1,}\.\d{4}) MiB.*')
        matches = [find.match(line) for line in m.splitlines()]
        values = [float(match.groups()[0]) for match in matches if match is not None]
        return max(values) - min(values)
    else:  # Output of print_peak_memory
        return float(m)

def _get_directory(name, path, no_date, debug):
    if no_date:
        directory = os.path.join(path, name)
    else:
        date = datetime.date.today()
        directory = os.path.join(path, 'w' + date.strftime('%W') + 'y' + str(date.year)[-2:], name)
    if debug:
        directory = os.path.join(directory,'debug')
    directory = os.path.abspath(directory)
    version = 0
    if os.path.exists(directory) and os.listdir(directory):
        candidates = [os.path.split(dir)[1] for dir in os.listdir(directory)  # @ReservedAssignment
                    if os.path.isdir(os.path.join(directory, dir))
                    and re.search('^v([0-9]|[1-9][0-9]+)$', dir)]
        if candidates:
            version = max([int(dir[dir.rindex('v') + 1:]) for dir in candidates]) + 1  # @ReservedAssignment
    directory = os.path.join(directory, 'v' + str(version))
    try:
        os.makedirs(directory)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    return directory

def _git_command(string, add_input=True):
    string = 'git ' + string
    output = '$ ' + string + '\n' if add_input else ''
    args = shlex.split(string)
    output += subprocess.check_output(args, stderr=subprocess.STDOUT).decode('UTF8')
    return output

def _git_id():
    return _git_command('log --format="%H" -n 1', add_input=False).rstrip()

def _git_snapshot(path, message, ID):
    initial_directory = os.getcwd()
    os.chdir(path)
    git_directory = _git_command('rev-parse --show-toplevel', add_input=False).rstrip()
    os.chdir(git_directory)
    active_branch = _git_command('rev-parse --abbrev-ref HEAD', add_input=False)
    try:
        out = ''
        out += _git_command('add --all')
        out += _git_command('rm -r --cached .')
        out += _git_command('add --all')
        out += _git_command('commit --allow-empty -m "Snapshot of working directory of branch {0} \n {1}"'.format(active_branch, message))
        out += _git_command('tag _scilogID{}'.format(ID))
        snap_id = _git_id()
        out += _git_command('reset HEAD~1')
    except subprocess.CalledProcessError as e:
        raise GitError(traceback.format_exc(), out + '\n' + str(e.output))
    except:
        raise GitError(traceback.format_exc(), out)
    os.chdir(initial_directory)
    return snap_id, out, git_directory

def main():
    import textwrap as _textwrap
    class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def _split_lines(self, text, width):
            text = self._whitespace_matcher.sub(' ', text).strip()
            return _textwrap.wrap(text, width)
    parser = argparse.ArgumentParser(formatter_class=LineWrapRawTextHelpFormatter,
        description=
        '''
        Call FUNC and store results along with auxiliary information about
        runtime and memory usage, installed modules, source code, hardware, etc.
        
        If INPUTS is provided, then FUNC is called once for each entry
        of INPUTS
        For example, FUNC can be a numerical algorithm and INPUTS
        can be a list of different mesh resolutions (with the goal to assess 
        convergence rates) a list of different subroutines (with the goal to find
        the best subroutine in terms of runtime/memory/...).
        In the following, each call of FUNC is called an 'experiment'.

        Scilog creates a directory (using NAME and the current date)
        with the following content:
            *summary.txt:
                *name: Name of scilog entry
                *ID: Alphanumeric 8 character string identifying the entry
                *modules: Module versions
                *time: Time of execution
                *experiments: For each experiment
                    *string representation of input, 
                    *string representation of output,
                    *runtime
                    *status
                    *(optional)peak memory usage
            *log.txt
            *(optional)err.txt
            *(optional)git.txt: stdout of git snapshot creation 
            *source.txt: Source code of the module containing FUNC
            *For each experiment a subdirectory 'experiment<i>' with:
                *output.pkl: Output of FUNC
                *(optional)input.pkl: Argument passed to FUNC
                *(optional) working_directory/: Working directory for call of FUNC
                    unless OUTPUTDIR is specified, in which
                    case the working directory is left as is
                *(optional)stderr.txt:
                *(optional)stdout.txt:
                *(optional)runtime_profile.txt: Extensive runtime information for each experiment
                *(optional)memory_profile.txt: Memory usage information for each experiment
            *(optional) analysis/: output of ANALYZE
                *(optional)stderr.txt
                *(optional)stdout.txt
                *(optional)working_directory/: Working directory for call of ANALYZE

        To load the contents of summary.txt in Python, use the function :code:`scilog.load`.
        That function additionally replaces the string representations of outputs and inputs in 
        summary.txt by the actual Python-object outputs and inputs. 

        To display information about an existing scilog entry in the command line,
        the switch --show may be used with this script.)
        ''')
    parser.add_argument("func", action='store',
        help=
        '''
        The function that is executed

        The standard way is to provide the full path of a Python function
            e.g.: `foo.func`.

        There are three alternatives:
            1) Provide the full name of a module that contains a class of the same name (up to capitalization).
                e.g.: `foo.bar`

            2) Provide the full name of a class.
                e.g.: `foo.bar.Bar2`

            3) Provide a bash command string, 
                e.g.: `echo {}s`

        In cases 1) and 2), the specified class is instantiated once
         and all experiments are performed by calling this instance.
        In case 3), curly braces in the command string are replaced by the inputs 
        ''')
    parser.add_argument('-i', '--inputs', action='store', default='None',
        help=
        '''
        List of inputs.

        e.g.: [2**l for l in range(10)]

        If not specified, FUNC is called once without arguments.

        If FUNC is a bash command string, inputs must be a list strings
        ''')
    parser.add_argument('-b', '--base', action='store', default='{}',
        help=
        '''
        Base configuration (in form of a dictionary) for all experiments.

        If argument FUNC is a function, this dictionary is passed
        along the entries of INPUTS in form of keyword arguments to FUNC.

        If argument FUNC specifies a class, the class is instantiated using
        this dictionary in form of keyword arguments.

        If argument FUNC is a bash command string, the entries of this dictionary
        are used to fill named braces after the intial empty braces, 
            e.g.: if FUNC is "my_func {} -d {dir}" and BASE is "{'dir':'/my/path'}" 
             and INPUTS is 'range(2)', then the following experiments will be run:
                 1) my_func 0 -d /my/path
                 2) my_func 1 -d /my/path
        Note that '/my/path' can be replaced by any valid Python expression that returns a string.
        ''')
    parser.add_argument('-n', '--name', action='store', default=None,
        help=
        '''
        Name of the scilog entry.

        If not provided, the name is derived from FUNC
        ''')
    parser.add_argument('-a', '--analyze', action='store',
        nargs='?', const='analyze', default=None,
        help=
        '''
        Function that is used to perform analysis after each experiment.
        
        By default, ANALYZE is the name of a function in the same module as FUNC.

        Alternatively, ANALYZE can be
            1) a full name of a function in some different module,
                e.g.: foo2.analyze

            2) a name of a method of the class specified by FUNC
        ''')
    parser.add_argument('-d', '--directory', action='store', default='scilog',
        help=
        '''
        Specify where scilog entry should be stored
        ''')
    parser.add_argument('-p', '--parallel', action='store_true',
        help=
        '''
        Perform experiments in parallel.
        ''')
    parser.add_argument('-m', '--memory_profile', action='store_true',
        help=
        '''
        Store memory information for each experiment
        ''')
    parser.add_argument('-r', '--runtime_profile', action='store_true',
        help=
        '''
        Store extensive runtime information for each experiment.

        The total time of each experiment is always stored.
        ''')
    parser.add_argument('-g', '--git', action='store_true',
        help=
        '''
        Create git snapshot commit
        
        The resulting commit is tagged with the entry ID and resides outside
        the branch history
        
        Add 'scilog' to your .gitignore to avoid storing the scilog entries in each snapshot 
        '''.format(STR_GIT_SCILOG))
    parser.add_argument('--no_date', action='store_true',
        help=
        '''
        Do not store scilog entry in subdirectories based on current date.
        ''')
    parser.add_argument('--external', action='store_true',
        help=
        '''
        Specify that FUNC describes an external call.
        
        This is only needed, when FUNC could be confused for a Python module, 
        e.g., when FUNC=`foo.bar`
        ''')
    parser.add_argument('-s', '--show', action='store_true',
        help=
        '''
        Print information of previous entry instead of creating new entry.

        In this case, FUNC must the path of an existing scilog entry.
        (Shell-style wildcards, e.g. 'foo*', are recognized)
        Furthermore, the --git flag can be used to show an interactive view of
        the differences of the working directory and the repository at the time
        of the creation of the scilog entry
        ''')
    parser.add_argument('-o', '--output', action='store', nargs='?', const='.',
        default=None,
        help=
        '''
        Specify directory where FUNC stores its output

        If no argument is specified, FUNC will be run in a clean working directory
        and it is assumed that its outputs are stored in that working directory
        ''')
    args = parser.parse_args()
    if args.show:
        entries = load(search_pattern=args.func, no_results=True, need_unique=False)
        entries = list(entries)
        if len(entries) != 1:
            print('Found {} entries'.format(len(entries)))
        for entry in entries:
            print('=' * 80)
            print('Entry \'{}\' at {}:'.format(entry[0]['name'], entry[1]))
            print('=' * 80)
            print(json.dumps(entry[0], sort_keys=True, indent=4, default=str)[1:-1])
            if args.git:
                print('\n The current working directory differs from the git repository at the time of the scilog entry as follows:')
                subprocess.call(['gitdiffuntracked', entry[0]['gitcommit']])
    else:
        args.experiments = eval(args.experiments)
        init_dict = eval(args.base)
        module_name = args.func
        regexp = re.compile('(\w+\.)+(\w+)')
        args.external = args.external or not regexp.match(module_name)
        if not args.external:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                real_module_name = '.'.join(module_name.split('.')[:-1])
                module = importlib.import_module(real_module_name)
            try:  # Assume class is last part of given module argument
                class_or_function_name = module_name.split('.')[-1]
                cl_or_fn = getattr(module, class_or_function_name)
            except AttributeError:  # Or maybe last part but capitalized?
                class_or_function_name = class_or_function_name.title()
                cl_or_fn = getattr(module, class_or_function_name)
            if args.name == '_':
                args.name = class_or_function_name
            if inspect.isclass(cl_or_fn):
                fn = cl_or_fn(**init_dict)
            else:
                if init_dict:
                    def fn(*experiment):  # Need to pass experiment as list, to be able to handle zero-argument calls
                        return cl_or_fn(*experiment, **init_dict)
                else:
                    fn = cl_or_fn
            if args.analyze:
                try:
                    split_analyze = args.analyze.split('.')
                    try:
                        if len(split_analyze) > 1:  # Analyze function in different module
                            analyze_module = importlib.import_module('.'.join(split_analyze[:-1]))
                        else:
                            analyze_module = module
                        analyze_fn = getattr(analyze_module, split_analyze[-1])
                    except AttributeError:  # is analyze maybe a function of class instance?
                        analyze_fn = getattr(fn, args.analyze)
                except:
                    analyze_fn = None
                    traceback.print_exc()
                    warnings.warn(MSG_ERROR_LOAD('function {}'.format(args.analyze)))
            else:
                analyze_fn = None
            module_path = os.path.dirname(module.__file__)
        else:  # Assume the module describes an external call
            module_name.format('{}', **init_dict)
            fn = module_name
            if args.analyze:
                raise ValueError(MSG_ERROR_BASH_ANALYSIS)
            analyze_fn = None
            module_path = os.getcwd()
        record(
            func=fn, path=args.directory,
            inputs=args.inputs,
            name=args.name,
            external=args.external,
            analyze=analyze_fn,
            runtime_profile=args.runtime_profile,
            memory_profile=args.memory_profile,
            git=args.git,
            no_date=args.no_date,
            parallel=args.parallel,
            git_path=module_path,
            output_directory=args.output
        )
if __name__ == '__main__':
    main()
