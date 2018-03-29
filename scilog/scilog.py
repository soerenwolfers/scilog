import timeit
import pickle
import time
import os
import errno
import datetime
import shutil
import warnings
import traceback
from swutil import files, plots, validation
from swutil.collections import unique
from swutil.decorators import add_runtime, measure_memory_usage
from swutil.logs import Log
import pstats
from io import StringIO
import itertools
import sys
import gc
from IPython.utils.capture import capture_output
import inspect
import argparse
import importlib
import random
import string
import numpy as np
import re
from multiprocessing import Manager
import _pickle
import subprocess
import shlex
import platform
from pip import operations
import json
import contextlib
import pathlib
from functools import partial
from swutil.aux import random_string
from matplotlib import pyplot
from swutil.plots import plot_convergence
from swutil.validation import Positive, Integer
class GitError(Exception):
    def __init__(self, message, git_log):
        super(GitError, self).__init__(message)
        self.git_log = git_log
MSG_MEMPROF = 'Could not find memory_profiler. Install memory_profiler via `pip install memory_profiler`.'
STR_GIT_SCILOG = '_scilog'
MSG_SERIALIZER = ('Could not find dill. Some items might not be storable. '
                  + 'Storage of numpy arrays will be slow'
                  + 'Install dill via `pip install dill`.')
MSG_STORE_RESULT = 'Could not serialize results'
MSG_STORE_RANDOMSTATE = 'Could not store random state'
MSG_STORE_INFO = lambda keys: 'Could not store keys {}.'.format(keys)
MSG_FINISH_EXPERIMENT = lambda i, runtime: 'Experiment {} finished (Elapsed time: {:.2f}s)'.format(i, runtime)
MSG_RUNTIME_SIMPLE = lambda runtime: str(runtime) + 's (Use \'runtime_profile=True\' for a more detailed breakdown)'
MSG_FINISHED = 'Scilog entry completed ({})'
MSG_NO_MATCH = 'Could not find matching scilog entry'
MSG_MULTI_MATCH = lambda series:'Multiple matching scilog entries (to iterate through all use need_unique=False):\n{}'.format('\n'.join(series))
# MSG_UNUSED = 'Passed configuration dictionary is unused when running experiment series with function'
MSG_ERROR_LOAD = lambda name: 'Error loading {}. Are all required modules in the Pyhton path?'.format(name)
MSG_ANALYSIS_START = 'Updating analysis'
MSG_ERR_PARALLEL = 'Error during parallel execution. Try running with parallel=False'
MSG_ERROR_GIT_BRANCH = 'Active branch is {}. This branch should only be used for archiving snapshots of other branches, not be archived itself'.format(STR_GIT_SCILOG)
MSG_ERROR_BASH_ANALYSIS = 'Cannot analyze output in bash mode'
MSG_ERROR_GIT_DETACHED = 'Git snapshots do not work in detached HEAD state'
MSG_WITHIN_GIT = 'Saving experiments within the git repository can cause problems with the creation of git snapshots'
MSG_CMD_ARG = 'Command line arguments to python call: "{}"'
MSG_WARN_PARALLEL = ('Could not find pathos. This might cause problems with parallel execution.'
    + 'Install pathos via `pip install pathos`.')
GRP_WARN = 'Warning'
GRP_ERR = 'Error'
LEN_ID = 8
LEN_TMP = 8
def conduct(func, experiments=None, name=None, path='scilog', supp_data=None,
            analyze=None, runtime_profile=False, memory_profile=False,
            git=False, no_date=False, no_dill=False, parallel=False, module_path=None,
            external=False, working_directory=None,debug = False):
    '''
    Call :code:`func` once for each item of :code:`experiments` and store
    results along with auxiliary information such as runtime and memory usage.
    Each item of experiments is passed as a whole to :code:`func`, e.g.:
            def func(experiment):
                return experiment['a']*experiment['x']**experiment['exponent']
            base={'exponent':2,'a':5}
            experiments=[dict('x'=x,**base) for x in range(10)]
            conduct(func,experiments)
    In practice, :code:`func` can be a numerical algorithm and :code:`experiments`
    can be a list of different mesh resolutions, a list of different
    subroutines, etc.

    This function stores the following files and directories in a directory
    specified by :code:`name` and :code:`path`:
        *info.pkl:
            *name: Name of labook entry (str)
            *ID: Alphanumeric 8 character string identifying the scilog entry (str)
            *modules: Module versions (list of str)
            *time: Time of execution (datetime.datetime)
            *experiments: Parameter :code:`experiments`
            *runtime: Runtime of each experiment (list of floats)
            *status: Status of each experiment (list of ('queued'/'finished'/'failed'))
            *(optional)supp_data: Parameter :code:`supp_data`
        *stdout.txt
        *(optional)stderr.txt
        *(optional)git.txt: stdout of git snapshot creation (in case this functionality ever crashes, you'll find your working tree in the git stash, and this information should help recovering it from there)
        *results.pkl: List of results of experiments
        *source.txt: Source code of the module containing :code:`func`
        *For each experiment a subdirectory "experiment<i>" with:
            *user_files/ (Working directory for call of :code:`func`)
            *(optional)input.txt: Argument passed to :code:`func`
            *stderr.txt:
            *stdout.txt:
            *(optional)runtime_profile.txt: Extensive runtime information for each experiment (list of strings)
            *(optional)memory_profile.txt: Memory usage information for each experiment (list of strings)
        *(optional) analysis/: output of function :analysis:
            *stderr.txt
            *stdout.txt
            *user_files/ (Working directory for call of :code:`analyze`


    Both info.pkl and results.pkl are created with pickle; for technical
    reasons they contain multiple concatenated pickle streams. To load these files,
    and automatically join the contents of info.pkl into a single dictionary and
    the contents of results.pkl into a single list, use function :code:`load`

    :param func: Function to be called with different experiment configurations
    :type func: function
    :param experiments: List experiment configurations. 
        If not passed, then `func` is called once, without arguments
        If passed and integer, then `func` is called as often as specified, without arguments.
    :type experiments: List(-like) or Integer
    :param name: Name of scilog entry. Using func.__name__ if not provided
    :type name: String
    :param path: Root directory for storage, absolute or relative
    :type path: String
    :param supp_data: Additional information that should be stored along with
        the results.
    :type supp_data: Any.
    :param analyze: Function that is called after each experiment with results so far, can be used for plotting
    :param runtime_profile: Provide extensive runtime information. This can slow
    down the execution.
    :type runtime_profile: Boolean.
    :param memory_profile: Track memory usage. This can slow down the execution.
    type memory_profile: Boolean
    :param git: Create git snapshot in branch _scilog
    :type git: Boolean.
    :param no_date: Do not store outputs in sub-directories grouped by calendar week.
    :type date: Boolean.
    :param no_dill: Do not use dill module. Explanation: Using pickle to store
        numpy arrays in Python2.x is slow. Furthermore, pickle cannot serialize
        Lambda functions, or not-module level functions. As an alternative, this
        function uses dill (if available) unless this parameter is set to True.
    :type no_dill: Boolean.
    :param module_path: Specify location of module of func. This is used for
    the creation of a git snapshot. If not specified, this is determined automatically
    :type module_path: String
    :param external: Specify whether :code:`func` is a python function or a string
        representing an external call, such as 'echo {}'. In this case, the curly
        braces in the string get replaced by the entries of :code:`experiments`
    :type external: Boolean
    :param debug: If true, output is printed, not redirected
    '''
    if working_directory:
        link = [os.path.join(os.path.abspath(working_directory), f)
              for f in os.listdir(working_directory)]
    else:
        link = []
    if external:
        external_string = func
        def func(wd, *experiment):
            subprocess.check_call(external_string.format(*experiment), cwd=wd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
    if not name:
        if external:
            regexp = re.compile('\w+')
            name = regexp.match(external_string).group(0)
        else:
            try:
                name = func.__name__
            except AttributeError:
                name = func.__class__.__name__
    directory = _get_directory(name, path, no_date)
    module_path = module_path or os.path.dirname(sys.modules[func.__module__].__file__)
    no_arg_mode = False
    if not experiments:
        no_arg_mode = True
        parallel = False
        experiments = [None]
        n_experiments = 1
    if (Positive&Integer).valid(experiments):
        no_arg_mode = True
        n_experiments = experiments
        experiments = [None]*experiments
    else:
        experiments = list(experiments)
        n_experiments = len(experiments)
    ###########################################################################
    log_file = os.path.join(directory, 'stdout.txt')
    stderr_file = os.path.join(directory, 'stderr.txt')
    results_file = os.path.join(directory, 'results.pkl')
    info_file = os.path.join(directory, 'info.pkl')
    source_file_name = os.path.join(directory, 'source.txt')
    git_file = os.path.join(directory, 'git.txt')
    ###########################################################################
    MSG_START = 'This is scilog entry \'{}\' (ID: {})'
    MSG_EXPERIMENTS = ('Will run {} experiment{}'.format(n_experiments, 's' if n_experiments != 1 else '')
                + (' with arguments: \n\t{}'.format('\n\t'.join(map(str, experiments))) if not no_arg_mode else '.'))
    MSG_INFO = 'This log and all outputs can be found in {}'.format(directory)
    MSG_TYPE = (('#Experiments were' if n_experiments != 1 else '#Experiment was')
                + ' conducted with a {}'.format(func.__class__.__name__)
               + (' called {}'.format(func.__name__) if hasattr(func, '__name__') else '')
              + ' from the module {} whose source code is given below:\n{}')
    MSG_EXCEPTION_ANALYSIS = 'Exception during online analysis. Check {}'.format(stderr_file)
    MSG_ERROR_GIT = 'Error while creating git snapshot Check {}'.format(stderr_file)
    MSG_GIT_DONE = 'Successfully created git commit {}'
    STR_GIT_LOG = '#Created git commit {} in branch {} as snapshot of current state of git repository using the following commands:\n{}'.format('{}', STR_GIT_SCILOG, '{}')
    STR_COMMIT = 'Created for scilog entry {} (ID {}) in {}'
    MSG_SOURCE = 'Could not find source code. Check {}'.format(stderr_file)
    MSG_GIT_START = 'Creating snapshot of current working tree in branch \'{}\' of repository \'{}\'. Check {}'.format(STR_GIT_SCILOG,'{}','{}')
    ###########################################################################
    log = Log(write_filter=True, print_filter=True, file_name=log_file)
    ID = random_string(LEN_ID)
    log.log(MSG_START.format(name, ID))
    log.log(MSG_INFO)
    info = dict()
    info['name'] = name
    info['ID'] = ID
    info['time'] = datetime.datetime.fromtimestamp(time.time())
    info['external'] = external
    if not external:
        info['modules'] = _get_module_versions()
        try:
            source = MSG_TYPE.format(sys.modules[func.__module__].__file__, ''.join(inspect.getsourcelines(sys.modules[func.__module__])[0]))
        except TypeError:
            _store_text(stderr_file, traceback.format_exc())
            log.log(group=GRP_WARN, message=MSG_SOURCE)
        _store_text(source_file_name, source)
    info['parallel'] = parallel
    info['system'] = _get_system_info()
    if supp_data:
        info['supp_data'] = supp_data
    info['runtime'] = [None] * n_experiments
    if memory_profile is not False:
        info['memory'] = [None] * n_experiments
        if memory_profile == 'detail':
            try:
                import memory_profiler  # @UnusedImport
            except ImportError:
                log.log(group=GRP_WARN, message=MSG_MEMPROF)
                memory_profile = True
        else:
            memory_profile = True
    info['status'] = ['queued'] * n_experiments
    if git:
        try:
            log.log(message=MSG_GIT_START.format(os.path.basename(os.path.normpath(module_path)),git_file))
            with capture_output() as c:
                snapshot_id, git_log, _ = _git_snapshot(message=STR_COMMIT.format(name,ID, directory), path=module_path)
            _store_text(git_file, STR_GIT_LOG.format(snapshot_id, git_log))
            # if directory.startswith(os.path.abspath(git_directory)+os.sep):
            # log.log(group=GRP_WARN,message=MSG_WITHIN_GIT)
            log.log(message=MSG_GIT_DONE.format(snapshot_id))
            info['gitcommit'] = snapshot_id
        except GitError as e:
            _store_text(stderr_file, c.stderr + str('Problem with git snapshot. Check stash. ' + e.message))
            _store_text(git_file, e.git_log)
            log.log(group=GRP_ERR, message=MSG_ERROR_GIT)
            raise
    info_list = [info, {'func': external_string if external else func.__repr__()}]
    if not no_arg_mode:
        info_list.append({'experiments':experiments})
    if not no_dill:
        try:
            import dill
            serializer = dill
        except ImportError:
            serializer = pickle
            log.log(group=GRP_WARN, message=MSG_SERIALIZER)
    else:
        serializer = pickle
    def store_info():
        with open(info_file, 'wb') as fp:
            for temp in info_list:
                try:
                    serializer.dump(temp, fp)
                except (TypeError, pickle.PicklingError):
                    log.log(group=GRP_WARN, message=MSG_STORE_INFO(temp.keys()))
    def _update_info(i, runtime, status, memory):
        info['runtime'][i] = runtime
        if memory_profile is not False:
            info['memory'][i] = memory
        info['status'][i] = status
        store_info()
    store_info()
    old_wd = os.getcwd()
    locker = Locker()
    args = ((i, experiment, directory, func, memory_profile,
             runtime_profile, results_file, log_file,
             'pickle' if serializer == pickle else 'dill', no_arg_mode, locker,
             external, link, debug)
            for i, experiment in enumerate(experiments))
    log.log(message=MSG_EXPERIMENTS)
    if parallel:
        try:
            from pathos.multiprocessing import ProcessingPool as Pool
            pool = Pool(nodes=n_experiments)
        except ImportError:
            log.log(group=GRP_WARN, message=MSG_WARN_PARALLEL)
            from multiprocessing import Pool
            pool = Pool(processes=n_experiments)
        try:
            outputs = pool.map(_run_single_experiment, args)
        except _pickle.PicklingError:  # @UndefinedVariable
            log.log(group=GRP_ERR, message=MSG_ERR_PARALLEL)
            raise
        for output in outputs:
            _update_info(*output)
        pool.close()
        pool.join()
    else:
        for arg in args:
            info['status'][arg[0]] = 'running'
            store_info()
            output = _run_single_experiment(arg)
            _update_info(*output)
            if analyze:
                log.log(message=MSG_ANALYSIS_START)
                with locker.analyze_lock:
                    try:
                        globals()['analyze'](func=analyze, path=directory, log=log)
                        #log.log(message=MSG_ANALYSIS_DONE)
                    except:
                        _store_text(stderr_file, traceback.format_exc())
                        log.log(group=GRP_ERR, message=MSG_EXCEPTION_ANALYSIS)
    os.chdir(old_wd)
    log.log(MSG_FINISHED.format('all experiments finished successfully' 
                                if all(s == 'finished' for s in info['status']) 
                                else 'some experiments failed'))
    return directory

class Locker(object):
    def __init__(self):
        mgr = Manager()
        self.lock = mgr.Lock()
        self.analyze_lock = mgr.Lock()

def _get_module_versions():
    names = sys.modules.keys()
    names = [name for name in names if not '.' in name]
    module_versions = {}
    for name in names:
        if hasattr(sys.modules[name], '__version__'):
                module_versions[name] = sys.modules[name].__version__ + '(__version__)'
        # else:
        #    try:
        #        module_versions[name]=pkg_resources.get_distribution(name).version+'(pip)'
        #    except:
        #        pass
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
    def __enter__(self):
        pass
    def __exit__(self):
        pass
def _run_single_experiment(arg):
    (i, experiment, directory, func, memory_profile,
     runtime_profile, results_file, log_file, serializer, no_arg_mode, locker, external,
     link,debug) = arg
    ###########################################################################
    stderr_file = os.path.join(directory, 'stderr.txt')
    stderr_files = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'stderr.txt')
    stdout_files = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'stdout.txt')
    input_files = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'input.txt')
    randomstate_files = lambda i: os.path.join(directory,'experiment{}'.format(i),'randomstate.pkl')
    runtime_profile_files = lambda i:os.path.join(directory, 'experiment{}'.format(i), 'runtime_profile.txt')
    memory_profile_files = lambda i:os.path.join(directory, 'experiment{}'.format(i), 'memory_profile.txt')
    experiment_user_directories = lambda i: os.path.join(directory, 'experiment{}'.format(i), 'user_files')
    MSG_FAILED_EXPERIMENT = lambda i:'Experiment {} failed. Check {}'.format(i, stderr_files(i))
    MSG_EXCEPTION_EXPERIMENT = lambda i: 'Exception during execution of experiment {}. Check {}'.format(i, stderr_file)
    MSG_START_EXPERIMENT = lambda i: ('Runnning experiment {}'.format(i) +
                                      (' with argument:\n\t{}'.format(str(experiment)) if not no_arg_mode else ''))
    ###########################################################################
    log = Log(write_filter=True, print_filter=True, file_name=log_file, lock=locker.lock)
    if serializer == 'pickle':
        serializer = pickle
    else:
        import dill
        serializer = dill
    log.log(MSG_START_EXPERIMENT(i))
    runtime = None
    output = None
    memory = None
    randomstate = None
    status = 'failed'
    if not external:
        try:
            import numpy
            randomstate = numpy.random.get_state()
        except ImportError:
            pass
    if  hasattr(func, '__name__'):
        temp_func = func
    else:
        temp_func = func.__call__
    experiment_directory = experiment_user_directories(i)
    os.makedirs(experiment_directory)
    os.chdir(experiment_directory)
    if external:
        temp_func = partial(temp_func, experiment_directory)
    for f in link:
        root = pathlib.Path(f)
        child = pathlib.Path(os.getcwd())
        if not root in child.parents:
            goal = os.path.basename(os.path.normpath(f))
            os.symlink(f, goal)
    try:
        if not no_arg_mode:
            _store_text(input_files(i), str(experiment))
        if memory_profile is not False:
            m = StringIO()
            if memory_profile == 'detail':
                import memory_profiler
                temp_func = memory_profiler.profile(func=temp_func, stream=m, precision=4)
            else:
                temp_func = measure_memory_usage(func = temp_func, stream = m)
        if runtime_profile:
            temp_func = add_runtime(temp_func)
        stderr_append = ""
        with open(stderr_files(i), 'a', 1) as err:
            with open(stdout_files(i), 'a', 1) as out:
                with contextlib.redirect_stdout(out) if not debug else _no_context():
                    with contextlib.redirect_stderr(err) if not debug else _no_context():
                        tic = timeit.default_timer()
                        try:
                            if no_arg_mode:
                                output = temp_func()
                            else:
                                output = temp_func(experiment)
                            status = 'finished'
                        except Exception:
                            status = 'failed'
                            stderr_append = traceback.format_exc()
        runtime = timeit.default_timer() - tic
        if stderr_append:
            log.log(group=GRP_ERR, message=MSG_FAILED_EXPERIMENT(i))
            _store_text(stderr_files(i), stderr_append)
        if runtime_profile:
            profile, output = output
            s = StringIO()
            ps = pstats.Stats(profile, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats()
            _store_text(runtime_profile_files(i), s.getvalue())
            s.close()
        else:
            _store_text(runtime_profile_files(i), MSG_RUNTIME_SIMPLE(runtime))
        if memory_profile:
            _store_text(memory_profile_files(i), m.getvalue()+('' if memory_profile == 'detail' else 'MB (Use `memory_profile==\'detail\'` for a more detailed breakdown)'))
            memory = _max_mem(m.getvalue(),type=memory_profile)
    except Exception:
        with locker.lock:
            _store_text(stderr_file, traceback.format_exc())
        log.log(group=GRP_ERR, message=MSG_EXCEPTION_EXPERIMENT(i))
    if status == 'finished':
        log.log(MSG_FINISH_EXPERIMENT(i, runtime))
    os.chdir(directory)
    try:
        os.rmdir(experiment_user_directories(i))
    except OSError:
        pass
    with locker.lock:
        with open(results_file, 'ab') as fp:
            try:
                serializer.dump([output], fp)
            except (TypeError, pickle.PicklingError):
                log.log(group=GRP_WARN, message=MSG_STORE_RESULT)
    del output
    gc.collect()
    if randomstate is not None:
        with open(randomstate_files(i),'wb') as fp:
            try:
                serializer.dump(randomstate,fp)
            except (TypeError,pickle.PicklingError):
                log.log(group = GRP_WARN,message = MSG_STORE_RANDOMSTATE)
    return (i, runtime, status, memory)

def _store_text(file_name, data):
    if data:
        with open(file_name, 'a') as fp:
            fp.write(data)

class Component():
    def __init__(self,k):
        self.k = k
    def __call__(self,x):
        return x[self.k]
class ConvergencePlotter():
    def __init__(self,qois = None,cumulative = False):
        if qois == None:
            qois = [lambda x: x]
        self.qois = qois
        self.cumulative = cumulative
    def __call__(self,results,info):
        if len(results)>2:
            times = info['runtime'][:len(results)]
            if self.cumulative:
                times = np.cumsum(times)
            if (Positive&Integer).valid(self.qois):
                self.qois = [Component(k) for k in range(len(results[0]))]
            for (k,qoi) in enumerate(self.qois):
                #try:
                    pyplot.figure(k).clf()
                    plot_convergence(times,[qoi(result) for result in results])
                    #pyplot.draw()
                    #pyplot.pause(0.1)
                    plots.save('convergence')
                #except:
                #    pass


def analyze(func, search_pattern='*', path='', need_unique=False, log=None, no_dill=False):
    if not log:
        log = Log(print_verbosity=True)
    if not no_dill:
        try:
            import dill
            serializer = dill
        except ImportError:
            serializer = pickle
            log.log(group=GRP_WARN, message=MSG_SERIALIZER)
    else:
        serializer = pickle
    MSG_FAILED_ANALYSIS = lambda stderr_file: 'Analysis could not be completed. Check {}'.format(stderr_file)
    MSG_STORE_ANALYSIS = lambda name: 'Could not serialize results of analysis'
    tmp = load(search_pattern=search_pattern, path=path, need_unique=False, no_results=False)
    generator = list(tmp) if need_unique else tmp
    for (info, results, directory) in generator:
        analysis_directory = os.path.join(directory, 'analysis')
        shutil.rmtree(analysis_directory, ignore_errors=True)
        os.mkdir(analysis_directory)
        analysis_user_directory = os.path.join(analysis_directory, 'user_files')
        shutil.rmtree(analysis_user_directory, ignore_errors=True)
        os.mkdir(analysis_user_directory)
        analysis_stderr_file = os.path.join(analysis_directory, 'stderr.txt')
        analysis_stdout_file = os.path.join(analysis_directory, 'stdout.txt')
        analysis_output_file = os.path.join(analysis_directory, 'output.pkl')
        os.chdir(analysis_user_directory)
        output = None
        stderr_append = ''
        with open(analysis_stderr_file, 'a', 1) as err:
            with open(analysis_stdout_file, 'a', 1) as out:
                with contextlib.redirect_stdout(out):
                    with contextlib.redirect_stderr(err):
                        try:
                            output = func(results, info)
                        except Exception:
                            stderr_append = traceback.format_exc()
        if stderr_append:
            if log:
                log.log(group=GRP_ERR, message=MSG_FAILED_ANALYSIS(analysis_stderr_file))
            else:
                warnings.warn(message=MSG_FAILED_ANALYSIS(analysis_stderr_file))
        _store_text(analysis_stderr_file, stderr_append)
        if output is not None:
            with open(analysis_output_file, 'wb') as fp:
                try:
                    serializer.dump(output, fp)
                except (TypeError, pickle.PicklingError):
                    if log:
                        log.log(group=GRP_WARN, message=MSG_STORE_ANALYSIS)
                    else:
                        warnings.warn(message=MSG_STORE_ANALYSIS)
        os.chdir(directory)
        try:
            os.rmdir(analysis_user_directory)
        except OSError:
            pass

def load(search_pattern='*', path='', ID=None, no_results=False, need_unique=True):
    '''
    Load scilog entry/entries.

    Return (generator of) tuple (info,results,directory) with the contents of
    info.pkl and results.pkl as well as the directory of the scilog entry

    :param search_pattern: Shell-style glob/search pattern using wildcards
        If there are multiple entries of the same name (those are stored as
        <name>/v1 <name>/v2 ... in the filesystem) and the should all be returned, 
        use `search_pattern=<name>/v*` and `need_unique=False`
    :type search_pattern: String, e.g. search_pattern='foo*' matches `foobar`
    :param path: Path of exact location is known (possibly only partially), relative or absolute
    :type path: String, e.g. '/home/work/2017/6/<name>' or 'work/2017/6'
    :param no_results: Only load information about scilog entry, not results
    :type no_results: Boolean
    :param need_unique: Require unique identification of scilog entry.
    :type need_unique: Boolean
    :return: Information about run(s) and list(s) of results
    :rtype: If need_unique=True, a single tuple (info[,results],directory),
    where `info` is a dictionary containing information regarding the experiment
    series and `results` is a list containing the results of each experiment.
    If need_unique=False, a generator of tuples (info[,results],directory)
    '''
    deserializer = pickle
    try:
        import dill
        deserializer = dill
    except ImportError:
        warnings.warn(MSG_SERIALIZER)
    def assemble_file_contents(file_name, iterable, need_start=False, update=False):
        try:
            with open(file_name, 'rb') as fp:
                output = iterable()
                for i in itertools.count():
                    try:
                        to_add = deserializer.load(fp)
                    except Exception as e:
                        if not (i == 0 and need_start) and isinstance(e,EOFError):
                            break
                        else:
                            raise
                    if update:
                        output.update(to_add)
                    else:
                        output += to_add
                return output
        except Exception:
            warnings.warn(MSG_ERROR_LOAD('file ' + file_name))
            traceback.print_exc()
    series = []
    if os.sep in search_pattern and path == '':
        path,search_pattern = search_pattern.rsplit(os.sep,1)
    series.extend(files.find_directories(search_pattern, path=path))
    series.extend(files.find_directories('*/' + search_pattern, path=path))
    series = [serie for serie in series if _is_experiment_directory(serie)]
    def get_output(serie, no_results):
        info_file_name = os.path.join(serie, 'info.pkl')
        info = assemble_file_contents(info_file_name, dict, need_start=True, update=True)
        if no_results:
            return (info, serie)
        else:
            results_file_name = os.path.join(serie, 'results.pkl')
            results = assemble_file_contents(results_file_name, list, need_start=False)
            return (info, results, serie)
    if ID:
        if len(ID) < LEN_ID:
            ID = ID + '.{' + str(LEN_ID - len(ID)) + '}'
        regexp = re.compile(ID)
        series = [serie for serie in series if regexp.match(get_output(serie, True)[0]['ID'])]
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
    return os.path.isfile(os.path.join(directory, 'info.pkl'))

def _max_mem(m,type):
    if type == 'detail':#Output of memory_profiler package
        find = re.compile('.*?(\d{1,}\.\d{4}) MiB.*')
        matches = [find.match(line) for line in m.splitlines()]
        values = [float(match.groups()[0]) for match in matches if match is not None]
        return max(values) - min(values)
    else:#Output of measure_memory_usage
        return float(m)

def _get_directory(name, path, no_date):
    if not no_date:
        date = datetime.date.today()
        directory = os.path.join(path, 'w' + date.strftime('%W') + 'y' + str(date.year)[-2:], name)
    else:
        directory = os.path.join(path, name)
    directory = os.path.abspath(directory)
    if os.path.exists(directory) and os.listdir(directory):
        if _is_experiment_directory(directory):  # Previous series will be moved in sub v0, new series will be in sub v1
            split_path = os.path.split(directory)
            temp_rel = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(LEN_TMP))
            temp_directory = os.path.join(split_path[0], '.tmp', temp_rel)
            shutil.move(directory, temp_directory)
            shutil.move(temp_directory, os.path.join(directory, 'v0'))
        candidates = [os.path.split(dir)[1] for dir in os.listdir(directory)  # @ReservedAssignment
                    if os.path.isdir(os.path.join(directory, dir))
                    and re.search('^v([0-9]|[1-9][0-9]+)$', dir)]
        if candidates:
            version = max([int(dir[dir.rindex('v') + 1:]) for dir in candidates]) + 1  # @ReservedAssignment
        else:
            version = 0
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

def _git_snapshot(path, message=''):
    initial_directory = os.getcwd()
    os.chdir(path)
    git_directory = _git_command('rev-parse --show-toplevel', add_input=False).rstrip()
    os.chdir(git_directory)
    active_branch = _git_command('rev-parse --abbrev-ref HEAD', add_input=False)
    if active_branch == STR_GIT_SCILOG:
        raise ValueError(MSG_ERROR_GIT_BRANCH)
    if active_branch == 'HEAD':
        raise ValueError(MSG_ERROR_GIT_DETACHED)
    try:
        out = ''
        tmp = _git_command('stash -u', add_input=False)
        out += '$ git stash -u \n' + tmp
        stash, keep_stash = (tmp.count('\n') > 1), False
        # if stash:
            # try:
            #    out+=_git_command('checkout stash@{0} -- .gitignore')#These lines are necessary to clean
            #    out+=_git_command('checkout stash@{0} -- */.gitignore')#the working directory of newly
            # except CalledProcessError:
            #    pass
            # out+=_git_command('clean -fd')#non-ignored files, to be able to apply the stash later on
        try:
            out += _git_command('checkout {}'.format(STR_GIT_SCILOG))
        except:
            out += _git_command('checkout -b {}'.format(STR_GIT_SCILOG))
        out += active_branch
        old_id = _git_id()
        out += _git_command('merge -s ours {0} --no-edit -m "Snapshot of working directory of branch {0} \n {1}"'.format(active_branch, message))
        new_id = _git_id()
        if new_id == old_id:
            out += _git_command('commit --allow-empty -m "Snapshot of working directory of branch {0} \n {1}"'.format(active_branch, message))
        out += _git_command('checkout --detach {}'.format(active_branch))
        out += _git_command('reset --soft {}'.format(STR_GIT_SCILOG))
        out += _git_command('checkout {}'.format(STR_GIT_SCILOG))
        out += _git_command('commit --allow-empty --amend -C HEAD')
        if stash:
            try:
                out += _git_command('stash apply --index')
            except subprocess.CalledProcessError as e:
                out += e.output
                try:
                    out += _git_command('stash apply --index')  # On second try, there is even more files that prevent the `stash apply`. To get these ...
                except subprocess.CalledProcessError as e:  # ...this exception is used
                    out += e.output
                    lines = e.output.splitlines()
                    for line in lines[:-2]:
                        out += _git_command(['rm', line.split(' ')[0]])
                    out += _git_command('stash apply --index')  # After removal of all preventing files, this should now work
        out += _git_command('add --all')
        out += _git_command('commit --allow-empty --amend -C HEAD')
        id = _git_id()  # @ReservedAssignment
        out += _git_command('checkout {}'.format(active_branch))
        if stash:
            try:
                out += _git_command('stash apply --index')
            except subprocess.CalledProcessError as e:
                out += e.output
                keep_stash = True
            if not keep_stash:
                out += _git_command('stash drop')
    except subprocess.CalledProcessError as e:
        raise GitError(traceback.format_exc(), out + '\n' + e.output)
    except:
        raise GitError(traceback.format_exc(), out)
    if keep_stash:
        raise GitError('Your previous working tree is stashed, but could not be reapplied.', out)
    os.chdir(initial_directory)
    return id, out, git_directory

def main():
    import textwrap as _textwrap
    class LineWrapRawTextHelpFormatter(argparse.RawDescriptionHelpFormatter):
        def _split_lines(self, text, width):
            text = self._whitespace_matcher.sub(' ', text).strip()
            return _textwrap.wrap(text, width)
    parser = argparse.ArgumentParser(formatter_class=LineWrapRawTextHelpFormatter,
        description=
        '''
        Performs experiment series and stores and retrieves results
        and runtime information in form of structured directories on the file system.

        For the creation of scilog entries, a given function is called once for
        each item of given list of experiments and stores results along with
        auxiliary information such as system information and runtime and memory usage.

        Each scilog entry consists of a directory with the following structure:
            *info.pkl:
                *name: Name of scilog entry (str)
                *ID: Alphanumeric 8 character string identifying the scilog entry
                *modules: Module versions
                *time: Time of execution (datetime.datetime)
                *experiments: Input `experiments`
                *runtime: Runtime of each experiment (list of floats)
                *status: Status of each experiment (list of ('queued'/'finished'/'failed'))
                *supp_data: Command line arguments that were passed to this function
            *stdout.txt
            *(optional)git.txt
            *results.pkl: List of results of experiments
            *source.txt: Source code of the specified
            *(optional)stderr.txt
            *For each experiment a subdirectory "experiment<i>" with:
                *user_files/ (Working directory for call of specified function)
                *(optional)input.txt: String representation of arguments
                *stderr.txt
                *stdout.txt
                *(optional)runtime_profile.txt: Extensive runtime information for each experiment (list of strings)
                *(optional)memory_profile.txt: Memory usage information for each experiment (list of strings)
            *(optional) analysis/: output of function :analysis:
                *stderr.txt
                *stdout.txt
                *user_files/ (Working directory for call of specified analysis function)

        Both info.pkl and results.pkl are created using the package pickle.
        They contain multiple concatenated pickle streams.
        To load these files and automatically join the contents of info.pkl into
        a single dictionary and the contents of results.pkl into a single list,
        the Python function :code:`scilog.load` may be used.

        To display information about an existing scilog entry, the switch --show may be used
        with this script.)
        ''')
    # parser.register('type', 'bool',
    #                lambda v: v.lower() in ("yes", "true", "t", "1", "y"))
    parser.add_argument("func", action='store',
        help=
        '''
        Specifies a function that performs the experiments.

        The standard way is to provide the full path of a Python function
        e.g.: `foo.func`.

        There are three alternatives:
        1) Provide the full name of a module that contains a class of the same name (up to capitalization).
        e.g.: `foo.bar`

        2) Provide the full name of a class.
        e.g.: `foo.bar.Bar2`

        3) Provide a bash command string, e.g.: `echo {}s`

        In both cases above, the specified class is instantiated
         and all experiments are performed by calling this instance.
        ''')
    parser.add_argument('-e', '--experiments', action='store', default='None',
        help=
        '''
        List of experiment configurations.

        e.g.: [2**l for l in range(10)]

        If no list of experiments is specified, FUNC is called once without arguments.

        If FUNC is a bash command string, the entires of experiments must be
        strings and are used to format FUNC (using str.format)
        ''')
    parser.add_argument('-b', '--base', action='store', default='{}',
        help=
        '''
        Base configuration (in form of a dictionary) for all experiments.

        If argument FUNC is a function, this dictionary is passed
        along each experiment in form of keyword arguments to FUNC.

        If argument FUNC specifies a class, the class is instantiated using
        this dictionary in form of keyword arguments.

        If argument FUNC is a bash command string, the entries of this dictionary
        are passed as keyword arguments along the experiment to format the string.
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

        The specified function is called with a tuple (results,info)
        containing the content of the files results.pkl and info.pkl described above, respectively.

        By default, ANALYZE is the name of a function in the same module as FUNC.

        Alternatively, ANALYZE can be
        1) a full name of a function in some different module,
        e.g.: foo2.analyze

        2) a name of a method of the class specified by FUNC
        ''')
    parser.add_argument('-o', '--output', action='store', default='scilog',
        help=
        '''
        Specify output directory
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
        Create git snapshot in branch {}.
        '''.format(STR_GIT_SCILOG))
    parser.add_argument('--no_date', action='store_true',
        help=
        '''
        Do not store scilog entry in subdirectories based on current date.
        ''')
    parser.add_argument('--no_dill', action='store_true',
        help=
        '''
        Do not use dill to store info.pkl and results.pkl. This
        is probably a bad idea.
        ''')
    parser.add_argument('--external', action='store_true',
        help=
        '''
        Specify that FUNC describes an external call.
        This is only needed, when FUNC looks like a Python module, e.g.:
        FUNC=`foo.bar`
        ''')
    parser.add_argument('-s', '--show', action='store_true',
        help=
        '''
        Print information of previous entry instead of creating new entry.

        In this case, FUNC must be the path to a previous entry.
        (Shell-style wildcards, e.g. 'foo*', are recognized)
        Furthermore, the --git flag triggers an interactive view of the differences
        of the working directory and the repository at the time of the creation of the scilog entry
        ''')
    parser.add_argument('-d', '--directory', action='store', nargs='?', const='.',
        default=None,
        help=
        '''
        Run FUNC with copy of specified directory as working directory

        If no argument is specified, current working directory is used.
        ''')
    args = parser.parse_args()
    if args.show:
        entries = load(search_pattern=args.func, no_results=True, need_unique=False)
        entries = list(entries)
        print('Found {} entr{}'.format(len(entries), 'y' if len(entries) == 1 else 'ies'))
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
        conduct(
            func=fn, path=args.output,
            experiments=args.experiments,
            name=args.name,
            external=args.external,
            analyze=analyze_fn,
            runtime_profile=args.runtime_profile,
            memory_profile=args.memory_profile,
            git=args.git,
            no_date=args.no_date,
            no_dill=args.no_dill,
            parallel=args.parallel,
            module_path=module_path,
            working_directory=args.directory
        )
if __name__ == '__main__':
    main()
