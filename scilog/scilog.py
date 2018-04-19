import timeit
import pickle
import os
import errno
import datetime
import shutil
import warnings
import traceback
import pstats
import io
import sys
import gc
import inspect
import importlib
import re
import subprocess
import shlex
import json
import contextlib
import stat
import argparse
import ast
import builtins
import signal

import numpy as np
from matplotlib import pyplot
from IPython.utils.capture import capture_output

from swutil import sys_info, np_tools, plots, aux
from swutil.validation import Positive, Integer
from swutil.logs import Log
from swutil.hpc import Locker
from swutil.aux import  string_dialog, no_context, random_word,\
    string_from_seconds
from swutil.files import append_text, delete_empty_files,\
    delete_empty_directories, find_directories
from swutil.decorators import print_peak_memory, add_runtime
from swutil.collections import unique

class GitError(Exception):
    def __init__(self, message, git_log):
        super(GitError, self).__init__(message)
        self.git_log = git_log
        
GRP_WARN = 'Warning'
GRP_ERROR = 'Error'
FILE_DEBUG = '.debug'
FILE_OUTPUT = 'output.pkl'
FILE_INPUT = 'input.pkl'
FILE_INFO = 'summary.txt'
FILE_AUX = 'aux_data.pkl'
FILE_RUNTIME = 'runtime.txt'
FILE_MEMORY = 'memory.txt'
FILE_LOAD = 'load.sh'
FILE_EXP_ERR = 'stderr.txt'
FILE_EXP_OUT = 'stdout.txt'
FILE_LOG = 'log.txt'
FILE_GITLOG = 'git.txt'
FILE_ERR = 'err.txt'
FILE_SOURCE = 'source.txt'
FILE_WD = 'working_directory'
FILE_ANALYSIS = 'analysis'
FILE_EXP = lambda i: 'experiment{}'.format(i)
FILE_RANDOMSTATE = 'randomstate.pkl'
STR_GIT_TAG = lambda ID: 'scilog_{}'.format(ID)
STR_GIT_LOG = lambda sha1, log: '#Created git commit {} as snapshot of current state of git repository using the following commands:\n{}'.format(sha1,log)
STR_GIT_COMMIT_TITLE = lambda branch: 'Snapshot of working directory of branch {}'
STR_GIT_COMMIT_BODY = lambda name, ID, directory: 'Created for scilog entry {}/{} in {}'.format(name,ID,directory)
STR_LOADSCRIPT = ('#!/bin/sh \n '
                     + ' xterm -e {} -i -c '.format(sys.executable)
                     + r'''"
print('>>> import scilog'); 
import scilog;
print('>>> entry = scilog.load()');
entry = scilog.load();
try:
  import pandas as pd;
  print('>>> import pandas as pd');
  print('>>> experiments = pd.DataFrame(entry[\'experiments\'])');
  experiments = pd.DataFrame(entry['experiments']);
  print(experiments);
except:
  pass;"''')
STR_MEMFILE = lambda value,memory_profile: value + (
                    '' if memory_profile == 'detail' 
                    else 'MB (Use `memory_profile==\'detail\'` for a more detailed breakdown)'
                )
STR_SOURCE = lambda n, func, module,source: (('#Experiments were' if n != 1 else '#Experiment was')
                + ' conducted with a {}'.format(func.__class__.__name__)
               + (' called {}'.format(func.__name__) if hasattr(func, '__name__') else '')
              + ' from the module {} whose source code is given below:\n{}'.format(module,source))
STR_TIME = '%y-%m-%d %H:%M:%S'
STR_KEYWORDS_PROMPT = 'Enter keywords: '
STR_KEYWORDS_FORMAT = 'Keywords must be passed in the form `<key>[,<key>]*` or `<key>:<value>[,<key>:<value>]*` where <key> and <value> are Python literals'
STR_GITDIFF = '\n The current working directory differs from the git repository at the time of the scilog entry as follows:'
STR_ENTRY = lambda entry: ('=' * 80+'\nEntry \'{}\' at {}:\n'.format(entry['name'], entry['path'])
            + '=' * 80 + '\n' + json.dumps(entry, sort_keys=True, indent=4, default=str)[1:-1])
STR_MULTI_ENTRIES = lambda n:'Found {} entries'.format(n)
MSG_DEBUG =  'Scilog is run in debug mode. Entry is not stored permanently, stdout and stderr are not captured, no git commit will be created'
MSG_START_ANALYSIS = 'Updating analysis'
MSG_START_EXPERIMENT = lambda i,inp: ('Running experiment {}'.format(i) + 
        (' with input{} {}'.format('\n\t' if '\n' in repr(inp) else '',repr(inp))
          if inp is not None else ''))
MSG_START_GIT = lambda repo:'Creating snapshot of current working tree of repository \'{}\'. Check {}'.format(repo,FILE_GITLOG)
def MSG_START_EXPERIMENTS(n,no_arg_mode,inputs):
    msg = 'Will run {} experiment{}'.format(n, 's' if n != 1 else '')
    if not no_arg_mode:
        strings = list(map(repr,inputs))
        sep = '\n\t' if any('\n' in s for s in strings) else ', '
        msg += ' with inputs \n\t{}'.format(sep.join(strings))
    return msg 
MSG_START_ENTRY = lambda directory: 'Scilog entry {}'.format(directory)
MSG_FINISH_EXPERIMENT = lambda i,runtime,result: 'Experiment {} finished (Elapsed time: {}){}'.format(i,string_from_seconds(runtime),'. Output: {}'.format(result))
MSG_FINISH_ENTRY_SUCCESS = 'Scilog entry completed -- all experiments finished successfully'
MSG_FINISH_ENTRY_FAIL = 'Scilog entry completed -- some experiments failed'
MSG_FINISH_GIT = lambda sha1: 'Successfully created git commit {}'.format(sha1)
MSG_ERROR_NOMATCH = 'Could not find matching scilog entry'
MSG_ERROR_MULTIMATCH = lambda entries:'Multiple matching scilog entries (to iterate through all use need_unique=False):\n{}'.format('\n'.join(entries))
MSG_ERROR_LOAD = lambda name: 'Error loading {}. Are all required modules in the Python path?'.format(name)
MSG_ERROR_PARALLEL = 'Error during parallel execution. Try running with `parallel=False`'
MSG_ERROR_BASH_ANALYSIS = 'Cannot analyze output in bash mode'
MSG_ERROR_GIT = lambda file:'Error during git snapshot creation. Check {}'.format(file)
MSG_ERROR_EXPERIMENT = lambda i,file:'Experiment {} failed. Check {}'.format(i, file)
MSG_ERROR_ANALYSIS = lambda file: 'Analysis could not be completed. Check {}'.format(file)
MSG_ERROR_DIR = 'Could not create scilog entry directory'
MSG_EXCEPTION_STORE = lambda file: 'Could not store {}'.format(file)
MSG_EXCEPTION_ANALYSIS = 'Exception during online analysis'
MSG_EXCEPTION_EXPERIMENT = lambda i:'Exception during execution of experiment {}'.format(i)
MSG_WARN_SOURCE = 'Could not find source code'
MSG_WARN_LOADSCRIPT = 'Error during load script creation'
MSG_WARN_PARALLEL = ('Could not find pathos. This might cause problems with parallel execution.'
    + 'Install pathos via `pip install pathos`.')
MSG_WARN_MEMPROF = 'Could not find memory_profiler. Install memory_profiler via `pip install memory_profiler`.'
MSG_WARN_DILL = ('Could not find dill. Some items might not be storable. '
                  + 'Storage of numpy arrays will be slow'
                  + 'Install dill via `pip install dill`.')
MSG_INTERRUPT = 'Kill signal received. Stored {}, closing now.'.format(FILE_INFO)
LEN_ID = 8

# TODO: Add keyword based load
def record(func, inputs=None, name=None, directory=None, aux_data=None,
            analysis=None, runtime_profile=False, memory_profile=False,
            git=False, no_date=False, parallel=False,
            external=False, copy_output = None, keywords=None,debug = None):
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

    Scilog creates a directory -- specified by :code:`directory`, :code:`name`, 
    and a randomly generated ID -- with the following content:
        *summary.txt:
            *name: Name of scilog entry
            *ID: Alphanumeric string identifying the entry
            *modules: Module versions
            *time: Time of execution
            *experiments: For each experiment
                *string representation of input, 
                *string representation of output,
                *runtime
                *status
                *(optional)peak memory usage
            *keywords: Further entry description
            *(optional): git_commit
            *(optional)aux_data: Argument :code:`aux_data`
        *log.txt
        *(optional)err.txt
        *(optional)git.txt: stdout of git snapshot creation 
        *source.txt: Source code of the module containing :code:`func`
        *For each experiment a subdirectory 'experiment<i>' with:
            *output.pkl: Output of :code:`func`
            *(optional)input.pkl: Argument passed to :code:`func`
            *(optional) working_directory/: Working directory for call of :code:`func`, 
                unless parameter :code:`working_directory` is specified
            *(optional)stderr.txt:
            *(optional)stdout.txt:
            *(optional)runtime_profile.txt: Extensive runtime information for each experiment
            *(optional)memory_profile.txt: Memory usage information for each experiment
        *(optional) analysis/: output of function :code:`analysis`
            *(optional)stderr.txt
            *(optional)stdout.txt
            *(optional)working_directory/: Working directory for call of :code:`analysis`

    To load the contents of summary.txt in Python, use the function :code:`scilog.load`.
    That function additionally replaces the string representations of outputs and inputs in 
    summary.txt by the actual Python object outputs and inputs. 

    :param func: Function to be called with different experiment configurations
    :type func: function
    :param inputs: List of inputs to :code:`func`. 
        If not specified, then :code:`func` is called once, without arguments
        If passed and integer, then `func` is called as often as specified, without arguments.
    :type inputs: List(-like) or Integer
    :param name: Name of scilog entry. 
        If not specified, :code:`func.__name__` is used
    :type name: String
    :param directory: Root directory for storage
    :type directory: String
    :param aux_data: Auxiliary data that should be stored along with the results
    :type aux_data: Any
    :param analysis: Function that is called after each experiment 
        Can be used, e.g., for plotting
    :param runtime_profile: Store extensive runtime information
        May slow down execution
    :type runtime_profile: Boolean
    :param memory_profile: Track memory usage
        May slow down execution
    type memory_profile: Boolean
    :param git: Create git snapshot commit
        The resulting commit is tagged with the entry ID and resides outside the branch history
        (Should you ever want get rid of the snapshots do `git tag --list '_scilog*'|xargs -I % git tag -d %`)
        The explicit path may be specified, else it will be automatically detected
    :type git: Boolean or String
    :param no_date: Do not store outputs in sub-directories grouped by calendar week
    :type date: Boolean
    :param external: Specify whether :code:`func` is a Python function or a string
        representing an external call, such as 'echo {}'
        If True, curly braces in the string get replaced by the items of :code:`inputs`
    :type external: Boolean
    :param keywords: Keywords describing the entry further
    :type keywords: Set or dictionary
    :param debug: Force debug mode (otherwise detected automatically)
    :param debug: Boolean
    :param copy_output:  The contents of this directory will be copied into the scilog entry directory
    :type copy_output: String
    
    :return: Directory of scilog entry
    :rtype: String
    '''
    debug = debug if debug is not None else aux.isdebugging()
    if debug: 
        git = False
    elif git is True:
        git = os.path.dirname(sys.modules[func.__module__].__file__)
    if external:
        external_string = func
        def func(*experiment):
            subprocess.check_call(external_string.format(*experiment), stdout=sys.stdout, stderr=sys.stderr, shell=True)
    if not name:
        if external:
            oneword = re.compile('\w+')
            name = oneword.match(external_string).group(0)
        else:
            try:
                name = func.__name__
            except AttributeError:
                name = func.__class__.__name__
    directory = directory or os.path.join(os.path.dirname(sys.modules[func.__module__].__file__),'scilog')
    directory,ID = _get_directory(name, directory, no_date, debug,git)
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
    log_file = os.path.join(directory, FILE_LOG)
    err_file = os.path.join(directory, FILE_ERR)
    info_file = os.path.join(directory, FILE_INFO)
    load_file = os.path.join(directory, FILE_LOAD)
    aux_data_file = os.path.join(directory, FILE_AUX)
    source_file_name = os.path.join(directory, FILE_SOURCE)
    git_file = os.path.join(directory, FILE_GITLOG)
    locker = Locker()
    _log = Log(write_filter=True, print_filter=True, file_name=log_file,lock = locker.get_lock())  # Logging strategy: 1) Redirect out and err of user functions (analysis and experiment) to their own files
    _err = Log(write_filter=True, print_filter=False, file_name=err_file,lock = locker.get_lock())  # 2) Log errors outside user functions in _err 3) Log everything (user-err and _err, as well as other info) in _log 
    _log.log(MSG_START_ENTRY(directory))
    if debug: _log.log(group =GRP_WARN,message = MSG_DEBUG)
    info = dict()
    if keywords is None:  # Could be None if previous failed or because None are desired
        keywords = {}
        try:
            keywords = vars(func)
        except TypeError:# func has no __dict__
            pass
        if not keywords:#empty
            while True:
                try:
                    keyword_string = string_dialog('scilog',STR_KEYWORDS_PROMPT)
                except Exception:
                    keyword_string = input(STR_KEYWORDS_PROMPT)
                try:
                    keywords = ast.literal_eval('{'+keyword_string+'}')
                except (ValueError,SyntaxError):
                    warnings.warn(STR_KEYWORDS_FORMAT)
                else:
                    break
    if isinstance(keywords,builtins.set):
        keywords = {str(key): True for key in keywords}
    else:
        keywords = {str(key):str(keywords[key]) for key in keywords}
    info['keywords'] = keywords
    info['name'] = name
    info['ID'] = ID
    info['time'] = datetime.datetime.now().strftime(STR_TIME)
    info['func'] = external_string if external else func.__repr__()
    info['experiments'] = {'runtime':[None] * n_experiments, 'memory':[None] * n_experiments, 'output':[None] * n_experiments, 'status':['queued'] * n_experiments}
    if not no_arg_mode:
        info['experiments'].update({'input':[str(input) for input in inputs]})
    if memory_profile is not False:
        info['experiments'].update({'memory':[None] * n_experiments})
    if not external:
        info['modules'] = sys_info.modules()
        try:
            source = STR_SOURCE(n_experiments,func,sys.modules[func.__module__].__file__, ''.join(inspect.getsourcelines(sys.modules[func.__module__])[0]))
            append_text(source_file_name, source)
        except Exception:  # TypeError only?
            _err.log(traceback.format_exc())
            _log.log(group=GRP_WARN, message=MSG_WARN_SOURCE)
    else:
        info['modules'] = None
    info['parallel'] = parallel
    info['hardware'] = sys_info.hardware()
    if memory_profile is not False:
        if memory_profile == 'detail':
            try:
                import memory_profiler  # @UnusedImport
            except ImportError:
                _log.log(group=GRP_WARN, message=MSG_WARN_MEMPROF)
                memory_profile = True
        else:
            memory_profile = True
    try: 
        with open(load_file, 'w') as fp:
            fp.write(STR_LOADSCRIPT)
        st = os.stat(load_file)
        os.chmod(load_file, st.st_mode | stat.S_IEXEC)
    except Exception:
        _err.log(message=traceback.format_exc())
        _log.log(group=GRP_WARN, message=MSG_WARN_LOADSCRIPT)
    if git:
        try:
            _log.log(message=MSG_START_GIT(os.path.basename(os.path.normpath(git))))
            with (capture_output() if not debug else no_context()) as c:
                snapshot_id, git_log, _ = _git_snapshot(path=git,commit_body=STR_GIT_COMMIT_BODY(name, ID, directory), ID=ID)
            append_text(git_file, STR_GIT_LOG(snapshot_id, git_log))
            _log.log(message=MSG_FINISH_GIT(snapshot_id))
            info['gitcommit'] = snapshot_id
        except GitError as e:
            _log.log(group=GRP_ERROR, message=MSG_ERROR_GIT(git_file))
            _err.log(message=str(e)+'\n'+c.stderr)
            append_text(git_file, e.git_log)
            raise
    else:
        info['gitcommit'] = None
    try:
        import dill
        serializer = dill
    except ImportError:
        serializer = pickle
        _log.log(group=GRP_WARN, message=MSG_WARN_DILL)
    try_store(aux_data,serializer,aux_data_file,_log,_err)
    def _update_info(i, runtime, status, memory, output_str):
        info['experiments']['runtime'][i] = runtime
        if memory_profile is not False:
            info['experiments']['memory'][i] = memory
        info['experiments']['status'][i] = status
        info['experiments']['output'][i] = output_str
        store_info()
    def store_info():
        with open(info_file,'w') as fp:
            json.dump(info,fp,indent = 1,separators = (',\n', ': '))
    store_info()
    old_wd = os.getcwd()
    args = ((i, input, directory, func, memory_profile,
             runtime_profile, _log,_err,
             'pickle' if serializer == pickle else 'dill', no_arg_mode,
            external, debug, copy_output)
            for i, input in enumerate(inputs))
    _log.log(message=MSG_START_EXPERIMENTS(n_experiments,no_arg_mode,inputs))
    def handler_stop_signals(*args):
        store_info()
        _log.log(MSG_INTERRUPT)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
    signal.signal(signal.SIGINT, handler_stop_signals)
    signal.signal(signal.SIGTERM, handler_stop_signals)
    signal.signal(signal.SIGABRT,handler_stop_signals)
    signal.signal(signal.SIGHUP,handler_stop_signals)
    if parallel and not debug:
        try:
            from pathos.multiprocessing import ProcessingPool as Pool
            pool = Pool(nodes=n_experiments)
        except ImportError:
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_WARN, message=MSG_WARN_PARALLEL)
            from multiprocessing import Pool
            pool = Pool(processes=n_experiments)
        info['experiments']['status'] = ['running']*n_experiments
        try:
            outputs = pool.map(_run_single_experiment, args)
        except _pickle.PicklingError:  # @UndefinedVariable
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_ERROR, message=MSG_ERROR_PARALLEL)
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
                _err.log(message=traceback.format_exc())
                _log.log(group=GRP_ERROR, message=MSG_EXCEPTION_EXPERIMENT(arg[0]))
            else:
                _update_info(*output)
            if analysis:
                _log.log(message=MSG_START_ANALYSIS)
                try:
                    with capture_output():
                        entry = load(path=directory, need_unique=True, no_objects=False)
                    analyze(func=analysis, entry=entry, _log=_log, _err=_err, debug=debug)
                except Exception:
                    _err.log(message=traceback.format_exc())
                    _log.log(group=GRP_ERROR, message=MSG_EXCEPTION_ANALYSIS)
    os.chdir(old_wd)
    _log.log(MSG_FINISH_ENTRY_SUCCESS 
             if all(s == 'finished' for s in info['experiments']['status']) 
             else MSG_FINISH_ENTRY_FAIL)
    return directory

def _run_single_experiment(arg):
    (i, input, directory, func, memory_profile,
     runtime_profile, _log,_err, serializer, no_arg_mode,
     external, debug, copy_output) = arg
    experiment_directory = os.path.join(directory, FILE_EXP(i))
    stderr_file = os.path.join(experiment_directory, FILE_EXP_ERR)
    stdout_file = os.path.join(experiment_directory, FILE_EXP_OUT)
    input_file = os.path.join(experiment_directory, FILE_INPUT)
    output_file = os.path.join(experiment_directory, FILE_OUTPUT)
    randomstate_file = os.path.join(experiment_directory, FILE_RANDOMSTATE)
    runtime_profile_file = os.path.join(experiment_directory, FILE_RUNTIME)
    memory_profile_file = os.path.join(experiment_directory, FILE_MEMORY)
    experiment_working_directory = os.path.join(experiment_directory, FILE_WD)
    if serializer == 'pickle':
        serializer = pickle
    else:
        import dill
        serializer = dill
    _log.log(MSG_START_EXPERIMENT(i,input if not no_arg_mode else None))
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
    if copy_output is None:
        os.makedirs(experiment_working_directory)
        os.chdir(experiment_working_directory)
    try:
        if not no_arg_mode:
            try_store(input,serializer,input_file,_log,_err)
        try_store(randomstate,serializer,randomstate_file,_log,_err)
        if memory_profile == 'detail':#Needs to be before runtime decorator so it gets the full view (otherwise it will profile the runtime decorator)
            m = io.StringIO()
            import memory_profiler
            temp_func = memory_profiler.profile(func = temp_func,stream =m ,precision = 4)
        if runtime_profile:
            temp_func = add_runtime(temp_func)
        if memory_profile is True:#Needs to be after runtime decorator so runtime gets the full view (since peak_memory is threaded)
            m = io.StringIO()
            temp_func = print_peak_memory(func=temp_func, stream=m)
        stderr_append = ''
        with open(stderr_file, 'a', 1) as err:
            with open(stdout_file, 'a', 1) as out:
                with contextlib.redirect_stdout(out) if not debug else no_context():
                    with contextlib.redirect_stderr(err) if not debug else no_context():
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
        delete_empty_files([stderr_file, stdout_file])
        runtime = timeit.default_timer() - tic
        if status == 'failed':
            append_text(stderr_file, stderr_append)
            _log.log(group=GRP_ERROR, message=MSG_ERROR_EXPERIMENT(i,stderr_file))
        else:
            if runtime_profile:
                profile, output = output
                s = io.StringIO()
                ps = pstats.Stats(profile, stream=s)
                ps.sort_stats('cumulative')
                ps.print_stats()
                append_text(runtime_profile_file, s.getvalue())
                s.close()
            if memory_profile:
                append_text(memory_profile_file,STR_MEMFILE(m.getvalue(),memory_profile))
                memory = _max_mem(m.getvalue(), type=memory_profile)
    except Exception:
        _err.log(message=traceback.format_exc())
        _log.log(group=GRP_ERROR, message=MSG_EXCEPTION_EXPERIMENT(i))
    if copy_output is None:
        os.chdir(directory)
    else:
        shutil.copytree(copy_output, experiment_working_directory, symlinks=False, ignore_dangling_symlinks=True)
    delete_empty_directories([experiment_working_directory])
    output_str = str(output)
    try_store(output,serializer,output_file,_log,_err)
    del output
    if status == 'finished':
        _log.log(MSG_FINISH_EXPERIMENT(i, runtime, output_str))
    gc.collect()
    return (i, runtime, status, memory, output_str)

class ConvergencePlotter():
    def __init__(self, *qois, cumulative=False, work=None, extrapolate=0,reference = 'self'):
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
        self.reference = reference
    def __call__(self, entry):
        experiments = entry['experiments']
        results = experiments['output']
        single_reference = (self.reference == 'self')
        ind_finished = [j for (j, status) in enumerate(experiments['status']) if status == 'finished']
        if len(ind_finished) > 2 + (self.extrapolate if self.extrapolate >= 0 else 0):
            if self.work is None:
                times = [experiments['runtime'][i] for i in ind_finished]
            else:
                times = [self.work(i) for i in ind_finished]
            results = [results[i] for i in ind_finished]
            if self.cumulative:
                times = np.cumsum(times)
            if not self.qois:
                if hasattr(results[0], '__len__') and not isinstance(results[0], np.ndarray):
                    self.qois = [lambda x,k = k: x[k] for k in range(len(results[0]))]
                else:
                    self.qois = [lambda x:x]
                    single_reference = True
            if single_reference:
                self.reference = [self.reference]*len(self.qois)
            for (k, qoi) in enumerate(self.qois):
                try:
                    pyplot.figure(k).clf()
                    qoi_values = np.array([qoi(result) for result in results])
                    qoi_times = np.array(times)
                    if self.extrapolate:
                        qoi_values, qoi_times = np_tools.extrapolate(qoi_values, qoi_times, self.extrapolate)
                    plots.plot_convergence(qoi_times, qoi_values,reference = self.reference[k]) 
                    plots.save('convergence')
                except Exception:
                    traceback.print_exc()
                    
def try_store(what,serializer,file,_log,_err):
    if what is not None:
        try:
            with open(file, 'wb') as fp:
                serializer.dump(what, fp)
        except (TypeError, pickle.PicklingError):
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_WARN, message=MSG_EXCEPTION_STORE(os.path.split(file)[-1]))

def analyze(entry,func, _log=None, _err=None, debug=False):
    '''
    Add analysis to scilog entry or entries
    
    :param func: Function that performs analysis
    :param entry: scilog entry or entries (as returned by scilog.load)
    :param _log: Log object to be used instead of writing to standard stdout
    :param _err: Log object to be used instead of writing to standard stderr
    :param debug: If True, output is printed instead of being redirected into files
    '''
    if not _log:
        _log = Log(print_filter=True)
    if not _err:
        _err = Log(print_filter=True)
    try:
        import dill
        serializer = dill
    except ImportError:
        serializer = pickle
        _log.log(group=GRP_WARN, message=MSG_WARN_DILL)
    cwd = os.getcwd()
    try:
        if not inspect.isgenerator(entry):
            entries = [entry]
        else:
            entries = entry
        for entry in entries:
            analysis_directory_tmp = os.path.join(entry['path'], 'tmp',FILE_ANALYSIS)
            working_directory = os.path.join(analysis_directory_tmp, FILE_WD)
            stderr_file = os.path.join(analysis_directory_tmp, FILE_EXP_ERR)
            stdout_file = os.path.join(analysis_directory_tmp, FILE_EXP_OUT)
            output_file = os.path.join(analysis_directory_tmp, FILE_OUTPUT)
            os.makedirs(analysis_directory_tmp)
            os.mkdir(working_directory)
            os.chdir(working_directory)
            output = None
            stderr_append = ''
            with open(stderr_file, 'a', 1) as err:
                with open(stdout_file, 'a', 1) as out:
                    with contextlib.redirect_stdout(out) if not debug else no_context():
                        with contextlib.redirect_stderr(err) if not debug else no_context():
                            try:
                                output = func(entry)
                            except Exception:
                                stderr_append = traceback.format_exc()
            delete_empty_files([stderr_file, stdout_file])
            delete_empty_directories([working_directory])
            if stderr_append:
                append_text(stderr_file, stderr_append)
                _log.log(group=GRP_ERROR, message=MSG_ERROR_ANALYSIS(stderr_file))
            try_store(output,serializer,output_file,_log,_err)
            os.chdir(cwd)
            analysis_directory = os.path.join(entry['path'], FILE_ANALYSIS)
            shutil.rmtree(analysis_directory, ignore_errors=True)
            shutil.move(analysis_directory_tmp, entry['path'])
            shutil.rmtree(os.path.split(analysis_directory_tmp)[0], ignore_errors=True)
    except Exception:
        os.chdir(cwd)
        
def _keyword_match(template, test):
    if isinstance(template,builtins.dict):
        return all(key in test and re.search(template[key],test[key]) for key in template)
    else:
        return all(key in test for key in template)
              
def load(search_pattern='*', path='', ID=None, no_objects=False, need_unique=True,keywords = None):
    '''
    Load scilog entry/entries.
   
    :param search_pattern: Shell-style glob/search pattern using wildcards
        If there are multiple entries of the same name (those are stored as
        <name>/v0 <name>/v1 ... in the filesystem) and they should all be returned, 
        use `search_pattern=<name>/v*` and `need_unique=False`
    :type search_pattern: String, e.g. search_pattern='foo*' matches `foobar`
    :param path: Path of exact location is known (possibly only partially), relative or absolute
    :type path: String, e.g. '/home/work/2017/6/<name>' or 'work/2017/6'
    :param no_objects: Only load information about scilog entry, not results
    :type no_objects: Boolean
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
        warnings.warn(MSG_WARN_DILL)
    entries = []
    if os.sep in search_pattern and path == '':
        temp_path, temp_search_pattern = search_pattern.rsplit(os.sep, 1)
        if os.path.isabs(temp_path):
            path, search_pattern = temp_path, temp_search_pattern
    entries.extend(find_directories(search_pattern, path=path))
    entries.extend(find_directories('*/' + search_pattern, path=path))
    entries = [entry for entry in entries if _is_experiment_directory(entry)]
    def get_output(entry, no_objects):
        file_name = os.path.join(entry, FILE_INFO)
        with open(file_name, 'r') as fp:
            info = json.load(fp)
        info['path'] = entry
        if not no_objects:
            for (j, status) in enumerate(info['experiments']['status']):
                if status == 'finished':
                    try:
                        output_file_name = os.path.join(entry, FILE_EXP(j), FILE_OUTPUT)
                        with open(output_file_name, 'rb') as fp:
                            output = deserializer.load(fp)
                        info['experiments']['output'][j] = output
                    except Exception:
                        warnings.warn(MSG_ERROR_LOAD('file ' + output_file_name))
                        traceback.print_exc()
                if status != 'queued':#No need to load input of experiments that weren't even attempted to be started yet
                    try:
                        input_file_name = os.path.join(entry,FILE_EXP(j),FILE_INPUT)
                        with open(input_file_name,'rb') as fp:
                            input = deserializer.load(fp)
                        info['experiments']['input'][j] = input
                    except Exception:
                        warnings.warn(MSG_ERROR_LOAD('file ' + input_file_name))
                        traceback.print_exc()
        return info
    if ID:
        partial_id = re.compile(ID)
        entries = filter(lambda entry: partial_id.match(get_output(entry, no_objects = True).ID),entries)
    if keywords:
        entries = filter(lambda entry: _keyword_match(keywords,get_output(entry,no_objects = True).keywords),entries)
    entries = unique(entries)
    if not need_unique:
        return (get_output(entry, no_objects=no_objects) for entry in entries)
    else:
        if len(entries) == 0:
            raise ValueError(MSG_ERROR_NOMATCH)
        if len(entries) > 1:
            raise ValueError(MSG_ERROR_MULTIMATCH(entries))
        return get_output(entries[0], no_objects=no_objects)

def _is_experiment_directory(directory):
    return os.path.isfile(os.path.join(directory, FILE_INFO))

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

def _get_directory(name, path, no_date, debug, git):
    if no_date:
        basepath = os.path.join(path, name)
    else:
        date = datetime.date.today()
        basepath = os.path.join(path, date.strftime('w%Wy%y'), name)
    basepath = os.path.abspath(basepath)
    if debug:
        directory = os.path.join(basepath, FILE_DEBUG)
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        os.makedirs(directory)
        return directory, FILE_DEBUG
    for attempt in range(20):  # Try normal words, then ficticious ones, fail if cannot find unused
        ID = random_word(length = LEN_ID,dictionary = (attempt<10))
        directory = os.path.join(basepath,ID)
        try:
            os.makedirs(directory)
            if git and _git_has_tag(git,STR_GIT_TAG(ID)):
                delete_empty_directories([directory])
            else:
                return directory,ID
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    raise ValueError(MSG_ERROR_DIR)

def _git_command(string, add_input=True):
    string = 'git ' + string
    output = '$ ' + string + '\n' if add_input else ''
    args = shlex.split(string)
    output += subprocess.check_output(args, stderr=subprocess.STDOUT).decode('UTF8')
    return output

def _git_id():
    return _git_command('log --format="%H" -n 1', add_input=False).rstrip()

def _git_has_tag(path,tag):
    initial_directory = os.getcwd()
    os.chdir(path)
    try:
        out = _git_command('tag --list',add_input = False)
        return tag in out.splitlines() 
    except subprocess.CalledProcessError:
        os.chdir(initial_directory)
        raise

def _git_snapshot(path, commit_body, ID):
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
        out += _git_command('commit --allow-empty -m "{0} \n {1}"'.format(STR_GIT_COMMIT_TITLE(active_branch), commit_body))
        out += _git_command('tag {}'.format(STR_GIT_TAG(ID)))
        snap_id = _git_id()
        out += _git_command('reset HEAD~1')
    except subprocess.CalledProcessError as e:
        raise GitError(traceback.format_exc(), out + '\n' + str(e.output))
    except Exception:
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
        
        If INPUTS is provided, then FUNC is called once for each entry of INPUTS.
        For example, FUNC can be a numerical algorithm and INPUTS
        can be a list of different mesh resolutions (with the goal to assess 
        convergence rates) a list of different subroutines (with the goal to find
        the best subroutine in terms of runtime/memory/...).
        In the following, each call of FUNC is called an 'experiment'.

        Scilog creates a directory (using NAME and the current date, and optionally DIRECTORY)
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
                    unless COPY is specified, in which
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
        ''')
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
    parser.add_argument('-c', '--copy', action='store', nargs='?', const='.',
        default=None,
        help=
        '''
        Specify directory where FUNC stores its output

        If no argument is specified, FUNC will be run in a clean working directory
        and it is assumed that its outputs are stored in that working directory
        ''')
    args = parser.parse_args()
    if args.show:
        entries = load(search_pattern=args.func, no_objects=True, need_unique=False)
        entries = list(entries)
        if len(entries) != 1:
            print(STR_MULTI_ENTRIES(len(entries)))
        for entry in entries:
            print(STR_ENTRY(entry))
            if args.git:
                print(STR_GITDIFF)
                subprocess.call(['gitdiffuntracked', entry[0]['gitcommit']])
    else:
        inputs = eval(args.inputs)
        init_dict = eval(args.base)
        module_name = args.func
        python_function_s = re.compile('(\w+\.)+(\w+)')
        external = args.external or not python_function_s.match(module_name)
        if not external:
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
                except Exception:
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
            func=fn, directory=args.directory,
            inputs=inputs,
            name=args.name,
            external=external,
            analysis=analyze_fn,
            runtime_profile=args.runtime_profile,
            memory_profile=args.memory_profile,
            git=args.git,
            no_date=args.no_date,
            parallel=args.parallel,
            git_path=module_path,
            copy_output=args.copy
        )
if __name__ == '__main__':
    main()
