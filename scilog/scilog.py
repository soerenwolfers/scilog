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
import pathlib
import types
import operator
import subprocess
import shlex
import json
import contextlib
import stat
import itertools
import ast
import builtins
import signal
from collections import OrderedDict
from string import Formatter

import numpy as np
from matplotlib import pyplot
from IPython.utils.capture import capture_output

from swutil import sys_info, np_tools, plots, aux
from swutil.validation import Positive, Integer, String, List, Tuple,Iterable
from swutil.logs import Log
from swutil.hpc import Locker
from swutil.aux import  string_dialog, no_context, random_word,\
    string_from_seconds,input_with_prefill,is_identifier,smart_range
from swutil.files import append_text, delete_empty_files,\
    delete_empty_directories, find_directories, path_from_keywords
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
FILE_EXP = lambda i: f'experiment{i}'
FILE_RANDOMSTATE = 'randomstate.pkl'
STR_GIT_TAG = lambda ID: f'scilog_{ID}'
STR_GIT_LOG = lambda sha1, log: f'#Created git commit {sha1} as snapshot of current state of git repository using the following commands:\n{log}'
STR_GIT_COMMIT_TITLE = lambda branch: f'Snapshot of working directory of branch {branch}'
STR_GIT_COMMIT_BODY = lambda name, ID, directory: f'Created for scilog entry {ID} in {directory}'
STR_LOADSCRIPT = ('#!/bin/sh \n '
                     + f' xterm -e {sys.executable} -i -c '
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
STR_SOURCE = lambda n, func, module,source: (('#Experiments were conducted with' if n != 1 else '#Experiment was conducted with ')
                + ('class' if inspect.isclass(func) else  
                    (f'{func.__class__.__name__}' if isinstance(func,(types.MethodType,types.FunctionType)) else 
                    f'instance of {func.__class__.__name__}'))
               + (f' called {func.__name__}' if hasattr(func, '__name__') else '')
              + f' from the module {module} whose source code is given below:\n{source}')
STR_TIME = '%y-%m-%d %H:%M:%S'
def STR_PARAMETERS_PROMPT(func,external,current_parameters,known_parameters,allow_variables,class_instance,allow_all_keys):
    if class_instance:
        why= f'to pass to instance of {func.__class__.__name__}'
    else:
        if external:
            why = f'to fill in `{external}`'
        else:
            name = _get_name(func)
            if inspect.isclass(func):
                why= f'to initialize class {name}'
            else:
                why = f'to pass to {name}'
    parameters_string = ', '.join(f'{key}={value!r}' for key,value in current_parameters.items())
    require_parameters=[key for key in known_parameters if key not in current_parameters]
    if require_parameters:
        parameters_string += (', ' if parameters_string else '') + ', '.join(f'{key}=' for key in require_parameters)
    if allow_all_keys:
        parameters_string += '[, <kwarg>=<value>]*'
    return f'>> Specify {"variables or " if allow_variables else ""}parameters {why} ({parameters_string}):\n\t'
def STR_PARAMETERS_ALLOWED(passed_keys,known_parameters):
    forbidden = [key for key in passed_keys if key not in known_parameters]
    if len(forbidden)>1:
        out = '!! Cannot specify parameters'+', '.join(f'`{key}`' for key in forbidden[:-1]) + f', and `{forbidden[-1]}`'
    else:
        out = '!! Cannot specify parameter '+f'`{forbidden[0]}`'
    return out
STR_PARAMETERS_FORMAT = '!! Input must have form `<key>=<value>[,<key>=<value>]*`\n!! Enter `help` for more information'
STR_PARAMETERS_HELP = lambda allow_variables: (
                            '?? Parameters are specified by `<key>=<value>` with <key> a Python identifier and <value> a Python expression.'
                            +(
                                (
                                '\n?? Variables have the same syntax, except <value> has the form var(<iterable>).\n'
                                '?? Variables are used to specify arguments that are varied in a specified range.\n'
                                '?? Note the difference between <key>=[0,1] and <key>=var([0,1]):\n'
                                '?? In the first case, `[0,1]` is passed at once; in the second case it is iterated over.'
                                ) if allow_variables else ''
                            )
                        )
MSG_DEBUG =  'Debug mode. Entry is not stored permanently, stdout and stderr are not captured, no git commit is created'
MSG_NOGIT = 'Could not find git repository. No snapshot commit will be created'
MSG_START_ANALYSIS = 'Updating analysis'
MSG_START_EXPERIMENT = lambda i,n_experiments,inp: (f'Running experiment {i}' + 
        (' with variable values {}{}'.format('\n\t' if '\n' in repr(inp) else '',repr(inp))
          if inp != {} else ''))
MSG_START_GIT = lambda repo:'Creating snapshot of current working tree of repository \'{}\'. Check {}'.format(repo,FILE_GITLOG)
def MSG_START_EXPERIMENTS(name,variables,parameters):
    msg = f'Will call `{name}`'
    extend=''
    new_line=False
    if parameters:
        new_line='\n' in str(parameters) 
        extend= ' with parameters {}{}'.format("\n\t" if new_line else "",parameters)
    if variables:
        s_var = 'variables' if len(variables)>1 else 'variable'
        variable_strings = [(variable[0],str(variable[1])) for variable in variables]
        newline_in_vs = any('\n' in vs[1] for vs in variable_strings)
        sep = '\n\t' if (len(variables)>1 or newline_in_vs) else ', '
        strings = [('' if sep == ', ' else  '-') +f'`{vs[0]}`'+ (f' varying in `{vs[1]}`' if not newline_in_vs else '') for vs in variable_strings]
        extend += (" \n" if new_line else " ") +(f'and {s_var}' if extend else f' with {s_var}')+(' ' if sep==', ' else sep)+sep.join(strings)
    if not extend:
        extend =' once'
    return msg + extend
MSG_START_ENTRY = lambda directory: f'Created scilog entry {directory}'
MSG_FINISH_EXPERIMENT = lambda i,n_experiments,runtime,result,external: 'Finished experiment {} in {}{}'.format(i,string_from_seconds(runtime),
    '' if ('\n' in f'{result}') else (f'. Check {os.path.join(FILE_EXP(i),FILE_EXP_OUT)}' if external else f'. Output: {result}'))
MSG_FINISH_ENTRY=lambda directory: f'Completed scilog entry {directory}'
MSG_SUCCESS = 'All experiments finished successfully'
MSG_FAIL = 'Some experiments failed'
MSG_FINISH_GIT = lambda sha1: f'Successfully created git commit {sha1}'
MSG_ERROR_NOMATCH = 'Could not find matching scilog entry'
MSG_ERROR_MULTIMATCH = lambda entries:'Multiple matching scilog entries (to iterate through all use need_unique=False):\n{}'.format('\n'.join(entries))
MSG_ERROR_LOAD = lambda name: f'Error loading {name}. Are all required modules in the Python path?'
MSG_ERROR_INSTANTIATION = lambda name:f'Could not instantiate class {name} with given parameters'
MSG_ERROR_PARALLEL = 'Error during parallel execution. Try running with `parallel=False`'
MSG_ERROR_BASH_ANALYSIS = 'Cannot analyze output in bash mode'
MSG_ERROR_GIT = lambda file:f'Error during git snapshot creation. Check {file}'
MSG_ERROR_EXPERIMENT = lambda i:f'Experiment {i} failed. Check {os.path.join(FILE_EXP(i),FILE_EXP_ERR)}'
MSG_ERROR_ANALYSIS = lambda file: f'Analysis could not be completed. Check {file}'
MSG_ERROR_DIR = 'Could not create scilog entry directory'
MSG_EXCEPTION_STORE = lambda file: f'Could not store {file}'
MSG_EXCEPTION_ANALYSIS = 'Exception during online analysis'
MSG_EXCEPTION_EXPERIMENT = lambda i: f'Exception during handling of experiment {i}. Check {FILE_ERR}'
MSG_WARN_SOURCE = 'Could not find source code'
MSG_WARN_LOADSCRIPT = 'Error during load script creation'
MSG_WARN_PARALLEL = ('Could not find pathos. This might cause problems with parallel execution.'
    + 'Install pathos via `pip install pathos`.')
MSG_WARN_MEMPROF = 'Could not find memory_profiler. Install memory_profiler via `pip install memory_profiler`.'
MSG_WARN_DILL = ('Could not find dill. Some items might not be storable. '
                  + 'Storage of numpy arrays will be slow'
                  + 'Install dill via `pip install dill`.')
MSG_INTERRUPT = f'Kill signal received. Stored {FILE_INFO}, closing now.'
LEN_ID = 8

#TODO(low,high) think about using inspect.formatargspec(inspect.getargspec(func)) to directly parse args and kwargs of user input even without named argument
#TODO(med,high) understand ellipses in variable input: Do this at the string input level, so 2**3,...,2**6 can be understood. 
#TODO(med,low) make scilog --show work for all scilog entries in current git repo even outside of cwd
#TODO(high,low) make scilog --show smarter: if FUNC doesn't match any scilog entry path, try if it matches a scilog entry ID
#TODO(high,high) remove analysis functionality from scilog.py and add scilog --analyze working as follows: provide a list of entry identifiers (ID or paths) as well as a function that accepts scilog entries (i.e. the summary dicts). the source code file (foo.py) of that function (foo.func) is copied in the analysis subsubdirectories `foo_x` of each scilog entry  
#TODO technically: `scilog --analyze X3DH,UH1X --parameters [...] --variables [...] foo.func` starts a scilog run with arguments func=foo.func, analysis = True[effects that git=False, base_directory =tempfile.mktemp(), func is  called  with parameters={**parameters, entries=[scilog.load(identifier) for identifier in analzsis]} log does not say 'created scilog entry' but instead says which entries will be analyzed with what, and finishes with "added analysis <classification_x to X3DH and <classification>_y to UH1X, and entry is copied into subdireoctry analyze/<classificatoin>_x of X3DH and UH1X with x possibly being adapted to what is already in the analysis of X3DH and UH1X ]
#TODO make load ignore subdirectories of scilog entries (to avoid listing analysis entries)
#TODO(?,?) comunicate to plots.save 
#TODO(low,med) understand <param>=<string> without quotes around <string> (simple and stupid: fail, add unrecognized variable to locals, repeat...)
#TODO(low,med) understand scilog foo(a=1)(b=var()) by defining foo in locals() and have it return another function that takes yet more arguments 
#TODO(med,low) if copy_output is a path, try to copy that path and only terminate when succeeded (e.g. when it starts existing) also, add argument check_done and if it is provided only try copying as soon as it returns True
#TODO(low,low) store completion flag
#TODO(high,low) make scilog --show show [log, current-stdout,current-stderr] if entry not completed, (so you can avoid screen -r and navigation to the filesystem directory)
#TODO(low,low) make scilog --show show scilog-stderr if it exists and, if all experiments failed, also show current-stderr of last experiment in that case (if at least one succeeded leave it to user to navigate to the failed experiment) 
#TODO(med,low) make scilog --show default to no-git (add --git)
#TODO(?,?) make scilog --show first look through screen sessions
#TODO(low,med) Store final note in notes.txt, add --update <REASON> to scilog which then flags all previous runs with same config as outdated in their notes.txt  
#TODO(low,low) add --note flag
#TODO(med,low) extend scilog ls output by scilog status (running, terminated, error,n_experiments) (store log files)
#TODO(med,low) include no-shutdown script
def record(func, variables=None, name=None, base_directory=None, aux_data=None,
            analysis=None, runtime_profile=False, memory_profile=False,
            git=True, no_date=False, parallel=False,
            copy_output = None, parameters=None,debug = None,classification= None,dry_run=False):
    '''
    Call :code:`func` once or multiple times and store results along with auxiliary information
    about runtime and memory usage, installed modules, source code, hardware, etc.
    
    code:`func` is called once for each combination of variable values as 
    specified by the variable ranges in :code:`variables`.
    For example, :code:`func` can be a numerical algorithm and :code:`variables`
    can be used to specify different mesh resolutions as follow
        `variables = {h:[2**(-l) for l in range(10)}` 
    with the goal to assess the rate of convergence.
    Another example would be to specify a list of subroutines with the goal to find
    the best subroutine in terms of runtime/memory consumption/....

    In the following, each call of :code:`func` is called an 'experiment'.

    Scilog creates a directory -- specified by :code:`directory`, :code:`name`, 
    and optional parameters or a randomly generated ID -- with the following content:
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
                *(optional)memory usage
            *(optional)parameters: Parameters that are equal for all experiments
            *(optional)git_commit: SHA1 of git commit 
            *(optional)aux_data: Argument :code:`aux_data`
        *log.txt
        *(optional)err.txt
        *(optional)git.txt: stdout of git snapshot creation 
        *source.txt: Source code of the module containing :code:`func`
        *For each experiment a subdirectory 'experiment<i>' with:
            *output.pkl: Output of :code:`func`
            *(optional)input.pkl: Argument passed to :code:`func`
            *(optional)working_directory/: Working directory for call of :code:`func`, 
                unless parameter :code:`copy_output` is specified
            *(optional)stderr.txt:
            *(optional)stdout.txt:
            *(optional)runtime_profile.txt: Extensive runtime information for each experiment
            *(optional)memory_profile.txt: Memory usage information for each experiment
        *(optional) analysis/: output of function :code:`analysis`
            *(optional)stderr.txt
            *(optional)stdout.txt
            *(optional)working_directory/: Working directory for call of :code:`analysis`

    To load a scilog entry, use the function :code:`scilog.load`.
    This function loads summary.txt and replaces the string representations of outputs and inputs
    by the actual Python objects. 

    :param func: Function to be called with different experiment configurations
    :type func: function
    :param variables: Arguments for call of :code:`func` that are varied. 
        If not specified, then :code:`func` is called once, without arguments
    :type variables: List(-like) if single variable or dictionary of lists
    :param name: Name of scilog entry. 
        If not specified, :code:`func.__name__` is used
    :type name: String
    :param base_directory: Root directory for storage
    :type base_directory: String
    :param aux_data: Auxiliary data that should be stored along with the results
    :type aux_data: Any
    :param analysis: Function that is called after each experiment 
        Can be used, e.g., for plotting
    :param runtime_profile: Store extensive runtime information
        May slow down execution
    :type runtime_profile: Boolean
    :param memory_profile: Track memory usage
        May slow down execution
    :type memory_profile: Boolean
    :param git: Create git snapshot commit
        The resulting commit is tagged with the entry ID and resides outside the branch history
        The repository path may be specified, else it will be automatically detected
        Add 'scilog' to your .gitignore to avoid storing the scilog entries in each snapshot.  
        (Should you ever want get rid of the snapshots, 
        use `git tag --list 'scilog_*'|xargs -I % git tag -d %` to remove all scilog commits or 
        use function `clean_git_repository` to remove all scilog commits whose scilog entry does not reside in repository anymore)
    :type git: Boolean or String
    :param no_date: Do not store outputs in sub-directories grouped by calendar week
    :type no_date: Boolean
    :param parameters: Parameters that are equal for all experiments
        If :code:`func` is a class, these are used to instantiate this class
    :type parameters: Dictionary
    :param debug: Force debug mode (otherwise detected automatically)
    :param debug: Boolean
    :param copy_output:  The contents of this directory will be copied into the scilog entry directory
    :type copy_output: String
    :param classification: Short, human readable description of entry
    :type classification: String
    :param dry_run: Only setup directory and experiments, don't execute anything
    :type dry_run: Boolean
    :return: Path of scilog entry
    :rtype: String
    '''
    ########################### FIX ARGUMENTS ###########################
    variables,parameters,func_initialized,classification_t = _setup_experiments(variables,parameters,func)
    classification = classification or classification_t
    name = name or _get_name(func)
    if dry_run:
        return variables,parameters,classification,name
    external = _external(func)
    debug = debug if debug is not None else aux.isdebugging()
    if debug: 
        git = False
    log_nogit = False
    if git is not False:
        if git is True:
            git = _get_func_directory(func)
        if not _has_git(git):
            log_nogit = True
            git = False
    ########################### SETUP INPUTS ##############################
    if len(variables)!=1:#Will result in infinite loop if one variable is infinite. 
        t = itertools.product(*[variable[1] for variable in variables])
    else:
        t = ([x] for x in variables[0][1])
    inputs = ({variable[0]:tt[i] for i,variable in enumerate(variables)} for tt in t)
    try:
        n_experiments = int(np.prod([len(variable[1]) for variable in variables]))
    except TypeError:
        n_experiments = None
    ########################### CREATE SCILOG ENTRY ########################
    entry_directory,ID = _get_directory(base_directory,func,name,no_date,debug,git,classification)
    log_file = os.path.join(entry_directory, FILE_LOG)
    err_file = os.path.join(entry_directory, FILE_ERR)
    info_file = os.path.join(entry_directory, FILE_INFO)
    load_file = os.path.join(entry_directory, FILE_LOAD)
    aux_data_file = os.path.join(entry_directory, FILE_AUX)
    source_file_name = os.path.join(entry_directory, FILE_SOURCE)
    git_file = os.path.join(entry_directory, FILE_GITLOG)
    locker = Locker()
    _log = Log(write_filter=True, print_filter=True, file_name=log_file,lock = locker.get_lock())  # Logging strategy: 1) Redirect out and err of user functions (analysis and experiment) to their own files
    _err = Log(write_filter=True, print_filter=False, file_name=err_file,lock = locker.get_lock())  # 2) Log errors outside user functions in _err 3) Log everything (user-err and _err, as well as other info) in _log 
    _log.log(MSG_START_ENTRY(entry_directory))
    if log_nogit:
        _log.log(group = GRP_WARN,message = MSG_NOGIT)
    if debug:
        _log.log(group =GRP_WARN,message = MSG_DEBUG)
    info = {
        'parameters' : {key:repr(parameters[key]) for key in parameters},
        'variables' : [repr(variable) for variable in variables],
        'name' : name,
        'ID' : ID,
        'time' : datetime.datetime.now().strftime(STR_TIME),
        'func' : external or repr(func),
        'parallel' : parallel,
        'hardware' : sys_info.hardware(),
        'gitcommit' : None,
        'modules' : None,
        'note': None,
        'experiments' : {
            'runtime':[],
            'memory':[],
            'status':[],
            'input':[],
            'output':[]     
        }
    }
    if not external:
        info['modules'] = sys_info.modules()
        try:
            source = STR_SOURCE(n_experiments,func,sys.modules[func.__module__].__file__, ''.join(inspect.getsourcelines(sys.modules[func.__module__])[0]))
            append_text(source_file_name, source)
        except Exception:  # TypeError only?
            _err.log(traceback.format_exc())
            _log.log(group=GRP_WARN, message=MSG_WARN_SOURCE)
    if memory_profile is not False:
        if memory_profile == 'detail':
            try:
                import memory_profiler  # @UnusedImport, just to check if this will be possible in _run_single_experiment
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
                snapshot_id, git_log, _ = _git_snapshot(path=git,commit_body=STR_GIT_COMMIT_BODY(name, ID, entry_directory), ID=ID)
            append_text(git_file, STR_GIT_LOG(snapshot_id, git_log))
            _log.log(message=MSG_FINISH_GIT(snapshot_id))
            info['gitcommit'] = snapshot_id
        except GitError as e:
            _log.log(group=GRP_ERROR, message=MSG_ERROR_GIT(git_file))
            _err.log(message=str(e)+'\n'+c.stderr)
            append_text(git_file, e.git_log)
            raise
    try:
        import dill
        serializer = dill
    except ImportError:
        serializer = pickle
        _log.log(group=GRP_WARN, message=MSG_WARN_DILL)
    _try_store(aux_data,serializer,aux_data_file,_log,_err)
    def _update_info(i, runtime, status, memory, input_str, output_str):
        for (key,val) in [('runtime',runtime),
            ('memory',memory if memory_profile is not False else None),
            ('status',status),
            ('input',input_str),
            ('output',output_str),
            ]:
            info['experiments'][key].append(val)
        store_info()
    def store_info():
        with open(info_file,'w') as fp:
            json.dump(info,fp,indent = 1,separators = (',\n', ': '))
    store_info()
    old_wd = os.getcwd()
    ########################### RUN EXPERIMENTS ###############################
    args = (
        (
            i, input, entry_directory, func_initialized, memory_profile,
            runtime_profile, _log,_err,
            'pickle' if serializer == pickle else 'dill',
            external, debug, copy_output,n_experiments
        )
        for i, input in enumerate(inputs)
    )
    _log.log(message=MSG_START_EXPERIMENTS(name,variables,parameters))
    def close_entry():
        try:
            os.chdir(old_wd)
        except Exception:
            pass
        success = all(s=='finished' for s in info['experiments']['status'])
        try:
            _log.log(MSG_FINISH_ENTRY(entry_directory))
            if not success:
                _log.log(MSG_FAIL)
        except Exception:
            pass
        if not debug:
            note = input('You may add a short note to this entry or simply press Enter to exit:')        
            if note:
                info['note'] = note
                store_info()
        return entry_directory
    if parallel and not debug:
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
        except pickle.PicklingError:  # @UndefinedVariable
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_ERROR, message=MSG_ERROR_PARALLEL)
            raise
        for output in outputs:
            _update_info(*output)
        pool.close()
        pool.join()
    else:
        for arg in args:
            try:
                output = _run_single_experiment(arg)
            except Exception:#These come from errors in the code of _run_single_experiments. The user function errors are caught within there
                _err.log(message=traceback.format_exc())
                _log.log(group=GRP_ERROR, message=MSG_EXCEPTION_EXPERIMENT(arg[0]))
            else:
                _update_info(*output)
            if analysis:
                try:
                    _log.log(message=MSG_START_ANALYSIS)
                except BrokenPipeError:#locks raise BrokenPipeError when experiments are terminated using <C-c>
                    sys.exit(1)
                try:
                    with capture_output():
                        entry = load(path=entry_directory, need_unique=True, no_objects=False)
                    analyze(func=analysis, entry=entry, _log=_log, _err=_err, debug=debug)
                except Exception:
                    _err.log(message=traceback.format_exc())
                    _log.log(group=GRP_ERROR, message=MSG_EXCEPTION_ANALYSIS)
    return close_entry()

def _has_git(git):
    cwd = os.getcwd()
    try:
        os.chdir(git)
        #subprocess.check_call(['git','status'],stdout = subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.check_call(['git','rev-parse','HEAD',],stdout = subprocess.PIPE,stderr=subprocess.PIPE)#Sometimes git status works but rev-parse, which is used later, fails; e.g. on repos without initial commit
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        os.chdir(cwd)

def _external(func):
    return func if isinstance(func,str) else False

def _get_func_directory(func):
    return os.getcwd() if _external(func) else os.path.dirname(sys.modules[func.__module__].__file__) 

def _get_base_directory(directory,func,name,no_date):
    directory = directory or _get_func_directory(func)
    directory = os.path.join(directory,'scilog')
    if no_date:
        basepath = os.path.join(directory, name)
    else:
        date = datetime.date.today()
        basepath = os.path.join(directory, date.strftime('w%Wy%y'), name)
    return os.path.abspath(basepath)

def _get_name(func):
    if _external(func): 
        nowhite = re.compile('\S*')
        path = nowhite.match(func).group(0)
        name = os.path.basename(path)
    else:
        try:#func is a function or a class
            name = func.__name__
        except AttributeError:#func is an object with __call__ method
            name = func.__class__.__name__
    return name

def _evaluator(what,locals_dict = None):
    locals_dict = locals_dict or {}
    return eval(f'(lambda **kwargs: kwargs)({what})',{'range':range,'count':itertools.count,'np':np,'__builtins__':{}},locals_dict)

class _var:
    def __init__(self,*obj):
        if len(obj)>1:#e.g. var('ab','cd','ef')
            self.obj = obj
        elif len(obj)==1:#e.g. var(range(3))
            if Iterable.valid(obj[0]):
                self.obj = list(obj[0])#turn into list so that numpy arrays go through later on (if you leave them as arrays, they will make problems in = comparison, for example)
            else:#Allows for --variables p=3 instead of --variables p=[3]
                self.obj = [obj[0]]
        elif len(obj)==0:
            raise ValueError()
    def __repr__(self):
        return 'var('+repr(self.obj)+')'

def _setup_experiments(variables,parameters,func):
    '''
    Note: input and output `variables` have iterator type, not _var. 
    _var only occurs in the processing 
    '''
    external = _external(func)
    def _get_kwargs(func,external,variables,parameters,class_instance=False):
        if inspect.isclass(func):
            allow_variables =False
            variables = None
        else:
            allow_variables=True
        parameters_passed = parameters is not None
        variables_passed = variables is not None
        if parameters is None:
            parameters = {}
        if variables is None:
            variables = {}
        if external:
            field_names = [fname for _, fname, _, _ in Formatter().parse(external) if fname is not None]
            new_var_n = 0
            new_names = []
            for i,fname in enumerate(field_names):
                if fname == '':
                    while True:
                        if f'arg{new_var_n}' not in field_names:
                            new_names.append(f'arg{new_var_n}')
                            break
                        else:
                            new_var_n+=1
            external = external.format(
                *[f'{{{new_name}}}' for new_name in new_names],
                **{fname:f'{{{fname}}}' for fname in field_names if fname !='' }
            )
            known_parameters = OrderedDict((fname,inspect._empty) for _, fname, _, _ in Formatter().parse(external) if fname is not None)
            if len(known_parameters)==1 and list(known_parameters.keys())[0] == None:
                known_parameters = []
            allow_all_keys = False
            default_parameters = {}
        else:
            func_parameters = inspect.signature(func).parameters
            default_parameters = {
                key:value.default for key,value in func_parameters.items() 
                if (value.default != inspect._empty)
            }
            allow_all_keys = any(value.kind ==4 for key,value in func_parameters.items())
            known_parameters = OrderedDict(
                (key,value.default) for key,value in func_parameters.items()
                if (value.kind not in [2,4])
            )
        kwargs=default_parameters.copy()
        free_keys=lambda : allow_all_keys or any(key not in variables and key not in parameters for key in known_parameters)
        required_keys=lambda : [key for key in known_parameters if key not in kwargs]
        is_allowed_key=lambda key: allow_all_keys or key in known_parameters
        if variables:
            if not isinstance(variables,dict):#allow for single range instead of var dictionary
                non_default_parameters = [key for key in known_parameters if key not in default_parameters]
                if len(non_default_parameters) == 1:
                    variables={non_default_parameters[0]:variables}
                elif len(known_parameters) ==1:
                    variables = {list(known_parameters.keys())[0]:variables}
                else:
                    raise ValueError(f'Must specify name for variable {variables}')
            if any(not is_allowed_key(key) or not is_identifier(key) for key in variables):
                raise ValueError('Invalid variable names for function {}: {}'.format(external or func,{key for key in variables if not is_allowed_key(key)}))
            variables_update = {key:_var(value) for key,value in variables.items()}
        else:
            variables_update = {}
        if parameters:
            if any(key in variables_update for key in parameters):
                raise ValueError('Parameter names already defined as variables: {}'.format({key for key in parameters if key in kwargs}))
            if any(not is_allowed_key(key) or not is_identifier(key) for key in parameters):
                raise ValueError('Invalid parameter names for function {}: {}'.format(external or func,{key for key in parameters if not is_allowed_key(key)}))
            parameters_update = parameters
        else:
            parameters_update = parameters
        kwargs.update(**variables_update,**parameters_update)
        if (((not parameters_passed and not class_instance) or (not variables_passed and allow_variables)) and free_keys()) or required_keys(): 
            while True:
                prefill=', '.join([key+'=' for key in required_keys()])
                parameters_string = input_with_prefill(STR_PARAMETERS_PROMPT(func,external,kwargs,known_parameters,allow_variables,class_instance,allow_all_keys),prefill)
                if parameters_string in ['?','help','--help','??']:
                    print(STR_PARAMETERS_HELP(allow_variables))
                    continue
                try:
                    update_kwargs = _evaluator(parameters_string,{'var':_var} if allow_variables else {})
                except Exception:#(ValueError,SyntaxError):
                    if '=help' in parameters_string:
                        print(STR_PARAMETERS_HELP(allow_variables))
                    else:
                        print(STR_PARAMETERS_FORMAT)
                else:
                    kwargs.update({key: value for key,value in update_kwargs.items() if is_allowed_key(key)})
                    done = True
                    if not all(key in kwargs for key in known_parameters):
                        if parameters_string =='':
                            print(STR_PARAMETERS_FORMAT)
                        done = False
                    if any(not is_allowed_key(key) for key in update_kwargs):
                        print(STR_PARAMETERS_ALLOWED(update_kwargs,known_parameters))
                        done = False
                    if done:
                        break
        return kwargs,external,default_parameters,known_parameters
    if external:
        def func(**kwargs):
            subprocess.check_call(external.format(**kwargs), stdout=sys.stdout, stderr=sys.stderr, shell=True)
    classification_variables = {}#variables.copy() if isinstance(variables,dict) else {}#can be None or a single unnamed iterable whose name will be found out only later 
    classification_parameters = {}#parameters.copy() if isinstance(parameters,dict) else {}# can be None
    if inspect.isclass(func):# in this case, parameters are for initialization and variables for function call
        parameters,_,default_parameters,_ = _get_kwargs(func,False,None,parameters)
        func_initialized=func(**parameters)
        variables,_,default_parameters_2,known_parameters_2 = _get_kwargs(func_initialized,False,variables,None,class_instance=True)
        real_variables = {key:value for key,value in variables.items() if isinstance(value,_var)}
        classification_parameters.update({key:value for key,value in parameters.items() if key not in default_parameters or (key in default_parameters and value !=default_parameters[key])})
        if len(variables)<=1:#nothing possibly interesting can be said if there is only one variable except if variable was not known (i.e. keyword argument) 
            if not classification_parameters:#If not any classification yet take what you have
                classification_variables.update({key:value for key,value in variables.items()})
            else:
                classification_variables.update({key:value for key,value in variables.items() if key not in known_parameters_2})
        else:
            classification_variables.update({key:value for key,value in variables.items() if key not in known_parameters_2 or (key in default_parameters_2 and value!=default_parameters_2[key])})
            if any(key not in real_variables for key in variables if not key in default_parameters_2):#Not all nondefault parameters actually vary, so list those that do
                classification_variables.update({key:value for key,value in real_variables.items() if key not in default_parameters_2})
        variables = {key:(value.obj if isinstance(value,_var) else [value]) for key,value in variables.items()}#Users are prompeted vor variables or parameters, but if they enter parameters, i.e. a single value, the willy still be handled as variables taking only one value
    else:
        kwargs,external,default_parameters,_ =_get_kwargs(func,external,variables,parameters)#use all as name, first params as usual (all hand selected, fill to 5), then __var_l_h
        variables = {key:value.obj for key,value in kwargs.items() if isinstance(value,_var)}
        parameters ={key:value for key,value in kwargs.items() if not isinstance(value,_var)}
        #use classification even if only one known parameter, this helps if the braces in a bash command string are changed and suddenly control something very different 
        classification_parameters.update({key:value for key,value in parameters.items() if key not in default_parameters or (key in default_parameters and value!=default_parameters[key])})
        classification_variables.update({key:value for key,value in variables.items() if key not in default_parameters or (key in default_parameters and value!=default_parameters[key])})
        def func_initialized(**experiment):
            return func(**experiment,**parameters)
    variables = list(variables.items())
    classification_p = path_from_keywords(classification_parameters,into='file')
    classification_v = '_'.join(s.replace('_','') for s in classification_variables.keys())
    classification = classification_p+('+' if classification_v else '') +classification_v 
    for j,variable in enumerate(variables):
        if (List|Tuple).valid(variable[1]) and Ellipsis in variable[1]:
            variables[j] = (variable[0],smart_range(*[e for e in variable[1] if e != Ellipsis]))
    return variables,parameters,func_initialized,classification

def _run_single_experiment(arg):
    (i, input, directory, func, memory_profile,
     runtime_profile, _log,_err, serializer,
     external, debug, copy_output,n_experiments) = arg
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
    _log.log(MSG_START_EXPERIMENT(i,n_experiments,input))
    runtime = None
    output = None
    input_str = repr(input)
    memory = None
    status = 'failed'
    randomstate = None
    if not external:
        randomstate = np.random.get_state()
    if hasattr(func, '__name__'):#func is function
        temp_func = func
    else:#func is object
        temp_func = func.__call__
    if copy_output is None:
        os.makedirs(experiment_working_directory)
        os.chdir(experiment_working_directory)
    else:
        os.makedirs(experiment_directory)
    try:
        _try_store(input,serializer,input_file,_log,_err)
        _try_store(randomstate,serializer,randomstate_file,_log,_err)
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
                            output = temp_func(**input)
                        except Exception:
                            status = 'failed'
                            if debug:
                                traceback.print_exc()
                                try:
                                    import ipdb as debugger
                                except ModuleNotFoundError:
                                    import pdb as debugger
                                debugger.post_mortem(sys.exc_info()[2])
                            stderr_append = traceback.format_exc()
                        else:
                            status = 'finished'
        runtime = timeit.default_timer() - tic
        delete_empty_files([stderr_file, stdout_file])
        if status == 'failed':
            append_text(stderr_file, stderr_append)
            _log.log(group=GRP_ERROR, message=MSG_ERROR_EXPERIMENT(i),use_lock = False)#locks are often broken already which leads to ugly printouts, also errors don't matter at this point anyway
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
    _try_store(output,serializer,output_file,_log,_err)
    del output
    if status == 'finished':
        _log.log(MSG_FINISH_EXPERIMENT(i, n_experiments, runtime, output_str,external))
    gc.collect()
    return (i, runtime, status, memory, input_str, output_str)

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
        single_reference = (self.reference == 'self')
        ind_finished = [j for (j, s) in enumerate(experiments['status']) if s == 'finished']
        if len(ind_finished) > 2 + (self.extrapolate if self.extrapolate >= 0 else 0):
            if self.work is None:
                times = experiments['runtime'][ind_finished]
            else:
                times = [self.work(i) for i in ind_finished]
            results = experiments[i]['output'][ind_finished]
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
                    
def _try_store(what,serializer,file,_log,_err):
    if what is not None:
        try:
            with open(file, 'wb') as fp:
                serializer.dump(what, fp)
        except (TypeError, pickle.PicklingError):
            _err.log(message=traceback.format_exc())
            _log.log(group=GRP_WARN, message=MSG_EXCEPTION_STORE(os.path.split(file)[-1]))

def clean_git_repository(directory=None,dry_run = True):
    '''
    Delete all commits in repository specified by :code:`directory` which do not have matching
    scilog entry in repository directory. 
    :param directory: Path that is under version control
    :type directory: String
    :param dry_run: Actually go ahead and delete commits, else just list them
    :type dry_run: Bool
    '''
    directory = directory or os.getcwd()
    os.chdir(directory)
    scilog_tag = re.compile('scilog_.*')
    tags = [tag for tag in _git_command('tag --list',add_input = False).splitlines() if scilog_tag.match(tag)]
    git_directory = _git_command('rev-parse --show-toplevel', add_input=False).rstrip()
    os.chdir(git_directory)
    entries = load(need_unique=False,no_objects=True)
    IDs = [entry['ID'] for entry in entries]
    unmatched = [tag for tag in tags if tag[7:] not in IDs]
    if unmatched:
        print(f'The following scilog git commits have no matching scilog entry in {directory}:')
        [print(tag) for tag in unmatched]
        if dry_run:
            print('Specify `dry_run=False` to remove unmatched commits')
        else:
            print('Removing unmatched commits...',end='')
            [_git_command(f'tag -d {tag}') for tag in unmatched]
            print('done')
    else:
        print(f'All scilog git commits have matching scilog entries in {directory}')

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
            _try_store(output,serializer,output_file,_log,_err)
            os.chdir(cwd)
            analysis_directory = os.path.join(entry['path'], FILE_ANALYSIS)
            shutil.rmtree(analysis_directory, ignore_errors=True)
            shutil.move(analysis_directory_tmp, entry['path'])
            shutil.rmtree(os.path.split(analysis_directory_tmp)[0], ignore_errors=True)
    except Exception:
        os.chdir(cwd)
        
class RE:
    def __init__(self,s):
        self.s=s

def load(search_pattern='*', path='', ID=None, no_objects=False, need_unique=True,include_modules=False,parameters=None,fix_broken_summary_txt=False):#TODO remove fix_borken_summary_txt
    '''
    Load scilog entry/entries.
   
    :param search_pattern: Shell-style glob/search pattern using wildcards
        If there are multiple entries of the same name (those are stored as
        <name>/v0 <name>/v1 ... in the filesystem) and they should all be returned, 
        use `search_pattern=<name>/v*` and `need_unique=False`
    :type search_pattern: String, e.g. search_pattern='foo*' matches `foobar`
    :param path: Path of exact location if known (possibly only partially), relative or absolute
    :type path: String, e.g. '/home/username/<project>' or '<project>'
    :param no_objects: To save time, only load information about scilog entry, not results
    :type no_objects: Boolean
    :param need_unique: Require unique identification of scilog entry.
    :type need_unique: Boolean
    :param parameters: Search pattern that is applied to the scilog parameters
    :type parameters: Dictionary of regular expression strings or objects (which will be converted to strings)
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
    if os.sep in search_pattern and path == '':#Use absolute path part of search pattern as path, if existent
        temp_path, temp_search_pattern = search_pattern.rsplit(os.sep, 1)
        if os.path.isabs(temp_path):
            path, search_pattern = temp_path, temp_search_pattern
    if search_pattern[-1]!='*':
        search_pattern = search_pattern+'*'
    entries = []
    entries.extend(find_directories(search_pattern, path=path))
    entries.extend(find_directories('*/' + search_pattern, path=path))
    entries = [entry for entry in entries if _is_experiment_directory(entry)]
    def get_output(entry, no_objects):
        file_name = os.path.join(entry, FILE_INFO)
        with open(file_name, 'r') as fp:
            try:
                info = json.load(fp)
            except Exception:
                raise ValueError(f'Problem with {file_name}') 
        info['path'] = entry
        if not include_modules:
            del info['modules']
        if not no_objects:
            #if isinstance(info['experiments'],dict):#Old version of scilog:
            #    DL = info['experiments']
            #    info['experiments'] = [dict(zip(DL,t)) for t in zip(*DL.values())]
            for j,s in enumerate(info['experiments']['status'] if not fix_broken_summary_txt else ('finished' for i in itertools.count())):
                if s == 'finished':
                    try:
                        output_file_name = os.path.join(entry, FILE_EXP(j), FILE_OUTPUT)
                        with open(output_file_name, 'rb') as fp:
                            output = deserializer.load(fp)
                        if fix_broken_summary_txt:
                            info['experiments']['output'].append(output)
                        else:
                            info['experiments']['output'][j] = output
                    except Exception:
                        warnings.warn(MSG_ERROR_LOAD('file ' + output_file_name))
                        if fix_broken_summary_txt:
                            break
                        traceback.print_exc()
                if not fix_broken_summary_txt: 
                    try:
                        input_file_name = os.path.join(entry,FILE_EXP(j),FILE_INPUT)
                        with open(input_file_name,'rb') as fp:
                            input = deserializer.load(fp)
                        info['experiments']['input'][j] = input
                    except Exception:
                        warnings.warn(MSG_ERROR_LOAD('file ' + input_file_name))
                        traceback.print_exc()
            for key in info['experiments']:
                try:
                    info['experiments'][key] = np.array(info['experiments'][key])
                except Exception:
                    pass
        return info
    if ID:
        partial_id = re.compile(ID)
        entries = [entry for entry in entries if partial_id.match(get_output(entry, no_objects = True)['ID'])]
    if parameters:
        parameters = {key:(repr(value) if not isinstance(value,RE) else value) for (key,value) in parameters.items()}
        def matches_parameters(entry):
            out = get_output(entry,no_objects=True)
            if not 'parameters' in out:
                return False
            else:
                test = out['parameters']
                for key,value in parameters.items():
                    if key not in test:
                        return False
                    if isinstance(value,RE):
                        if not re.match(value.s,test[key]):
                            return False
                    else:
                        if not value == test[key]:
                            return False 
                return True
        entries = [entry for entry in entries if matches_parameters(entry)]
    if len(entries)>1 and need_unique:
        basenames = [os.path.basename(get_output(entry,no_objects=True)['path']).rsplit('_',1) for entry in entries]
        if len(set(bn[0] for bn in basenames))==1:
            entries = [max(entries,key = lambda entry: get_output(entry,no_objects=True)['time'])]
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
    
def _get_directory(directory,func,name,no_date, debug, git, classification):
    basepath = _get_base_directory(directory,func,name,no_date)
    if debug:
        directory = os.path.join(basepath, FILE_DEBUG)
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass
        os.makedirs(directory)
        return directory, FILE_DEBUG
    try_classification_based_directory = 1 if classification else 0
    for attempt in range(20):  # Try keyword format, then random words, fail if cannot find unused
        ID = random_word(length = LEN_ID,dictionary = (attempt<10))
        if try_classification_based_directory:
            directory = os.path.join(basepath,classification+f'_{try_classification_based_directory-1}')
        else:
            directory = os.path.join(basepath,ID)
        try:
            os.makedirs(directory)
            if git and _git_has_tag(git,STR_GIT_TAG(ID)):
                delete_empty_directories([directory])
            else:
                return directory,ID
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                if try_classification_based_directory:
                    try_classification_based_directory  += 1
            else:
                if try_classification_based_directory:#There was a problem creating a directory with the keyword format
                    try_classification_based_directory = 0#Maybe illegal characters in parameters, try it with random words
                else:#Already tried random words, something else is wrong
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
