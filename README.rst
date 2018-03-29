scilog: Digital bookkeeping of numerical experiments for reproducible science
==========================================
:Author: Soeren Wolfers <soeren.wolfers@gmail.com>
:Organization: King Abdullah University of Science and Technology (KAUST) 

This package provides a command line tool to save information about source code, hardware, installed packages, runtime, memory usage along with the actual results of numerical experiments.

In its easiest form, :code:`scilog 'my_program'`, runs `my_program` and stores output, runtime, and system information in the directory`scilog/my_program`. 
(Don't worry. Running the same command again, creates subfolders `scilog/my_program/v0` and `scilog/my_program/v1`. Nothing ever gets overwritten.)
If the flag `--git` is used, `scilog` additionally creates a snapshot commit of the current working tree in the git branch `_scilog` of the containing git repository.

Using :code:`scilog 'my_program {}' -e range(4)`, the program is run with the inputs [0,1,2,3]. Here, `range(4)` can be replaced by any valid Python code that creates an iterable of experiment configurations.
 
For numerical experiments with Python, one can use the simple syntax `scilog my_package.my_module.my_function` instead of
`scilog 'python -m my_package.my_module.my_function'`. The flags `--memory_profile` and `--runtime_profile` can then be used store additional detailed runtime and memory usage information.

Information on previous scilog entries can be displayed using `scilog --show 'my_program'`

scilog can also be called within python, using the functions `scilog.conduct` and `scilog.load`.
---

To install run :code:`pip install scilog`

---

