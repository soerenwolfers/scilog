scilog: Digital bookkeeping of numerical experiments for reproducible science
==========================================
:Author: Soeren Wolfers <soeren.wolfers@gmail.com>
:Organization: King Abdullah University of Science and Technology (KAUST) 

This package provides a command line tool to execute numerical experiments and store results along with auxiliary information about inputs, source code, hardware, installed packages, runtime, memory usage.

In its easiest form, :code:`scilog 'foo.bar(x=1,y=2)'`, runs the function Python function `bar` in the module `foo` and stores output, input, runtime, system, and source code information. 

If the flag `--git` is used, `scilog` additionally creates a snapshot commit of the current working tree that resides outside the actual branch history.

Since most computational research requires series of experiments, :code:`scilog` supports such series through the specification of *variable ranges*. 
Using :code:`scilog 'foo.bar(x=var(1,2),y=2)'` does the same as above, but foo.bar is run once with `x=1` and once with `x=2`.
 
Scilog can also be used for numerical experiments that are not based on Python. Using :code:`scilog 'my_tool {p}' --variables p=[0,1]` the 
command line tool :code:`my_tool` is run twice, with inputs `1` and `2`, respectively. 

Information on previous scilog entries can be displayed using `scilog --show 'my_tool'` or by simple navigating scilog's directory hierarchy, where 
all entries are stored in binary form and as much as possible easily accessible text form. 

---

To install run :code:`pip install scilog`

---

