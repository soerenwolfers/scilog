scilog: record-keeping for computational experiments
=======================================================
:Author: Soeren Wolfers <soeren.wolfers@gmail.com>
:Organization: King Abdullah University of Science and Technology (KAUST) 

This package provides a command line tool that executes numerical experiments and stores results along with auxiliary information about inputs, source code, hardware, installed packages, runtime, memory usage.

Getting started
---------------

In its easiest form, ``scilog 'foo.bar(x=1,y=2)'``, runs the function Python function ``bar`` in the module ``foo``.

Since most computational research requires series of experiments, ``scilog`` supports such series through the specification of *variable ranges*. 
Using :code:`scilog 'foo.bar(x=var(1,2),y=2)'` does the same as above, but foo.bar is run once with ``x=1`` and once with ``x=2``.
 
Scilog can also be used for numerical experiments that are not based on Python. Using ``scilog --variables p=[0,1] 'my_tool {p}'``  the 
command line tool ``my_tool`` is run twice, with arguments ``1`` and ``2``, respectively. 

Information on previous scilog entries can be displayed using ``scilog --show 'my_tool'`` or by simple navigating scilog's directory hierarchy, where 
all entries are stored in the most human-readable form possible. 

Currently running experiments can be deattached from and will continue in the background.
They can be listed and reattached to with ``scilog -ls`` and ``scilog -r``, respectively. 

Installation
------------

Run ``pip install scilog``

Requirements
------------

- Python 3.6+
- Unix-like operating system
- GNU screen 
