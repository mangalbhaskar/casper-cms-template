# Python In Nutshell

## Basics
* http://www.astro.ufl.edu/~warner/prog/python.html

### Variables
* Variables in Python follow the standard nomenclature of an alphanumeric name beginning in a letter or underscore.
* Variable names are case sensitive.
* Variables do not need to be declared and their datatypes are inferred from the assignment statement.
* Variable Scope: Most variables in Python are local in scope to their own function or class
* Global Variables: Global variables, however, can be declared with the global keyword.

### Datatypes
Python supports the following data types:
* boolean (True/False)
* integer
* long
* float
* string
* list
* object
* None
* tuple
* dictionary

```python
bool = True
name = "Craig"
age = 26
pi = 3.14159
print(name + ' is ' + str(age) + ' years old.')

# Scope
a = 1
b = 2
def Sum():
   global a, b
   b = a + b
Sum()
print(b)
```

### Statements and Expressions
* print

```python
print "Hello World"
print('Print works with or without parenthesis')
print("and single or double quotes")
print("Newlines can be escaped like\nthis.")
print("This text will be printed"),
print("on one line becaue of the comma.")
```
* input
* raw_input
* The assignment statement: Assigns a value to a variable.
* import

```python
name = raw_input("Enter your name: ")
a = int(raw_input("Enter a number: "))
print(name + "'s number is " + str(a))
a = b = 5
a = a + 4
print a,b

a = b = 5 #The assignment statement
b += 1 #post-increment
c = "test"
import os,math #Import the os and math modules
from math import * #Imports all functions from the math module
```

### Operators

* Arithmatic: `+, -, *, /, and % (modulus), // (modulus)`
* Comparison: `==, !=, <, >, <=, >=`
* Logical: `and, or, not`
* Exponentiation: `**`
* Execution: `os.system('ls -l') #Requires import os`

### Maths
* Maths: Requires `import math`
* Absolute Value: `a = abs(-7.5)`
* Arc sine: `x = asin(0.5) #returns in rads`
* Ceil (round up): `print(ceil(4.2))`
* Cosine: `a = cos(x) #x in rads`
* Degrees: `a = degrees(asin(0.5)) #a=30`
* Exp: `y = exp(x) #y=e^x`
	- https://stackoverflow.com/questions/31951980/what-exactly-does-numpy-exp-do
	- The exponential function is e^x where e is a mathematical constant called Euler's number, approximately 2.718281. This value has a close mathematical relationship with pi and the slope of the curve e^x is equal to its value at every point. np.exp() calculates e^x for each value of x in your input array.
* Floor (round down): `a = floor(a+0.5)`
* Log: `x = log(y); #Natural Log`
   `x = log(y,5); #Base-5 log`
* Log Base 10: `x = log10(y)`
* Max: `mx = max(1, 7, 3, 4) #7`
   `mx = max(arr) #max value in array`
* Min: `mn = min(3, 0, -1, x) #min value`
* Powers: `x = pow(y,3) #x=y^3`
* Radians: `a = cos(radians(60)) #a=0.5`
* Random #: Random number functions require import random
   `random.seed() #Set the seed based on the system time.`
   `x = random() #Random number in the range [0.0, 1.0)`
   `y = randint(a,b) #Random integer in the range [a, b]`
* Round: `print round(3.793,1; #3.8 - rounded to 1 decimal`
   `a = round(3.793,0) #a=4.0`
* Sine: `a = sin(1.57) #in rads`
* Square Root: `x = sqrt(10) #3.16...`
* Tangent: `print tan(3.14)# #in rads`

### Strings

### Arrays
* Arrays in basic Python are actually lists that can contain mixed datatypes
* Array Indices begin at 0, like other Python sequences (and C/C++). In contrast, in Fortran or Matlab, indices begin at 1.
* Creating lists
	* A list can be created by defining it with [].
	* A numbered list can also be created with the range function which takes start and stop values and an increment.
```python
list = [2, 4, 7, 9]
list2 = [3, "test", True, 7.4]
a = range(5) #a = [0,1,2,3,4]
a = range(10,0,-2) #a = [10,8,6,4,2]
b = range(5,-5,-1) #b = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4]
```
* An empty list can be initialized with [] and then the append command can be used to append data to the end of the list:
```python
a=[]
a.append("test")
a.append(5)
print a
```
* Finally, if you want a list to have a predetermined size, you can create a list and fill it with None's:
```python
length=7
a=[None]*length
a[5] = "Fifth"
a[3] = 6
print len(a)
```
* Removing from lists: The pop method can be used to remove any item from the list
* Creating arrays: An array can be defined by one of four procedures: zeros, ones, arange, or array. zeros creates an array of a specified size containing all zeros:
```python
import numpy as np
np.random.randn()  #int
np.random.randn(1) #1D
np.random.randn(1,5) #2D
np.zeros(5) #array([ 0.,  0.,  0.,  0.,  0.])
np.zeros(5,dtype=int) #array([0, 0, 0, 0, 0])
np.ones(5) #array([ 1.,  1.,  1.,  1.,  1.])
np.arange(10,-11,-1) #array([ 10,   9,   8,   7,   6,   5,   4,   3,   2,   1,   0,  -1,  -2, -3,  -4,  -5,  -6,  -7,  -8,  -9, -10])
z=np.arange(10) #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.append(z,4) #array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4])
```
*  Core Python has an array data structure, but it’s not nearly as versatile, efficient, or useful as the NumPy array. We will not be using Python arrays at all. Therefore, whenever we refer to an “array,” we mean a “NumPy array.”

* Multi-dimensional lists: Because Python arrays are actually lists, you are allowed to have jagged arrays. Multi-dimensional lists are just lists of lists:
```python
a=[[0,1,2],[3,4,5]] #List
type(a) #<type 'list'>
len(a) #length
import numpy as np
z=np.asarray(a)
z.ndim
z.shape
np.size(z)
np.append(z,np.array([0,0]))
#
# Iterate over list
b=[1,3,4,5]
for i in range(len(b)): #range or xrange can be used
	print b[i]
# Slicing
a=[0, 1, 1, 2, 3, 5, 8, 13]
# Syntax: a[start:end:step]
# (-)ve sign means look from the end
#
# Range
range(10)      # makes a list of 10 integers from 0 to 9
#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
range(3,10)    # makes a list of 10 integers from 3 to 9
#[3, 4, 5, 6, 7, 8, 9]
range(0,10,2)  # makes a list of 10 integers from 0 to 9
                        # with increment 2
#[0, 2, 4, 6, 8]
```

### Tupels
* Tuples are lists that are immutable. That is, once defined, the individual elements of a tuple cannot be changed. Whereas a list is written as a sequence of numbers enclosed in square brackets, a tuple is written as a sequence of numbers enclosed in round parentheses.
```python
t=(1, 1, 2, 3, 5, 8, 13)
type(t) #<type 'tuple'>
```

### Numpy Array
* Numpy Array (**type 'numpy.ndarray'**) is similar to a list but where all the elements of the list are of the same type.
* The elements of a NumPy array, or simply an array, are usually numbers, but can also be boolians, strings, or other objects
* When the elements are numbers, they must all be of the same type. For example, they might be all integers or all floating point numbers.
```python
import numpy as np
x=[1,2,3,4,5]
a=np.array(x)
```
* **linspace or logspace functions**
	* The linspace function creates an array of N evenly spaced points between a starting point and an ending point. The form of the function is linspace(start, stop, N). If the third argument N is omitted, then N=50.
```python
linspace(0, 10, 5) # array([  0. ,  2.5,  5. ,  7.5, 10. ])
```
	* logspace that produces evenly spaced points on a logarithmically spaced scale. The arguments are the same as those for linspace except that start and stop refer to a power of 10. That is, the array starts at $10^{\mathrm{start}}$ and ends at $10^{\mathrm{stop}}$.
* **arange**
The third way arrays can be created is using the NumPy arange function, which is similar to the Python range function for creating lists.
	* arange(start, stop, step). If the third argument is omitted step=1. If the first and third arguments are omitted, then start=0 and step=1.


#### References
* http://www.physics.nyu.edu/pine/pymanual/html/chap3/chap3_arrays.html

## FAQ's
* **How to find the version of installed library?**
```python
import pandas as pd
pd.__version__
```

* **How to view all the defined variables in python shell?**
```python
dir() #will give you the list of in scope variables:
globals() #will give you a dictionary of global variables
locals() #will give you a dictionary of local variables
vars()
vars().keys()
vars().values()
```
	* https://stackoverflow.com/questions/633127/viewing-all-defined-variables

* **How to install using `pip`?**
```shell
sudo pip install numpy
```
* **How to setup & publish project dependencies?**
	* create txt file, say name requirements, each line mention package name ex: ploty=version (optional)
```shell
sudo pip install -r requirements
```
* **List all python pakages**
```bash
pip freeze
pip list
```

* **Which version of python should pip point to, by default? Can it be changed?**
* **How to reinstall pip? How to fix `pip` topoint to python2 instead of python3, when python points to v2 and python3 points to v3?**
- https://askubuntu.com/questions/780502/ubuntu-16-pip-install-installs-to-python-3-instead-of-2
```bash
pip freeze
pip list
#
pip -V
#pip 10.0.0 from /home/game/.local/lib/python3.5/site-packages/pip (python 3.5)
#
pip2 -V
#pip 10.0.0 from /home/game/.local/lib/python2.7/site-packages/pip (python 2.7)
#
sudo python -m pip install -U --force-reinstall pip
```

* **Which python points to which version**
```bash
python -V
#Python 2.7.12
#
python2 -V
#Python 2.7.12
#
python3 -V
#Python 3.5.2
```
* **How to preserve files downloaded by pip after failed installation?** 
- https://superuser.com/questions/769565/how-to-preserve-files-downloaded-by-pip-after-failed-installation
```bash
sudo pip install -d $HOME/softwares/pip-cache
sudo pip3 install -d $HOME/softwares/pip-cache
```

* **How to automatically-creating-directories-with-file-output?
- https://stackoverflow.com/questions/12517451/automatically-creating-directories-with-file-output
```python
import os
import errno

filename = "/foo/bar/baz.txt"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with open(filename, "w") as f:
    f.write("FOOBAR")
```
## TIPs
* Use python shell to learn and ``help()`` command to learn the details of the functions

## Setting up for Web Application development
- https://optimalbi.com/blog/2016/03/31/apache-meet-python-flask/
- [scientific packages](https://www.nyayapati.com/srao/2014/05/how-to-install-numpy-scipy-pandas-matplotlib-and-scikit-learn-on-mavericks/)
- http://geoffboeing.com/2016/03/scientific-python-raspberry-pi/
- https://unix.stackexchange.com/questions/37313/how-do-i-grep-for-multiple-patterns-with-pattern-having-a-pipe-character
- https://www.thegeekstuff.com/2011/10/grep-or-and-not-operators/
```bash
sudo pip install numpy
sudo pip install scipy
sudo pip install pandas
sudo pip install matplotlib
sudo pip install scikit-learn
sudo pip install Flask
#
pip list | grep -E 'numpy|scipy|pandas|matplotlib|scikit-learn|Flask'
sudo pip list --format=legacy | grep -E 'h5py|pillow|numpy|scikit-learn|tensorflow|keras|cv2'
sudo pip list --format=columns | grep -E 'h5py|pillow|numpy|scikit-learn|tensorflow|keras|cv2'
#
#Flask                         0.12.2     
#matplotlib                    2.2.2      
#numpy                         1.14.2     
#pandas                        0.22.0     
#scikit-learn                  0.19.1     
#scipy                         1.0.1      
```
* **Install form the directory**
```bash
sudo pip install `ls -1 | tr '\n' ' '`
```

**Natural Language Toolkit**
- https://www.nltk.org/

**Installing scikit-learn; Python Data Mining Library**
- https://calebshortt.com/2016/01/15/installing-scikit-learn-python-data-mining-library/
```python
import scipy
scipy.__version__
'1.0.1'
#
import numpy as np
np.__version__
'1.14.2'
#
import flask
flask.__version__
'0.12.2'
```

## Basic Python Concepts

**Book notes**
1. ScipyLectures-simple.pdf

**List Comprehensions**
`[i**2 for i in range(4)]`

**Mutable Types**
- dictionary
- list

**IMMutable Types**
- sting
- tuble: Tuples are lists that are immutable

**Standalone scripts may also take command-line arguments**
- Importing objects from modules
```python
import sys
print sys.argv
#
# Don’t implement option parsing yourself. Use modules such as optparse , argparse or
:mod‘docopt‘.
#
import os
os.listdir('.')
dir(os)
```

**Modules & Packages**
- Modules are cached: if you modify demo.py and re-import it in the old session, you will get the old one.
```python
reload(demo)
```
- Sometimes we want code to be executed when a module is run directly, but not when it is imported by another module. if __name__ == '__main__' allows us to check whether the  module is being run directly.
```python
if __name__ == '__main__':
	do_something_here
```
- A directory that contains many modules is called a package
- A package is a module with submodules (which can have submodules themselves, etc.)
- A special file called __init__.py (which may be empty) tells Python that the directory is a Python package, from which modules can be imported.
```python
import scipy
scipy.__file__
#
import scipy.ndimage.morphology
#
from scipy.ndimage import morphology
```

**Standard Library**
* **os module**
	- operating system functionality - A portable way of using operating system dependent functionality.
```python
os.getcwd() #Current directory:
os.listdir(os.curdir) #List a directory
os.mkdir('junkdir') #Make a directory:
os.rename('junkdir', 'foodir') #Rename the directory
os.rmdir('foodir') #Delete directory
os.remove('junk.txt') #Delete file
#
# os.path: path manipulations
# os.path provides common operations on pathnames.
a = os.path.abspath('junk.txt')
os.path.split(a)
os.path.dirname(a)
os.path.basename(a)
os.path.splitext(os.path.basename(a))
os.path.exists('junk.txt')
os.path.isfile('junk.txt')
os.path.isdir('junk.txt')
os.path.expanduser('~/local')
os.path.join(os.path.expanduser('~'), 'local', 'bin')
#
# Running an external command
#
os.system('ls')
#
# Walking a directory
# os.path.walk generates a list of filenames in a directory tree
#
for dirpath, dirnames, filenames in os.walk(os.curdir):
	for fp in filenames:
		print os.path.abspath(fp)
#
# Environment Variables
#
os.environ['PYTHONPATH']
os.getenv('PYTHONPATH')
```

* **urllib**
- URL handling library
```python
import urllib
import os

file = 'airfares.txt'
url = 'http://www.stat.ufl.edu/~winner/data/airq4.dat'
if not os.path.exists(file):
	urllib.urlretrieve(url,file)
#
data = pandas.read_csv(file,sep=' +',header=0,names=['city1', 'city2', 'pop1', 'pop2','dist', 'fare_2000', 'nb_passengers_2000','fare_2001', 'nb_passengers_2001'])
data = pandas.read_csv(file,sep=' +',header=0,names=['city1', 'city2', 'pop1', 'pop2','dist', 'fare_2000', 'nb_passengers_2000','fare_2001', 'nb_passengers_2001'])
#
```
__main__:1: ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators (separators > 1 char and different from '\s+' are interpreted as regex); you can avoid this warning by specifying engine='python'.

* **sh module**
  - Which provides much more convenient ways to obtain the output, error stream and exit code of the external command.
```python
#
# Alternative to os.system
#
import sh
com = sh.ls()
print com.exit_code
```
* **shutil**
  - high-level file operations
	- The shutil provides useful file operations:
		- shutil.rmtree : Recursively delete a directory tree.
		- shutil.move : Recursively move a file or directory to another location.
		- shutil.copy : Copy files or directories.

* **glob**
  - Pattern matching on files
	- The glob module provides convenient file pattern matching. example: Find all files ending in .txt
```python
import glob
glob.glob('*.txt')
```

* **sys module**
	- system-specific information
  - System-specific information related to the Python interpreter
  - Which version of python are you running and where is it installed
```python
import sys
sys.platform
sys.version
sys.prefix
sys.argv # List of command line arguments passed to a Python script
sys.path # List of strings that specifies the search path for modules. Initialized from PYTHONPATH
```

* **pickle**
  - easy persistence
  - Useful to store arbitrary objects to a file
  - **Not safe or fast!**
  - **TBD**: alternative to pickle
```python
import pickle
l = [1, None, 'Stan']
pickle.dump(l, file('test.pkl', 'w'))
pickle.load(file('test.pkl'))
```
* **timeit**
- https://docs.python.org/2/library/timeit.html
```python
import timeit
timeit.timeit(stmt='t=[i**2 for i in range(1000)]',number=100)
timeit.timeit(stmt='t=[i**2 for i in range(1000)]',number=1000)
timeit.timeit(stmt='t=[i**2 for i in range(1000)]',number=10000)
```

**Exception Handling**
- Exceptions are raised by errors in Python
- There are different types of exceptions for different errors
- Capturing and reraising an exception
- Use exceptions to notify certain conditions are met (e.g. StopIteration) or not (e.g. custom error raising)

**Object-oriented programming (OOP)**
- Python supports object-oriented programming (OOP)
- class, methods, attributes
- constructor: `__init__`

**Iterators, generator expressions and generators**
* **Iterator**
	- An iterator is an object adhering to the iterator protocol — basically this means that it has a next method, which, when called, returns the next item in the sequence, and when there’s nothing to return, raises the StopIteration exception.
	- When used in a loop, StopIteration is swallowed and causes the loop to finish. But with explicit invocation,
we can see that once the iterator is exhausted, accessing it raises an exception.
```python
num=[1,2,3]
it=iter(num)
next(it)
next(it)
next(it)
next(it) # StopIteration Exception
```
* **Generator expressions**
	- A second way in which iterator objects are created is through generator expressions
	- the basis for list comprehensions
	- a generator expression must always be enclosed in parentheses or an expression
	- If round parentheses are used, then a generator iterator is created
	- If rectangular parentheses are used, the process is short-circuited and we get a list
	- The list comprehension syntax also extends to dictionary and set comprehensions
	- A set is created when the generator expression is enclosed in curly braces
	- A dict is created when the generator expression contains “pairs” of the form key:value.
```python
(i for i in num) # generator is created
[i for in num] # list is created
#
{i for i in range(3)} # set is created: set([0, 1, 2])
#
# in old Pythons the index variable ( i ) would leak, and in versions >= 3 this is fixed.
{i:i**2 for i in range(3)} # dictionary is created: {0: 0, 1: 1, 2: 4}
```
* **Generators**
	- A generator is a function that produces a sequence of results instead of a single value
	- A third way to create iterator objects is to call a generator function
	- A generator is a function containing the keyword `yield`
	- When a normal function is called, the instructions contained in the body start to be executed. When a generator is called, the execution stops before the first instruction in the body.
	- An invocation of a generator function creates a generator object, adhering to the iterator protocol.
	- As with normal function invocations, concurrent and recursive invocations are allowed
	- When next is called, the function is executed until the first yield 
	- Each encountered yield statement gives a value becomes the return value of next
	- After executing the yield statement, the execution of this function is suspended
	- execution is strictly single-threaded, but the interpreter keeps and restores the state in between the requests for the next value
	- Usefulness: it is easier for the author of the generator to understand the state which is kept in local variables, as opposed to instance attributes, which have to be used to pass data between consecutive invocations of next on an iterator object
	- When an iterator is used to power a loop, the loop becomes very simple
	- When the generator resumes execution after a yield statement, the caller can call a method on the generator object to either pass a value into the generator, which then is returned by the yield statement, or a different method to inject an exception into the generator
	- `throw(type, value=None, traceback=None)`
	- `send(value)` equivalent to `g.next() and g.send(None)` # raise type, value, traceback
	- `close()` method, which can be used to force a generator that would otherwise be able to provide more values to finish immediately
```python
def f():
	yield 1
	yield 2
#
gen=f()
#
next(f)	# 1
next(f) # 2
next(f) # StopIteration
```

* **Decorators**
  - Since functions and classes are objects, they can be passed around. Since they are mutable objects, they can be modified. The act of altering a function or class object after it has been constructed but before it is bound to its name is called decorating
  - Decorators can be applied to functions and to classes
  - Decorators can be stacked — the order of application is bottom-to-top, or inside-out.
  - The only requirement on decorators is that they can be called with a single argument

* **Context managers**: `with` statement
	- A context manager is an object with __enter__ and __exit__ methods which can be used in the with statement
	- Use cases:
		* Using generators to define context managers: use a decorator to turn generator functions into context managers
```python
with manager as var:
  do_something(var)
#
# simplest case equivalent to
var = manager.__enter__()
try:
	do_something(var)
finally:
	manager.__exit__()
#
# calls close and can be used as a context manager itself
with open('/tmp/file', 'a') as f:
	f.write('more contents\n')
```

* **Others**
```python
# Return a list of tuples, where each tuple contains the i-th element
## from each of the argument sequences.  The returned list is truncated
### in length to the length of the shortest argument sequence.
#
zip
```

### Debugging

* pyflakes
	- Detects syntax errors, missing imports, typos on names
	- Integrating pyflakes (or flake8) in your editor or IDE is highly recommended, it does yield productivity gains.
* pylint
* pychecker
* pyflakes
* flake8
* nosetests
* gdb for the C-debugging part
* python debugger, pdb
	- https://docs.python.org/library/pdb.html
* pycharm
  - https://www.jetbrains.com/help/pycharm/debugging-your-first-python-application.html

**Debugging segmentation faults using gdb - Pg322**
- http://wiki.python.org/moin/DebuggingWithGdb


You can launch a Python program through pdb by using pdb myscript.py or python -m pdb myscript.py.

There are a few commands you can then issue, which are documented on the pdb page.

Some useful ones to remember are:

b: set a breakpoint
c: continue debugging until you hit a breakpoint
s: step through the code
n: to go to next line of code
l: list source code for the current file (default: 11 lines including the line being executed)
u: navigate up a stack frame
d: navigate down a stack frame
p: to print the value of an expression in the current context

If you don't want to use a command line debugger, some IDEs like Pydev have a GUI debugger.



### Optimizing Code
- http://packages.python.org/line_profiler/
- https://docs.python.org/library/timeit.html
- No optimization without measuring! Measure: profiling, timing
- use timeit to time elementary operations:
- with switches -l, --line-by-line and -v, --view to use the line-by-line profiler and view the results in addition to saving them
- Profiler: `cProfile`
```bash
python -m cProfile -o demo.prof demo.py
```

**Commonly encountered tricks to make code faster**
- moving computation or memory allocation outside a for loop
- Vectorizing for loops
	* Find tricks to avoid for loops using numpy arrays. For this, masks and indices arrays can be useful.
- Broadcasting
	* Use broadcasting to do operations on arrays as small as possible before combining them.
- Be easy on the memory: use views, and not copies
	* Copying big arrays is as costly as making simple numerical operations on them
- Beware of cache effects
	* Memory access is cheaper when it is grouped: accessing a big array in a continuous way is much faster than random access. This implies amongst other things that smaller strides are faster

### Sparse Matrices
- scipy.sparse
- PyAMG
- Pysparse

## NumPy
- creating and manipulating numerical data
- NumPy, the core tool for performant numerical computing with Python
- closer to hardware (efficiency)
- designed for scientific computation (convenience)
- Memory-efficient container that provides fast numerical operations
- **Applications:**
	- values of an experiment/simulation at discrete time steps
	- signal recorded by a measurement device, e.g. sound wave
	- pixels of an image, grey-level or colour
	- 3-D data measured at different X-Y-Z positions, e.g. MRI scan
- Different data-types allow us to store data more compactly in memory, but most of the time we simply work with floating point numbers.
- The default data type for number array is floating point
- Few datatypes:
```python
dtype('int64')
dtype('float64')
dtype('complex128')
dtype('bool')
dtype('S7') #strings containing max. 7 letters
#
int32
int64
uint32
uint64
```
- Array Indices begin at 0, like other Python sequences (and C/C++). In contrast, in Fortran or Matlab, indices begin at 1.
- Usual python idiom for reversing a sequence is supported
- For multidimensional arrays, indexes are tuples of integers
- In 2D, the first dimension corresponds to rows, the second to columns.
- for multidimensional a , a[0] is interpreted by taking all elements in the unspecified dimensions.
- Slicing: Arrays, like other Python sequences can also be sliced
- A slicing operation creates a view on the original array, which is just a way of accessing array data. Thus the original array is not copied in memory.
- The axes of an array describe the order of indexing into the array e.g. acis refers to the first index coodrinate
- The shape of an array is a tuple indicating the number of elements along each axis.
- An existing array `a` has an attribute `a.shape` which contains this tuples
- Array functions
```python
import numpy as np
#
np.lookfor('create array')
#
help(np.array)
#
# 1-D Array
a = np.array([0, 1, 2, 3])	# 1D
b = np.array([[0, 1, 2], [3, 4, 5]]) #2D: 2 x 3 array
c = np.array([[[1], [2]], [[3], [4]]])	#3D
type(a)
a.ndim
a.shape
a.dtype
np.shape(a)
len(a)
#
# Evenly spaced
a = np.arange(10) # 0 .. n-1 (!)
a[2:9:3] # [start:end:step]
#
# by number of points
np.linspace(0, 1, 6) # start, end, num-points
np.linspace(0, 1, 5, endpoint=False)
#
np.ones((3, 3)) # reminder: (3, 3) is a tuple
np.zeros((2, 2))
np.eye(3)
np.diag(np.array([1, 2, 3, 4]))
np.random.rand(4) # Gaussian
np.random.seed(1234) # Setting the random seed
#
np.lookfor(np.empty)
help(np.empty)
#
# explicitly specify which data-type
np.array([1, 2, 3], dtype=float)
#
# complex number/datatype
np.array([1+2j, 3+4j, 5+6*1j])
#
# Exercise
# Odd number counting backwards using np.linspace
np.linspace(1,20,20,dtype='int')[::2][::-1]
# Even number counting forward using np.linspace
np.linspace(2,21,20,dtype='int')[::2]
#
np.arange(0,51,10)[:,np.newaxis]
#
# Tiling array
# np.tile
a=np.array([(4,3),(2,1)])
np.tile(a,(2,3))
```
- `np.may_share_memory()` to check if two arrays share the same memory block. Note  however, that this uses heuristics and may give you false positives.
- NumPy arrays can be indexed with slices, but also with boolean or integer arrays (masks). This method is called fancy indexing. It creates copies not views
```python
import numpy as np
#
# copy(); np.copy()
a=np.arange(10)
c=a[::2].copy()
#
# np.may_share_memory
np.may_share_memory(a,c)
np.sqrt
np.nonzero
#
# Return an array representing the indices of a grid.
x, y = np.indices((1, 1))
#
np.searchsorted
#
```
- **masking:** Indexing with a mask can be very useful to assign a new value to a sub-array
- NumPy arrays can be indexed with slices, but also with boolean or integer arrays (masks). This method is called **fancy indexing**. It creates copies not views
```python
import numpy as np
np.random.seed(3)
a=np.random.randint(0, 21, 15) # 15 numbers between 0 and 21
mask=(a%3==0)
extract_from_a = a[mask]
a[mask] = 0
```
- Indexing can be done with an array of integers, where the same index is repeated several time
```python
import numpy as np
a = np.arange(0, 100, 10)
a[[2, 3, 2, 4, 2]]
a[[7,9]]=0
a[[7,9]]=[70,90]
```
- When a new array is created by indexing with an array of integers, the new array has the same shape as the array of integers
```python
import numpy as np
a=np.arange(10)
idx=np.array([[3,4],[9,7]])
idx.shape
a[idx]
```
- **Numerical operations on arrays**
- Elementwise operations: Basic operations on numpy arrays (addition, etc.) are elementwise
```python
import numpy as np
x=np.arange(1,11) # (start,stop,step)
x+1
x-2
x*2
x**2
2**x
#
# Array-wise comparisons
y=np.arange(1,11)
x==y
x>y
np.array_equal(x,y)
#
# Logical operations
np.logical_or(x,y)
np.logical_and(x,y)
np.all([True, True, False]) # Test whether `all` array elements along a given axis evaluate to True
np.any([True, True, False]) #  Test whether `any` array element along a given axis evaluates to True
#
# Transcendental functions
np.sin(x)
np.log(x)
np.exp(x)
#
# Computing sums
x=np.arange(5)
np.sum(x)
x.sum()
#
# Sum by rows and by columns
x=np.array([[1,1],[2,2]])
x.sum(axis=0) # columns (1st dimension)
x.sum(axis=1) # rows (2nd dimension)
#
# Unique Values
np.unique(x)
#
# cummulative sum
x.cumsum()
#
# Same idea in higher dimensions
#
x=np.array([[1,2],[3,4]])
x.min()
x.min(axis=0)
x.min(axis=1)
x.max()
x.max(axis=0)
x.max(axis=1)
x.argmin() # index of minimum
x.argmax() # index of maximum
#
# Datatype
a=np.arange(10)
a.shape,type(a),a.ndim,a.dtype # ((10,), <type 'numpy.ndarray'>, 1, dtype('int64'))
#
b=np.arange(10).astype(float) # ((10,), <type 'numpy.ndarray'>, 1, dtype('float64'))
b.shape,type(b),b.ndim,b.dtype
```
- Array multiplication is not matrix multiplication
- Array multiplication is Elementwise operations
```python
import numpy as np
x=np.ones((3,3))
c*c
c**c # c to the power of c
```
- Matrix multiplication
```python
c.dot(c)
```
- **Transposition**
- The transposition is a view
```python
import numpy as np
help(np.triu) # Upper triangle of an array
help(np.tril) # Lower triangle of an array
x=np.ones((3,3))
np.triu(x)
a=np.triu(x,1)
a.T
```
- **Some Useful Transformations**
```python
# Flip array in the up/down direction.
np.flipud(x)
```
- **Linear algebra**
- `numpy.linalg` implements basic linear algebra such as solving linear systems, singular value  decomposition, etc. However, it is not guaranteed to be compiled using efficient routines
- instead of `numpy.linalg` use of `scipy.linalg`
- **Statistics**
```python
import numpy as np
x=np.arange(1,11)
#
x.mean()
np.mean(x)
#
x.median() # AttributeError: 'numpy.ndarray' object has no attribute 'median'
np.median(x)
#
x.std()
np.std(x)
#
np.random.normal(0,1,n)
np.random.uniform(0.5, 1.0, n)
```
- **loading data**
- [populations.txt](https://www.scipy-lectures.org/_downloads/populations.txt)
```python
import numpy as np
data=np.loadtxt('data/populations.txt')
year, hares, lynxes, carrots = data.T # trick: columns to variables
#
import matplotlib.pyplot as plt
plt.axes([0.2,0.1,0.5,0.8])
plt.plot(year, hares, year, lynxes, year, carrots)
plt.legend(('Hare','Lynx','Carrot'),loc=(1.05,0.5))
plt.show()
```
- **Broadcasting**
- It’s also possible to do operations on arrays of different sizes if NumPy can transform these arrays so that they all have the same size: this conversion is called broadcasting.
- to solve a problem whose output data is an array with more dimensions than input data
- A lot of grid-based or network-based problems can also use broadcasting.
- Some Errors in broadcasting
	* IndexError: shape mismatch: indexing arrays could not be broadcast together with shapes (384,) (512,)
```python
import numpy as np
a = np.tile(np.arange(0, 40, 10), (3, 1)).T
b = np.array([0, 1, 2])
a+b
```
- For instance, if we want to compute the distance from the origin of points on a 10x10 grid, we can do:
- `np.ogrid()` function allows to directly create vectors x and y of the previous example, with two “significant dimensions”
- `np.mgrid` directly provides matrices full of indices for cases where we can’t (or don’t want to) benefit from broadcasting
- `np.mgrid`, `np.ogrid`, `np.meshgrid`
```python
import numpy as np
import matplotlib.pyplot as plt
x, y = np.arange(5), np.arange(5)[:, np.newaxis] # defining x and y variables in a single line
distance = np.sqrt(x ** 2 + y ** 2) # linear distance formula
plt.pcolor(distance)
plt.colorbar()
plt.show()
#
# np.ogrid
x,y=np.ogrid[0:5,0:5]
#
# np.mgrid
x1,y1=np.mgrid[0:5,0:5]
#
#
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
```
- **Array shape manipulation**
- ++Flattening++: Higher dimensions: last dimensions ravel out “first”.
- ++Reshaping++: ndarray.reshape may return a view or copy. To understand this you need to learn more about the memory layout of a numpy arrays
- ++Adding a dimension++: Indexing with the `np.newaxis` object allows us to add an axis to an array
- ++Dimension shuffling++:
- ++Resizing++: Size of an array can be changed with `ndarray.resize`
- ++Sorting data++: `np.sort`
- `np.argsort, np.argmax, np.argmin, np.random.shuffle(a),  np.random.random`
```python
import numpy as np
#
# Flattening
a = np.array([[1, 2, 3], [4, 5, 6]])
a.ravel()
a.T.ravel()
a.flatten()
#
# Reshaping
a.ravel().reshape((2,3))
a.ravel().reshape((2,-1)) # unspecified (-1) value is inferred
#
# Adding a dimension
z = np.array([1, 2, 3])
z[:,np.newaxis]
#
# Dimension shuffling
a = np.arange(4*3*2).reshape(4, 3, 2)
b = a.transpose(1, 2, 0)
#
# Resizing
a = np.arange(4)
a.resize((8,))
#
# others
## Generate a NumPy array of 10,000 random numbers
b=np.random.randint(1000, size=10000)
a=np.random.randint(0,40,10)
a
np.random.shuffle(a)
a
np.max(a) # max value
np.argmax(a) # index of max value
a[np.argmax(a)]==a.max()
```
-  As a general rule, NumPy should be used for larger lists/arrays of numbers, as it is significantly more memory efficient and faster to compute on than lists.
- **How to square or raise to a power (elementwise) a 2D numpy array?**
  - https://stackoverflow.com/questions/25870923/how-to-square-or-raise-to-a-power-elementwise-a-2d-numpy-array
  - The fastest way is to do `a*a` or `a**2` or `np.square(a)` whereas `np.power(a, 2)` showed to be considerably slower
- **Universal functions**
  - Ufunc performs and elementwise operation on all elements of an array.
  - http://wiki.cython.org/MarkLodato/CreatingUfuncs

- **chararray**
	- chararray: vectorized string operations
	- `.view()` has a second meaning: it can make an ndarray an instance of a specialized ndarray subclass
```python
x = np.array(['a', ' bbb', ' ccc']) # type: <type 'numpy.ndarray'>
y = x.view(np.chararray) # type: <class 'numpy.core.defchararray.chararray'>
y.lstrip(' ').upper()
```
- **maskedarray**
	- Masked arrays are arrays that may have missing or invalid entries
	- Not all NumPy functions respect masks, for instance np.dot , so check the return types.
	- The masked_array returns a view to the original array
```python
x = np.array([1, 2, 3, -99, 5])
mx = np.ma.masked_array(x, mask=[0, 0, 0, 1, 0]) # type: <class 'numpy.ma.core.MaskedArray'>
mx # masked_array(data = [1 2 3 -- 5],mask = [False False False True False],fill_value = 999999)
```

- **matrix**
	- `np.matrix`
	- always 2-D
	- * is the matrix product, not the elementwise one
```python
np.matrix([[1, 0], [0, 1]]) * np.matrix([[1, 2], [3, 4]]) # type: <class 'numpy.matrixlib.defmatrix.matrix'>
```
- **Summary**
	- Know how to create arrays : array , arange , ones , zeros .
	- Know the shape of the array with array.shape , then use slicing to obtain different views of the array: array[::2] , etc. Adjust the shape of the array using reshape or flatten it with ravel.
	- Obtain a subset of the elements of an array and/or modify their values with masks
	- Know miscellaneous operations on arrays, such as finding the mean or max ( array.max() , array.mean() ). No need to retain everything, but have the reflex to search in the documentation (online docs, help() , lookfor() )
	- master the indexing with arrays of integers, as well as broadcasting. Know more NumPy functions to handle various array operations
- numpy.lookfor looks for keywords inside the docstrings of specified module
- `numexpr` is designed to mitigate cache effects in array computing
```python
import numpy as np
np.pi
#
a=np.arange(0,11)
a.strides
#
# some functions
help()
str()
len()
```
- other functions
```python
import nummpy as np
from skimage import data, exposure, img_as_float
image = img_as_float(data.camera())
np.histogram(image,bins=2)
```
**Tutorials**
* https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-39.php

## Matplotlib
- 2D plotting package
- start with basic visualization of data arrary
- pyplot: provides a procedural interface to the matplotlib object-oriented plotting library
- **TBD**: Learn about `figure` and `subplot`
- Figures, Subplots, Axes and Ticks
```python
import numpy as np
import matplotlib.pyplot as plt

# 1D plotting
# line-plot
x=np.linspace(0,3,20)
y=np.linspace(0,9,20)
plt.plot(x,y)
plt.show()
plt.plot(x, y, 'o') # dot plot
#
# 2D arrays (such as images)
image = np.random.rand(30, 30)
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()
plt.show()
#
import numpy as np
import matplotlib.pyplot as plt
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
plt.plot(X, C)
plt.plot(X, S)
plt.show()
# commands
plt.plot
#
plt.figure
fig = plt.figure(figsize=(6, 6)) # figure size in inches
fig.subplots_adjust
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
#
plt.subplot
plt.xlim
plt.ylim
plt.xticks
plt.yticks
plt.set_xticklabels
plt.set_yticklabels
plt.gca() # gca stands for 'get current axis'
plt.legend
plt.annotate
plt.close
plt.show
plt.text
plt.title
#
plt.imshow
plt.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
plt.imshow(digits.images[i], cmap=plt.cm.gray, interpolation='nearest')
plt.imshow(digits.images[i], cmap="gray", interpolation='nearest')
#
plt.axes
plt.pie
plt.bar
plt.scatter
plt.quiver
plt.contour
plt.contourf
plt.pcolormesh
plt.scatter
# Automatically adjust subplot parameters to give specified padding
plt.tight_layout
#
# Create color maps
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
```
* **Subplots**
	- https://matplotlib.org/gallery/subplots_axes_and_figures/subplots_demo.html
* **Legend**
	- https://matplotlib.org/2.0.2/users/legend_guide.html

### mpl_toolkits - Basic 3D in Matplotlib
https://matplotlib.org/1.4.3/mpl_toolkits/index.html


## Scipy: High Level Scientific Computing

- `scipy` is the core package for scientific routines in Python
- The scipy package contains various toolboxes dedicated to common issues in scientific computing
- it is meant to operate efficiently on numpy arrays, so that numpy and scipy work hand in hand
- Scipy’s routines are optimized and tested, and should therefore be used when possible
- Its different submodules correspond to different applications, such as interpolation, integration, optimization, image processing, statistics, special functions, etc.
- scipy is composed of task-specific sub-modules, some are:-
	* scipy.cluster - Vector quantization / Kmeans
	* scipy.constants - Physical and mathematical constants
	* scipy.fftpack - Fast Fourier transforms
	* scipy.integrate - Numerical integration routines
	* scipy.interpolate - Interpolation
	* scipy.io - Data/File input and output
	* scipy.linalg - Linear algebra routines/operations
	* scipy.ndimage - Image manipulation: n-dimensional image package
	* scipy.odr - Orthogonal distance regression
	* scipy.optimize - Optimization and fit
	* scipy.signal - Signal processing
	* scipy.sparse - Sparse matrices
	* scipy.spatial - Spatial data structures and algorithms
	* scipy.special - Special functions/Any special mathematical functions
	* scipy.stats - Statistics and random numbers
```python
import numpy as np
from scipy import stats # same for other sub-modules
from scipy import io as spio
```

### Statistics and random numbers: `scipy.stats`
- `scipy.stats.ttest_1samp()` tests if the population mean of data is likely to be equal to a given value.
- It returns `T-statistic` and the `p-value`
```python
from scipy import stats
stats.ttest_1samp(data['VIQ'], 0)
```
- to To test if the difference of a mean between two groups for a particular variable is significant. Do a 2-sample t-test with `scipy.stats.ttest_ind()`
```python
from scipy import stats
stats.ttest_ind(viq_f,viq_m)
```

### Linear algebra operations: `scipy.linalg`

```python
scipy.linalg.det() # determinant of a square matrix
scipy.linalg.inv() # inverse of a square Matrix
#
# SVD - Singular Value Decomposition
a=np.arange(9).reshape((3,3))+np.diag([1,0,1])
u,s,v=linalg.svd(a)
## verify svd
aa=u.dot(np.diag(s)).dot(v)
np.allclose(a,aa)
```
- principal component analysis (PCA) or SVD: PCA is a technique for dimensionality reduction, i.e. an algorithm
to explain the observed variance in your data using less dimensions.
- independent component analysis (ICA): ICA is a source seperation technique, for example to unmix multiple signals that have been recorded through multiple sensors. Doing a PCA first and then an ICA can be useful if you have more sensors than signals.

### Interpolate: `scipy.interpolatescipy.interpolate`
- is useful for fitting a function from experimental data and thus evaluating points where no measure exists
- example: Maximum wind speed prediction at the Sprogø station for a more advanced spline interpolation
```python
scipy.interpolate.interp1d
scipy.interpolate.interp2d
```

### Optimization and fit: `scipy.optimize`
- Optimization is the problem of finding a numerical solution to a minimization or equality
- this package provides algorithms for function minimization (scalar or multi-dimensional), curve fitting and root finding.
- least squares curve fitting
- http://www.scipy-lectures.org/intro/scipy/auto_examples/plot_curve_fit.html
- If the function is a smooth function, gradient-descent based methods are good options. The `lBFGS` algorithm is a good choice in general
- A possible issue with this approach is that, if the function has local minima, the algorithm may find these local minima instead of the global minimum depending on the initial point x0
- If we don’t know the neighborhood of the global minimum to choose the initial point, we need to resort to 
costlier global optimization
- To find the global minimum, we use `scipy.optimize.basinhopping()` (added in version 0.12.0 of Scipy). It combines a local optimizer with sampling of starting points
```python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Seed the random number generator for reproducibility
np.random.seed(0)

x_data = np.linspace(-5, 5, num=50)
y_data = 2.9 * np.sin(1.5 * x_data) + np.random.normal(size=50)

def test_func(x, a, b):
    return a * np.sin(b * x)

params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
print(params)
#
# plot the resulting curve on the data
plt.figure(figsize=(6, 4))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, test_func(x_data, params[0], params[1]), label='Fitted function')
plt.legend(loc='best')
plt.show()
#
optimize.minimize(f, x0=0)
optimize.minimize(f, x0=0, method="L-BFGS-B")
#
# global minimum
optimize.basinhopping(f, 0)
optimize.minimize(f, x0=1,bounds=((0, 10), ) )
```
- Minimizing functions of several variables: To minimize over several variables, the trick is to turn them into a function of a multi-dimensional variable (a vector)
- `optimize.minimize_scalar()` is a function with dedicated methods to minimize functions of only one variable
- Filters should be created using the scipy filter design code
- Exercise
```python
temp_max = np.array([17, 19, 21, 28, 33, 38, 37, 37, 31, 23, 19, 18])
temp_min = np.array([-62, -59, -56, -46, -32, -18, -9, -13, -25, -46, -52, -58])
```
- Optimization of a two-parameter function
```python
import numpy as np
# Define the function that we are interested in
def sixhump(x):
	return ((4 - 2.1*x[0]**2 + x[0]**4 / 3.) * x[0]**2 + x[0] * x[1] + (-4 + 4*x[1]**2) * x[1] **2)

# Make a grid to evaluate the function (for plotting)
x = np.linspace(-2, 2)
y = np.linspace(-1, 1)
xg, yg = np.meshgrid(x, y)
#
# (AxisConcatenator) |  Translates slice objects to concatenation along the second axis.
help(np.c_)
#
np.dstack([xg.flat,yg.flat])
np.vstack
#
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(sixhump([xg, yg]), extent=[-2, 2, -1, 1])
plt.colorbar()
#
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xg, yg, sixhump([xg, yg]), rstride=1, cstride=1,
cmap=plt.cm.jet, linewidth=0, antialiased=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Six-hump Camelback function')
#
from scipy import optimize
x_min = optimize.minimize(sixhump, x0=[0, 0])
plt.figure()
# Show the function in 2D
plt.imshow(sixhump([xg, yg]), extent=[-2, 2, -1, 1])
plt.colorbar()
# And the minimum that we've found:
plt.scatter(x_min.x[0], x_min.x[1])
plt.show()
#
# variety of filters
from scipy import ndimage
from scipy import signal
#
# fft - fast fourier transform
from scipy import fftpack
#
np.random.standard_normal
np.copy(face).astype(np.float)
```
- image blur by convolution with a Gaussian kernel
- Image denoising by FFT


## Mayavi: 3D plotting with Mayavi

- Example
```python
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
plt.show()
```
* Image plane widgets
* Isosurfaces

## Other Libraries/Packages/Extensions/Modules

### Cython
- http://cython.org/
- https://en.wikipedia.org/wiki/Cython
- Cython is an optimising static compiler for both the Python programming language and the extended Cython programming language (based on Pyrex). It makes writing C extensions for Python as easy as Python itself
- Cython is a superset of the Python programming language, designed to give C-like performance with code that is mostly written in Python
- Cython is a compiled language that generates CPython extension modules. These extension modules can then be loaded and used by regular Python code using the import statement

### PIL
- https://en.wikipedia.org/wiki/Python_Imaging_Library
- PIL is the Python Imaging Library
- PIL is one of the core libraries for image manipulation in Python. Unfortunately, its development has stagnated, with its last release in 2009

### Pillow
- http://python-pillow.org/
- https://pillow.readthedocs.io/
- Actively-developed fork of PIL called Pillow - it’s easier to install, runs on all operating systems, and supports Python 3
```bash
# Installation
pip install Pillow
```

## Image Processing in Python
- image-processing.md

**Some Common tasks in image processing:**
1. Input/Output, displaying images
2. Basic manipulations:
	- Geometrical transformations: cropping, flipping, rotating, ...
	- Statistical information
3. Image filtering:
	- Blurring/smoothing
	- Sharpening
	- Denoising
	- Mathematical morphology
4. Image segmentation: labeling pixels corresponding to different objects
5. Classification
6. Feature extraction:
	- Edge detection
	- Segmentation
7. Measuring objects properties: ndimage.measurements
8. Registration

## Interfacing with C/C++
- A process commonly referred to wrapping.
- Cython is the most modern and advanced. In particular, the ability to optimize code incrementally by adding types to your Python code is unique.
- These four techniques are perhaps the most well known ones, of which Cython is probably the most advanced one and the one you should consider using first
	* Python-C-Api
	* Ctypes
	* SWIG
	* Cython

## Statistics in Python
* Standard scientific Python environment (numpy, scipy, matplotlib)
* Pandas
* [Statsmodels](http://statsmodels.sourceforge.net/stable/example_formulas.html)
* [Seaborn](http://seaborn.pydata.org)
	- Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics: `import seaborn as sns`

**Why Python for statistics?**
- When it comes to building complex analysis pipelines that mix statistics with e.g. image analysis, text mining, or control of a physical experiment, the richness of Python is an invaluable asset.

- multiple observations or samples described by a set of different attributes or features

## Pandas
* data-frame
	- We will store and manipulate this data in a pandas.DataFrame , from the pandas module. It is the Python equivalent of the spreadsheet table
	- It is different from a 2D numpy array as it has named columns, can contain a mixture of different data types by column, and has elaborate selection and pivotal mechanisms.

The installed version of numexpr 2.4.3 is not supported in pandas and will be not be used The minimum supported version is 2.4.6

```
sudo pip list | grep numexpr
sudo pip install -U numexpr
```
- **numexpr:** for accelerating certain numerical operations. numexpr uses multiple cores as well as smart chunking and caching to achieve large speedups. If installed, must be Version 2.1 or higher.
- numexpr (2.6.5), numpy (1.14.3)
```python
import pandas
data = pandas.read_csv('brain_size.csv',sep=';',na_values=".")
data.shape
data.columns
data['Gender']
data[data['Gender'] == 'Female']
data[data['Gender'] == 'Female']['VIQ'].mean()	
```
- pandas can input data from SQL, excel files, or other formats
- data is a pandas.DataFrame , that resembles R’s dataframe
- A pandas.DataFrame can also be seen as a dictionary of 1D ‘series’, eg arrays or lists.
- 3 numpy arrays can be exposed as a pandas.DataFrame
```python
pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})
```
- For a quick view on a large dataframe, use its describe method: `pandas.DataFrame.describe()`
- `groupby`: splitting a dataframe on values of categorical variables
- Other common grouping functions are **median**,**count** (useful for checking to see the amount of missing values in different subsets) or **sum**
- Groupby evaluation is lazy, no work is done until an aggregation function is applied.
```python
gGender = data.groupby('Gender')
for gender, value in gGender['VIQ']:
	print((gender, value.mean()))
gGender.mean()
#
gGender.boxplot()
```
- Pandas comes with some plotting tools ( `pandas.tools.plotting` , using `matplotlib` behind the scene) to display statistics of the data in dataframes

**Hypothesis testing**
- comparing two groups
- t-test: the simplest statistical tes
- 2 sample test
- “paired test”, or “repeated measures test”
- T-tests assume Gaussian errors.
- Wilcoxon signed-rank test - it that relaxes this Gaussian errors assumption
- non paired case is the **Mann–Whitney U test**: `scipy.stats.mannwhitneyu()`
- non parametric statistics to test the difference between a variable in two groups

## Statsmodel
**Given two set of observations, x and y, we want to test the hypothesis that y is a linear function of x**
- `y=mx+c+e`
```python
import numpy as np
m = -5
x = np.linspace(-5,5,20)
e = 4*np.random.normal(size=x.shape)
y = -5 + 3*x + e
data = pandas.DataFrame({'x':x,'y':y})
#
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
model.summary()
```
- Statsmodels uses a statistical terminology: the y variable in statsmodels is called ‘endogenous’ while the x variable is called exogenous.
- y (endogenous) is the value you are trying to predict
- x (exogenous) represents the features you are using to make the prediction
- http://statsmodels.sourceforge.net/devel/endog_exog.html
- non-float data type or categorical value, statsmodels is able to automatically infer this

**Categorical variables: comparing groups or multiple categories**
```python
model = ols("VIQ ~ Gender + 1", data).fit()
data = pandas.read_csv('brain_size.csv', sep=';', na_values=".")
#
# An integer column can be forced to be treated as categorical using
model = ols('VIQ ~ C(Gender)', data).fit()
#
# We can remove the intercept using - 1 in the formula, or force the use of an intercept using + 1
```
- By default, statsmodels treats a categorical variable with K possible values as K-1 ‘dummy’ boolean variables (the last level being absorbed into the intercept term). This is almost always a good default choice - however, it is possible to specify different encodings for categorical variables
- http://statsmodels.sourceforge.net/devel/contrasts.html
```python
import pandas
import
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
#
from statsmodels.formula.api import ols
model = ols("iq ~ type", data_long).fit()
model.summary()
#
# t-test
stats.ttest_ind(data['FSIQ'], data['PIQ'])
```

**Multiple Regression: including multiple factors**
- Consider a linear model explaining a variable z (the dependent variable) with 2 variables x and y:
```python
z = x*c1 + y*c2 + i + e
```
- Such a model can be seen in 3D as fitting a plane to a cloud of (x, y, z) points.

## Seaborn
**More visualization: seaborn for statistical exploration**
- http://seaborn.pydata.org
- http://lib.stat.cmu.edu/datasets/CPS_85_Wages
- http://gael-varoquaux.info/stats_in_python_tutorial/auto_examples/plot_wage_data.html
- Seaborn changes the default of matplotlib figures to achieve a more “modern”, “excel-like” look. It does that upon import. You can reset the default using:
	```python
	from matplotlib import pyplot as plt
	plt.rcdefaults()
	```
	- `lmplot`: plotting a univariate regression
	```python
	import seaborn
	seaborn.lmplot(y='VIQ',x='Height',data=data)
	```
- http://seaborn.pydata.org/tutorial/aesthetics.html
- To compute a regression that is less sentive to outliers, one must use a robust model.
- Formulate a single model that tests for a variance of slope across the two populations. This is done via an “interaction”.
- http://www.statsmodels.org/devel/example_formulas.html#multiplicative-interactions

## [Sympy](http://www.sympy.org/en/index.html)
- Symbolic Mathematics in Python
- What is SymPy? SymPy is a Python library for symbolic mathematics
- It aims to become a full-featured computer algebra system (CAS) while keeping the code as simple as possible in order to be comprehensible and easily extensible. SymPy is written entirely in Python.
- SymPy defines three numerical types: Real , Rational and Integer
- SymPy uses mpmath in the background, which makes it possible to perform computations using arbitrary-precision arithmetic.
- That way, some special constants, like e, pi , oo (Infinity), are treated as symbols and can be evaluated with arbitrary precision
- mathematical infinity, called `oo`
```python
import sympy as sym
a = sym.Rational(1, 2)
sym.pi**2
sym.pi.evalf()
(sym.pi + sym.exp(1)).evalf()
#
# mathematical infinity
sym.oo > 99999
```

## Key Learnings in Stats
* Hypothesis testing and p-values give you the significance of an effect / difference.
* Formulas (with categorical variables) enable you to express rich links in your data.
* Visualizing your data and fitting simple models give insight into the data.
* Conditionning (adding factors that can explain all or part of the variation) is an important modeling aspect that changes the interpretation.

## [Scikit-image: image processing](http://scikit-image.org/)
- scikit-image is a Python package dedicated to image processing, and using natively NumPy arrays as image objects

### References
- http://pymc-devs.github.io/pymc/
- http://greenteapress.com/thinkstats2/thinkstats2.pdf

## Online Practice Tools
- https://www.dataquest.io

## Contribution to documentation
- Refer: Pg 310

## References

1. List Comprehensions
  - https://www.analyticsvidhya.com/blog/2016/01/python-tutorial-list-comprehension-examples/
  - https://stackoverflow.com/questions/30245397/why-is-list-comprehension-so-faster/30245489
2. Profiling and Timing Code
	- https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html

http://josephcslater.github.io/scipy-numpy-matplotlib-pylab.html

## TBD List
- Pg 72: for now skipping to next chaptrer - matplotlib
- Pg 313: debugging code

## Python IDEs and Editors
- https://www.techradar.com/news/best-ide-for-python
- http://ubuntuhandbook.org/index.php/2018/03/pycharm-2018-1-released/
- https://cewing.github.io/training.codefellows/assignments/day01/sublime_as_ide.html
- https://wiki.python.org/moin/PythonEditors

**Pyzo**
http://www.pyzo.org/install_linux.html#install-linux
```bash
sudo python3 -m pip install pyzo
```

## Python - Building Interactive GUI
**Tool Suite**
* Enthought Tool Suite
	- The Enthought Tool Suite enable the construction of sophisticated application frameworks for data analysis, 2D plotting and 3D visualization

The main packages of the Enthought Tool Suite are:
* Traits
	- component based approach to build our application
* Kiva
	- 2D primitives supporting path based rendering, affine transforms, alpha blending and more.
* Enable
	- object based 2D drawing canvas.
* Chaco
  - plotting toolkit for building complex interactive 2D plots.
* Mayavi
	- 3D visualization of scientific data based on VTK.
* Envisage
	- application plugin framework for building scriptable and extensible applications

- wxPython, PyQt or PySide
- Numpy and Scipy

## Mayavi - 3D plotting
- `Mayavi` is an interactive 3D plotting package.
- `matplotlib` can also do simple 3D plotting, but Mayavi relies on a more powerful engine ( VTK ) and is more suited to displaying large or complex data

**Mlab: the scripting interface**
- `mayavi.mlab` module provides simple plotting functions to apply to numpy arrays, similar to matplotlib

## `sklearn` ML python package
* scikit-learn: machine learning in Python
* [Refer here: sklearn: Machine Learning in Python](https://github.com/mangalbhaskar/technotes/blob/master/machine-learning.md)

**Documentation**
* http://scikit-learn.org

**Tutorials**
* https://www.youtube.com/watch?v=r4bRUvvlaBw
* https://github.com/jakevdp/sklearn_tutorial
* https://github.com/ujjwalkarn/DataSciencePython

## Description on Different Python Packages
* **imutils (0.4.6)**
A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with OpenCV and both Python 2.7 and Python 3.


## Gif, Videos in Python
* https://www.idiotinside.com/2017/06/06/create-gif-animation-with-python/
```bash
sudo pip install imageio MoviePy
```

* http://free-tutorials.org/two-python-modules-moviepy-and-images2gif-part-001/
```
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
canvas = FigureCanvas(fig)
```

## Different Python Environments
* [Anaconda](https://anaconda.org/)
  - Anaconda is the most widely used Python distribution for data science and comes pre-loaded with all the most popular libraries and tools
  - https://docs.anaconda.com/anaconda/packages/pkg-docs

## [iPython](https://ipython.org/)
- iPython: Interactive Python

**ipython-jupyter**
- https://www.datacamp.com/community/blog/ipython-jupyter

## [Jupyter](https://jupyter.org/)
* The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.
* The Jupyter project is the successor to the earlier **IPython Notebook**, which was first published as a prototype in 2010. 

**[Getting Started](http://jupyter.readthedocs.io/en/latest/content-quickstart.html)**
* http://jupyter.readthedocs.io/en/latest/content-quickstart.html
* https://www.dataquest.io/blog/jupyter-notebook-tutorial/
- The Jupyter Notebook is an incredibly powerful tool for interactively developing and presenting data science projects.
- A notebook integrates code and its output into a single document that combines visualisations, narrative text, mathematical equations, and other rich media.

* **[important terminology](http://jupyter.readthedocs.io/en/latest/glossary.html#term-kernel)**
- A kernel provides programming language support in Jupyter. IPython is the default kernel. Additional kernels include R, Julia, and many more.

* **installing Jupyter and creating notebook**
```bash
sudo pip install jupyter
sudo pip3 install jupyter
```

* **[Configurations](http://jupyter.readthedocs.io/en/latest/projects/jupyter-directories.html)**
- Config files are stored by default in the `~/.jupyter` directory.
- To list the config directories currrently being used:
```bash
jupyter --paths
```
- Jupyter uses a search path to find installable data files

* **how to run and save notebooks**
```bash
# the current working directory will be the start-up directory
jupyter notebook
sudo jupyter notebook --allow-root
python /usr/local/bin/IPython
sudo python -m pip install -U --force-reinstall pip
```

- The first line of /usr/local/bin/ipython is "#!/usr/bin/python3" I could edit that line to use python instead of python3 or it was as simple as run with:
* **share and publishe notebooks online**

**How to run an .ipynb Jupyter Notebook from terminal?**
An IPYNB file is a notebook document used by Jupyter Notebook, an interactive computational environment designed to help scientists work with the Python language 
`.ipynb`
```bash
ipython nbconvert --to python <YourNotebook>.ipynb
#
# From the command line you can convert a notebook to python with this command:
ipython nbconvert --to python <YourNotebook>.ipynb
#
# You may have to install the python mistune package:
#
sudo pip install mistune
```

## Caffe, TensorFlow, Keras for Python2 and Python3

* some Issues:
```bash
usr/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: numpy.dtype size changed, may indicate binary incompatibility. Expected 96, got 88
  return f(*args, **kwds)
```

## Images
- https://matplotlib.org/gallery/images_contours_and_fields/image_clip_path.html#sphx-glr-gallery-images-contours-and-fields-image-clip-path-py

1. Reading Images in python - Input/output, data types and colorspaces
* skimage
- `skimage.data_dir` = `/usr/local/lib/python2.7/dist-packages/skimage/data`
- imsave also uses an external plugin such as PIL
```python
from skimage import io
import os
filename = os.path.join(skimage.data_dir, 'camera.png')
# reading image files
camera = io.imread(filename)
logo = io.imread('http://scikit-image.org/_static/img/logo.png')
io.imsave('local_logo.png', logo)
```
* matplotlib
```python
import matplotlib.pyplot as plt
plt.imread('MarshOrchid.jpg')
plt.imshow()
plt.show()
```
* cv2 (openCV)
* pillow, PIL

## Misc Topics, Keywords etc
* `__import__`
* `getattr`

## OOP in Python
> Object Oriented Programming in Python

* Classes and objects in Python
* `self` and `__init__`
  - https://www.programiz.com/article/python-self-why
  * The use of self makes it easier to distinguish between instance attributes (and methods) from local variables.
  * `self` is not yet a keyword in Python, like this in C++ and Java
  * `__init__` is not a constructor but `__init__` gets called when we create an object
  * Technically speaking, constructor is a method which creates the object itself. In Python, this method is `__new__`
  * __new__() is always called before __init__()
  * Generally, __init__() is used to initialize a newly created object while __new__() is used to control the way an object is created.
  * One practical use of __new__() however, could be to restrict the number of objects created from a class.
* Function Decorators
  - https://www.programiz.com/python-programming/decorator