# Is your code efficient?
Program optimisation can speed up your script thousands of times.

## Python is slow
![Loop iterations](../../image/language_speed.gif "Loop iterations")
Cython is a superset of the programming language Python, which allows developers to write Python code that yields performance comparable to that of C. Cython is a compiled language that is typically used to generate CPython extension modules.

## Keep your variable types
If your Cython modules are not efficient, please ensure the variable types are stable. Because crossing types are not recommended in Cython.