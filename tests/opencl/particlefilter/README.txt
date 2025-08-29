file: README.txt
author: Donnie Newell - den4gr@virginia.edu
last modified: 18Jan2012

The optimized version of particlefilter has significant changes from 
the original particle filter kernel prior to December 2011. A brief list 
is included below:

- float data types changed to double because of overflow issues
- increased synchronization to correctly handle shared memory and partial blocks
- additional memcopies to print intermediate results for each frame

These additions result in different results and runtimes than previous versions.

Single: single precision
Double: double precision
