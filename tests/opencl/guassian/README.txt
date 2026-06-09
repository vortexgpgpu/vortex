The Gaussian Elimination application solves systems of equations using the
gaussian elimination method.

The application analyzes an n x n matrix and an associated 1 x n vector to solve a 
set of equations with n variables and n unknowns. The matrix and vector describe equations
of the form:

             a0x + b0y + c0z + d0w = e0
             a1x + b1y + c1z + d1w = e1
             a2x + b2y + c2z + d2w = e2
             a3x + b3y + c3z + d3w = e3

where in this case n=4.  The matrix for the above equations would be as follows:

            [a0 b0 c0 d0]
            [a1 b1 c1 d1]
            [a2 b2 c2 d2]
            [a3 b3 c3 d3]
            
and the vector would be:

            [e0]
            [e1]
            [e2]
            [e3]

The application creates a solution vector:

            [x]
            [y]
            [z]
            [w]
            

The Makefile may need to be adjusted for different machines, but it was written for Mac OS X and
Linux with either NVIDIA or AMD OpenCL SDKs.

Additional input files can be created with the matrixGenerator.py file in the data folder.

Gaussian Elimination Usage

    gaussianElimination [filename] [-hqt] [-p [int] -d [int]]
    
    example:
    $ ./gaussianElimination matrix4.txt
    
    filename     the filename that holds the matrix data
    
    -h, --help   Display the help file
    -q           Quiet mode. Suppress all text output.
    -t           Print timing information.
    
    -p [int]     Choose the platform (must choose both platform and device)
    -d [int]     Choose the device (must choose both platform and device)
    
    
    Notes: 1. The filename is required as the first parameter.
           2. If you declare either the device or the platform,
              you must declare both.

