# PNS

[GitHub](https://github.com/gokcehan/pns)
| [Zenodo](https://zenodo.org/record/5502051)
| [Article](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.6659)

Parallel network simplex algorithm for the minimum cost flow problem.

This code is structured as a standalone tool to use as a solver for DIMACS
minimum cost flow problem files. Solution is then given as a DIMACS minimum
cost flow solution file optionally with flow value assignments. See the
provided samples as an example for these file formats.

Most procedures are similar to the network simplex algorithm implemented in the
[LEMON graph library](https://lemon.cs.elte.hu).
Our implementation includes OpenMP parallelism and AVX2/AVX512 vectorization on
top of these.

This documentation assumes a UNIX based operating system with essential
build tools installed and a POSIX compatible shell for commands. You may
need to adjust these to your setup for other combinations.

# Building

You can run `make` to compile all versions of the program. Alternatively, you
can give a single specific version to build (e.g. `make pns-omp-avx512`). See
`Makefile` for more information.

You need a compiler supporting C++11 and OpenMP. You also need the
[Boost Align library](https://www.boost.org/doc/libs/release/libs/align/)
installed somewhere in the standard include paths of the compiler, otherwise
you may need to provide the path manually yourself.

# Usage

The program reads the input from a file with the given name and then gives the
output to stdout.

An example file is as follows:

    $ cat samples/dimacs.min
    c Example instance provided in the DIMACS web page:
    c
    c http://lpsolve.sourceforge.net/5.5/DIMACS_mcf.htm
    c
    c This is a simple example file to demonstrate the DIMACS
    c input file format for minimum cost flow problems. The solution
    c vector is [2,2,2,0,4] with cost at 14.
    c
    c Problem line (nodes, links)
    p min 4 5
    c
    c Node descriptor lines (supply+ or demand-)
    n 1 4
    n 4 -4
    c
    c Arc descriptor lines (from, to, minflow, maxflow, cost)
    a 1 2 0 4 2
    a 1 3 0 2 2
    a 2 3 0 2 1
    a 2 4 0 3 3
    a 3 4 0 5 1
    c
    c End of file

This file can be used as follows:

    $ ./pns-seq samples/dimacs.min
    c pns v1.0.0
    c # nodes             : 4
    c # arcs              : 5
    c # arcs in a block   : 5
    c Init Time           : 0 ms
    c # iterations        : 4
    c Time                : 0 ms
    c - find entering arc : 0 ms (0.113433)
    c - find join node    : 0 ms (0.0742715)
    c - find leaving arc  : 0 ms (0.0953802)
    c - change flows      : 0 ms (0.0282871)
    c - change states     : 0 ms (0.0348259)
    c - update tree       : 0 ms (0.403412)
    c - update pots       : 0 ms (0.0274343)
    s 14

Run `pns-seq -h` to see all runtime options along with the usage.

Input and output both use the DIMACS file format described in the
[DIMACS mcf web page](http://lpsolve.sourceforge.net/5.5/DIMACS_mcf.htm).
See the following files in the `samples` directory for an input and output
example for this format:

    samples/dimacs.min -> example in the DIMACS mcf web page
    samples/dimacs.sol -> solution for dimacs.min
    samples/figure.min -> example in the paper
    samples/figure.sol -> solution for figure.min

Input problem type is assumed to be the minimum cost flow problem and non-zero
lower bounds are silently ignored. Also, problems are assumed to be feasible
and bounded. Other cases are not handled in the implementation and they can
result in unexpected behavior.

# OpenMP Options

OpenMP behavior is controlled with the environment variables defined in the
standard. You can use the following to display information about the OpenMP
environment when a program is running:

    export OMP_DISPLAY_ENV=TRUE

Scheduling behavior used for the experiments in the paper can be set as
follows:

    export OMP_SCHEDULE=DYNAMIC,16
    export OMP_PROC_BIND=TRUE
    export OMP_PLACES=sockets

You can set the number of threads as follows:

    OMP_NUM_THREADS=2 ./pns-omp-avx512 samples/dimacs.min

When these variables are not set, default values of these options are
implementation dependent.

# Experiments

Some scripts are provided to be able to easily run the experiments presented in
the paper. Note, it can take several days for all the runs to finish with
different instances and parameters. You may want to change these scripts to
disable parts that you are not interested or to use smaller instances.

We used two generators in the experiments namely `gridgen` and `netgen`. The
original versions of these programs can be found in the
[DIMACS netflow archive](http://archive.dimacs.rutgers.edu/pub/netflow/).
We patched these two programs to be able to generate bigger instances. These
patched versions are included in the repository along with their original
versions. You can use the following to compile these generators:

    make -C generators/gridgen-patched
    make -C generators/netgen-bcjl-patched

There is a script provided to generate instances using these two generators and
write the resulting files to `dataset` directory under the base directory. You
can see this script for the parameters we used to generate the instances used
in the paper. You can use this script as follows:

**Note:** These instances require about 60GB of disk space in total. Also, the
generators can use excessive memory during the execution.

    scripts/generate-all

The rest of the instances used for the experiments in the paper are downloaded
from the
[LEMON MinCostFlowData web page](https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData).
There is a script provided to download and extract these instances to `dataset`
directory under the base directory. You can see this script for the dimensions
of instances used in the paper. You can use this script as follows (requires
`curl` and `gunzip`):

**Note:** These instances require about 2GB of disk space in total.

    scripts/download-all

There is a script provided to run multiple versions of our program each with
various parameters. Outputs are shown in the terminal and also written to files
in `outputs-new` directory under the base directory. Each version and parameter
set is written to a separate file. Outputs from different instances are
separated with a trailing form feed `\f` character within a file to be used
with the rest of our scripts. You can see this script to change the versions
and parameters used in the execution. You can use this script to run all
instances in the dataset with a glob as follows:

**Note:** It can take several days for all the experiments to finish in its
default form. Also, each execution of the script appends their output to the
previous output files.

    scripts/run-all dataset/*

Outputs for the results in the paper are provided in `outputs` directory under
the base directory.

There are helpful scripts provided to filter specific values from output files
to display them in table form in the `filter` directory under the base
directory. You can use these scripts as follows (requires `python` and
`column`):

    filter/time                    outputs/pns-omp-avx512-p16-k16.txt | column -t
    filter/solution                outputs/pns-omp-avx512-p16-k16.txt | column -t
    filter/iterations              outputs/pns-omp-avx512-p16-k16.txt | column -t
    filter/distribution-time       outputs/pns-omp-avx512-p16-k16.txt | column -t
    filter/distribution-percentage outputs/pns-omp-avx512-p16-k16.txt | column -t

There is a script provided to join results from a filter with multiple output
files. You can use this script with multiple files as a glob as follows
(requires `bash`):

    filter/all filter/time outputs/lemon.txt outputs/pns-omp-avx512-p*-k16.txt | column -t

When the number of instances are few but the versions and parameters are many,
you may consider transposing the table as follows (requires `datamash`):

    filter/all filter/time outputs/lemon.txt outputs/pns-omp-avx512-p*-k16.txt | datamash transpose -W | column -t

Filters for time and solution works with the `dimacs-solver` standalone tool
that comes with LEMON library. You may add the iteration count as an output of
the expected form in this tool to make it work with iteration filter as well.
See the script sources and lemon output for more information. Also note, join
script only works with filters for time, solution and iterations, but not with
filters for distribution.

# Files

List of files and brief descriptions of subdirectories are as follows:

    .
    |-- CITATION.cff
    |-- LICENSE
    |-- Makefile
    |-- README.md
    |-- filter                         -- scripts for convenient output display
    |   |-- all
    |   |-- distribution-percentage
    |   |-- distribution-time
    |   |-- iterations
    |   |-- solution
    |   `-- time
    |-- generators                     -- patched versions of graph generators
    |   |-- gridgen-patched
    |   |   |-- Makefile
    |   |   |-- gridgen.c
    |   |   |-- gridgen.c.BAK
    |   |   |-- gridgen.c.orig
    |   |   `-- readme
    |   `-- netgen-bcjl-patched
    |       |-- README
    |       |-- README.BAK
    |       |-- index.c
    |       |-- makefile
    |       |-- makefile.orig
    |       |-- netgen.c
    |       |-- netgen.c.orig
    |       |-- netgen.h
    |       |-- netgen.h.orig
    |       |-- random.c
    |       `-- random.c.orig
    |-- outputs                        -- outputs for the results in the paper
    |   |-- lemon-cas.txt
    |   |-- lemon-cos.txt
    |   |-- lemon-ns.txt
    |   |-- pns-omp-avx2-p1-k1.txt
    |   |-- pns-omp-avx2-p1-k16.txt
    |   |-- pns-omp-avx2-p1-k4.txt
    |   |-- pns-omp-avx2-p16-k1.txt
    |   |-- pns-omp-avx2-p16-k16.txt
    |   |-- pns-omp-avx2-p16-k4.txt
    |   |-- ...
    |-- outputs-extended               -- outputs for the extended results
    |   |-- pns-omp-avx2-p32-k1.txt
    |   |-- pns-omp-avx2-p32-k16.txt
    |   |-- pns-omp-avx2-p32-k4.txt
    |   |-- pns-omp-avx2-p64-k1.txt
    |   |-- pns-omp-avx2-p64-k16.txt
    |   |-- pns-omp-avx2-p64-k4.txt
    |   |-- ...
    |-- pns.cpp
    |-- samples                        -- sample DIMACS input/output files
    |   |-- dimacs.min
    |   |-- dimacs.sol
    |   |-- figure.min
    |   `-- figure.sol
    `-- scripts                        -- scripts to prepare/run experiments
        |-- download-all
        |-- generate-all
        `-- run-all

    8 directories, 111 files
