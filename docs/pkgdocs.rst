
================
Package Metadata
================

- **classifier**:: 

    Intended Audience :: Science/Research
    Topic :: Scientific/Engineering

- **description-file:** README.txt

- **entry_points**:: 

    [openmdao.variable]
    SU2_wrapper.SU2_wrapper.ConfigVar=SU2_wrapper.SU2_wrapper:ConfigVar
    [openmdao.component]
    SU2_wrapper.opt_NACA.OptModel=SU2_wrapper.opt_NACA:OptModel
    SU2_wrapper.SU2_wrapper.Solve=SU2_wrapper.SU2_wrapper:Solve
    SU2_wrapper.SU2_wrapper.Deform=SU2_wrapper.SU2_wrapper:Deform
    [openmdao.container]
    SU2_wrapper.opt_NACA.OptModel=SU2_wrapper.opt_NACA:OptModel
    SU2_wrapper.SU2_wrapper.Solve=SU2_wrapper.SU2_wrapper:Solve
    SU2_wrapper.SU2_wrapper.Deform=SU2_wrapper.SU2_wrapper:Deform

- **keywords:** openmdao

- **name:** SU2_wrapper

- **requires-dist:** openmdao.main

- **requires-python**:: 

    >=2.6
    <3.0

- **static_path:** [ '_static' ]

- **version:** 0.2

