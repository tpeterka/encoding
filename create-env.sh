#!/bin/bash

export SPACKENV=mfa-env
export YAML=$PWD/env.yaml

# create spack environment
echo "creating spack environment $SPACKENV"
spack env deactivate > /dev/null 2>&1
spack env remove -y $SPACKENV > /dev/null 2>&1
spack env create $SPACKENV $YAML

# activate environment
echo "activating spack environment"
spack env activate $SPACKENV

spack add mpich@4
# spack add mfa+python thread=tbb tests=false
spack add mfa@local+python thread=tbb tests=false build_type='Debug'

# install everything in environment
echo "installing dependencies in environment"
spack install

# deactivate environment
spack env deactivate


