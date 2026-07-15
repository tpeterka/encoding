#!/bin/bash

# activate the environment
export SPACKENV=mfa-env
spack env deactivate > /dev/null 2>&1
spack env activate $SPACKENV
echo "activated spack environment $SPACKENV"

echo "setting flags for building mfa"
export MFA_PATH=`spack location -i mfa`


