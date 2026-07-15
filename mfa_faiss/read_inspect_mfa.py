# reads an MFA file in Python and demonstrates access to a few model items
# assumes that a 3-d domain with one scalar science variable was previously modeled and saved

import diy
import mfa
import numpy as np
import argparse

if __name__ == "__main__":

    # parse args to get input file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', dest='infile')
    config = parser.parse_args()
    infile = config.infile

    # MPI, DIY world and master
    w = diy.mpi.MPIComm()           # world
    m = diy.Master(w)               # master

    # load the results and print them out
    print("\n\nLoading blocks and printing them out\n")
    a = diy.ContiguousAssigner(w.size, -1)
    diy.read_blocks(infile, a, m, load = mfa.load_block)
    m.foreach(lambda b,cp: b.print_block(cp, False))

    # test evaluating a point
    param   = np.array([0.5, 0.5, 0.5], dtype=np.float64)       # input parameters where to decode the point
    pt      = np.zeros(4, dtype=np.float64)                     # assigning fake values defines shape and type
    m.foreach(lambda b, cp: b.decode_point(cp, param, pt))
    print("\nThe point at [u = ", param[0], ", v = ", param[1], "w = ", param[2], "] =", pt)

    # access geometry control points
    m.foreach(lambda b, cp: print(type(b.mfa_model().geom().tmesh.tensor_prods[0].ctrl_pts)))
    m.foreach(lambda b, cp: print(np.asarray(b.mfa_model().geom().tmesh.tensor_prods[0].ctrl_pts).shape))
    m.foreach(lambda b, cp: print(np.asarray(b.mfa_model().geom().tmesh.tensor_prods[0].ctrl_pts)))

    # access science variable control points
    # assumes only one tensor product (fixed, not adaptive encoding)
    m.foreach(lambda b, cp: print(type(b.mfa_model().var(0).tmesh.tensor_prods[0].ctrl_pts)))
    m.foreach(lambda b, cp: print(np.asarray(b.mfa_model().var(0).tmesh.tensor_prods[0].ctrl_pts).shape))
    m.foreach(lambda b, cp: print(np.asarray(b.mfa_model().var(0).tmesh.tensor_prods[0].ctrl_pts[:10])))
