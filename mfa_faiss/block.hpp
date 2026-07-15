//--------------------------------------------------------------
// one diy block
//
// Tom Peterka
// Argonne National Laboratory
// tpeterka@mcs.anl.gov
//--------------------------------------------------------------
#ifndef _MFA_BLOCK
#define _MFA_BLOCK

#include    <random>
#include    <stdio.h>
#include    <mfa/types.hpp>
#include    <mfa/mfa.hpp>
#include    <mfa/block_base.hpp>
#include    <diy/master.hpp>
#include    <diy/reduce-operations.hpp>
#include    <diy/decomposition.hpp>
#include    <diy/assigner.hpp>
#include    <diy/io/block.hpp>
#include    <diy/io/bov.hpp>
#include    <diy/pick.hpp>
#include    <Eigen/Dense>

#include    "domain_args.hpp"

// 3d point or vector
struct vec3d
{
    float x, y, z;
    float mag() { return sqrt(x*x + y*y + z*z); }
    vec3d(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    vec3d() {}
};



// block
template <typename T, typename U=T>
struct Block : public BlockBase<T, U>
{
    using Base = BlockBase<T, U>;
    using Base::dom_dim;
    using Base::pt_dim;
    using Base::core;
    using Base::bounds;
    using Base::domain;
    using Base::core_mins;
    using Base::core_maxs;
    using Base::bounds_mins;
    using Base::bounds_maxs;
    using Base::overlaps;
    using Base::input;
    using Base::approx;
    using Base::errs;
    using Base::mfa;
    using Base::verbose;

    static
        void* create()              { return mfa::create<Block>(); }

    static
        void destroy(void* b)       { mfa::destroy<Block>(b); }

    static
        void add(                                   // add the block to the decomposition
            int                 gid,                // block global id
            const Bounds<T>&    core,               // block bounds without any ghost added
            const Bounds<T>&    bounds,             // block bounds including any ghost region added
            const Bounds<T>&    domain,             // global data bounds
            const RCLink<T>&    link,               // neighborhood
            diy::Master&        master,             // diy master
            int                 dom_dim,            // domain dimensionality
            int                 pt_dim,             // point dimensionality
            T                   ghost_factor = 0.0) // amount of ghost zone overlap as a factor of block size (0.0 - 1.0)
    {
        mfa::add<Block, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim, ghost_factor);
    }

    static
        void add_int(                                   // add the block to the decomposition
            int                 gid,                // block global id
            const Bounds<int>&  core,               // block bounds without any ghost added
            const Bounds<int>&  bounds,             // block bounds including any ghost region added
            const Bounds<int>&  domain,             // global data bounds
            const RCLink<int>&  link,               // neighborhood
            diy::Master&        master,             // diy master
            int                 dom_dim,            // domain dimensionality
            int                 pt_dim)             // point dimensionality
    {
        mfa::add_int<Block, T>(gid, core, bounds, domain, link, master, dom_dim, pt_dim);
    }

    static
        void save(const void* b_, diy::BinaryBuffer& bb)    { mfa::save<Block, T>(b_, bb); }
    static
        void load(void* b_, diy::BinaryBuffer& bb)          { mfa::load<Block, T>(b_, bb); }

    // NB: Because BlockBase, the parent of Block, is templated, the C++ compiler requires
    // access to members in BlockBase to be preceded by "this->".
    // Otherwise, the compiler can't be sure that the member exists. [Myers Effective C++, item 43]
    // This is annoying but unavoidable.

    // debug: print control points in a block
    void print_ctrl(const diy::Master::ProxyWithLink& cp)
    {
        cerr << "\n----- science variable models -----" << endl;
        for (int i = 0; i < mfa->nvars(); i++)
        {
            if (mfa->var(i).tmesh.tensor_prods.size() == 1)
                print_ctrl_pts(this->mfa->var(i).tmesh);
        }
    }

    //  debug: print control points and weights in all tensor products of a tmesh
    void print_ctrl_pts(const mfa::Tmesh<T>& tmesh)
    {
        for (auto i = 0; i < tmesh.tensor_prods.size(); i++)
        {
            fmt::print(stderr, "tensor_prods[{}]:\n", i);
            fmt::print(stderr, "nctrl_pts [{}]\n", tmesh.tensor_prods[i].nctrl_pts.transpose());
            fmt::print(stderr, "ctrl_pts\n{}\n", tmesh.tensor_prods[i].ctrl_pts.topRows(10));
            fmt::print(stderr, "weights\n{}\n", tmesh.tensor_prods[i].weights.topRows(10));
        }
    }

    // read a floating point 3d coordinates plus scalar dataset
    // f = (x, y, z, val)
    void read_3d_coords_and_scalar_data(
            const               diy::Master::ProxyWithLink& cp,
            mfa::MFAInfo&       mfa_info,
            DomainArgs&         args,
            bool                swap_order = false)
    {
        assert(mfa_info.dom_dim == dom_dim);
        assert(mfa_info.dom_dim == 3);
        assert(mfa_info.pt_dim() == pt_dim);
        assert(mfa_info.nvars() == 1);
        assert(mfa_info.geom_dim() == 3);
        assert(mfa_info.var_dim(0) == 1);

        const int nvars         = mfa_info.nvars();
        const int gdim          = mfa_info.geom_dim();
        const VectorXi mdims    = mfa_info.model_dims();

        DomainArgs* a = &args;
        int tot_ndom_pts = 1;
        this->max_errs.resize(nvars);
        this->sum_sq_errs.resize(nvars);
        VectorXi ndom_pts(dom_dim);
        this->bounds_mins.resize(pt_dim);
        this->bounds_maxs.resize(pt_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            ndom_pts(i)                     =  a->ndom_pts[i];
            tot_ndom_pts                    *= ndom_pts(i);
        }

        // Construct point set to contain input
        if (args.structured)
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts, ndom_pts);
        else
            input = new mfa::PointSet<T>(dom_dim, mdims, tot_ndom_pts);

        vector<float> pt(4 * tot_ndom_pts);

        FILE *fd = fopen(a->infile.c_str(), "r");
        assert(fd);

        // read all four components of point (x,y,z,value)
        if (fread(&pt[0], sizeof(float), tot_ndom_pts * 4, fd) != tot_ndom_pts * 4)
        {
            fprintf(stderr, "Error: unable to read file\n");
            exit(0);
        }

        // store the points in the input domain array
        // MFA expects the order to be x (first coordinate) changing fastest
        // if this is opposite of the file order, we swap the order here
        int nx = ndom_pts(0);
        int ny = ndom_pts(1);
        int nz = ndom_pts(2);
        size_t src, dst;

        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int k = 0; k < nz; k++)
                {
                    if (swap_order)
                        src = k + nz * (j + ny * i);         // file: last dim fastest
                    else
                        src = i + nx * (j + ny * k);         // file: first dim fastest
                    dst = i + nx * (j + ny * k);             // MFA:  first dim fastest

                    input->domain(dst, 0) = pt[4 * src    ];
                    input->domain(dst, 1) = pt[4 * src + 1];
                    input->domain(dst, 2) = pt[4 * src + 2];
                    input->domain(dst, 3) = pt[4 * src + 3];
                }
            }
        }

        // debug
//         fmt::print(stderr, "domain row 0 {} row xmax {} row ymax {} row zmax {}\n",
//                 input->domain.row(0),
//                 input->domain.row(nx - 1),
//                 input->domain.row(nx * (ny - 1)),
//                 input->domain.row(nx * ny * (nz - 1)));

        // find extent of coordinates and value
        for (size_t i = 0; i < (size_t)input->domain.rows(); i++)
        {
            for (auto j = 0; j < pt_dim; j++)
            {
                if (i == 0 || input->domain(i, j) < bounds_mins(j))
                    bounds_mins(j) = input->domain(i, j);
                if (i == 0 || input->domain(i, j) > bounds_maxs(j))
                    bounds_maxs(j) = input->domain(i, j);
            }
        }

        core_mins.resize(dom_dim);
        core_maxs.resize(dom_dim);
        for (int i = 0; i < dom_dim; i++)
        {
            core_mins(i) = bounds_mins(i);
            core_maxs(i) = bounds_maxs(i);
        }

        input->set_domain_params();

        // initialize MFA models (geometry, vars, etc)
        this->setup_MFA(cp, mfa_info);

        // debug
        cerr << "domain extent:\n min\n" << bounds_mins << "\nmax\n" << bounds_maxs << endl;
    }

};

template<typename U>
void max_err_cb(Block<real_t, U> *b,                  // local block
        const diy::ReduceProxy &rp,                // communication proxy
        const diy::RegularMergePartners &partners) // partners of the current block
{
    unsigned round = rp.round();    // current round number

    // step 1: dequeue and merge
    for (int i = 0; i < rp.in_link().size(); ++i) {
        int nbr_gid = rp.in_link().target(i).gid;
        if (nbr_gid == rp.gid()) {

            continue;
        }

        std::vector<real_t> in_vals;
        rp.dequeue(nbr_gid, in_vals);

        for (size_t j = 0; j < in_vals.size() / 2; ++j) {
            if (b->max_errs_reduce[2 * j] < in_vals[2 * j]) {
                b->max_errs_reduce[2 * j] = in_vals[2 * j];
                b->max_errs_reduce[2 * j + 1] = in_vals[2 * j + 1]; // received from this block
            }
        }
    }

    // step 2: enqueue
    for (int i = 0; i < rp.out_link().size(); ++i) // redundant since size should equal to 1
    {
        // only send to root of group, but not self
        if (rp.out_link().target(i).gid != rp.gid()) {
            rp.enqueue(rp.out_link().target(i), b->max_errs_reduce);
        } //else
        //fmt::print(stderr, "[{}:{}] Skipping sending to self\n", rp.gid(), round);
    }
}


#endif // _MFA_BLOCK
