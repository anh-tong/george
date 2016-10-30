#ifndef _GEORGE_LATENTSOLVER_H_
#define _GEORGE_LATENTSOLVER_H_

#include <cmath>
#include <Eigen/Dense>
#include <HODLR_Tree.hpp>
#include <HODLR_Matrix.hpp>
#include <iostream>

#include "constants.h"
#include "kernels.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using george::kernels::LatentModelKernel;
using std::cout;

namespace george {

// Eigen is column major and numpy is row major. Barf.
typedef Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor> > RowMajorMap;

class HODLRLatentSolverMatrix : public HODLR_Matrix {
public:
    HODLRLatentSolverMatrix (LatentModelKernel* kernel)
        : kernel_(kernel)
    {
        stride_ = kernel_->get_ndim();
    };
    void set_values (const double* v) {
        t_ = v;
    };

    void set_ZTZ(double* ztz){
        kernel_->set_ZTZ(ztz);
    }

    void add_inversed(double* inv){
        kernel_->add_inversed(inv);
    }

    void reset_inversed(){
        kernel_->reset_inversed();
    }

    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        int d1 = i % kernel_->get_K();
        int d2 = j % kernel_->get_K();
        double v = kernel_->value(&(t_[d1*stride_]), &(t_[d2*stride_]), i, j);
        return kernel_->value(&(t_[d1*stride_]), &(t_[d2*stride_]), i, j);
    };

private:
    LatentModelKernel* kernel_;
    unsigned int stride_;
    const double* t_;
};

class LatentSolver {

public:

    //
    // Allocation and deallocation.
    //
    LatentSolver (LatentModelKernel* kernel, unsigned nLeaf = 10, double tol = 1e-10)
        : tol_(tol), nleaf_(nLeaf), kernel_(kernel)
    {
        matrix_ = new HODLRLatentSolverMatrix(kernel_);
        solver_ = NULL;
        status_ = SOLVER_OK;
        computed_ = 0;
    };
    ~LatentSolver () {
        if (solver_ != NULL) delete solver_;
        delete matrix_;
    };


    void set_ZTZ(double* ztz){
        matrix_->set_ZTZ(ztz);
    }

    void add_inversed(double* inv){
        matrix_->add_inversed(inv);
    }

    void reset_inversed(){
        matrix_->reset_inversed();
    }

    //
    // Accessors.
    //
    int get_status () const { return status_; };
    int get_computed () const { return computed_; };
    double get_log_determinant () const { return logdet_; };

    //
    // Pre-compute and factorize the kernel matrix.
    //
    int compute (const unsigned int n, const double* x, const double* yerr,
                 unsigned int seed) {
        unsigned int ndim = kernel_->get_ndim();

        // It's not computed until it's computed...
        computed_ = 0;

        // Compute the diagonal elements.
        VectorXd diag(n);
        for (unsigned int i = 0; i < n; ++i) {
            diag[i] = yerr[i]*yerr[i];
            int d = i % kernel_->get_K();
            diag[i] += kernel_->value(&(x[d*ndim]), &(x[d*ndim]), i, i);
        }

        // Set the time points for the kernel.
        matrix_->set_values(x);

        // Set up the solver object.
        if (solver_ != NULL) delete solver_;
        solver_ = new HODLR_Tree<HODLRLatentSolverMatrix> (matrix_, n, nleaf_);
        solver_->assemble_Matrix(diag, tol_, 's', seed);

        // Factorize the matrix.
        solver_->compute_Factor();

        // Extract the log-determinant.
        solver_->compute_Determinant(logdet_);

        // Update the bookkeeping flags.
        computed_ = 1;
        status_ = SOLVER_OK;
        return status_;
    };

    void apply_inverse (const unsigned int n, const unsigned int nrhs,
                        double* b, double* out) {
        unsigned int i, j;
        MatrixXd b_vec = RowMajorMap(b, n, nrhs), alpha(n, nrhs);
        solver_->solve(b_vec, alpha);
        for (i = 0; i < n; ++i)
            for (j = 0; j < nrhs; ++j)
                out[i*nrhs+j] = alpha(i, j);
    };

private:
    double logdet_, tol_;
    unsigned nleaf_;
    int status_, computed_;
    LatentModelKernel* kernel_;
    HODLRLatentSolverMatrix* matrix_;
    HODLR_Tree<HODLRLatentSolverMatrix>* solver_;
    MatrixXd x_;
};

};

#endif
