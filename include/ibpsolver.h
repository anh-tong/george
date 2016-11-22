#ifndef _GEORGE_IBPSOLVER_H_
#define _GEORGE_IBPSOLVER_H_

#include <cmath>
#include <Eigen/Dense>
#include <HODLR_Tree.hpp>
#include <HODLR_Matrix.hpp>

#include "constants.h"
#include "ibpkernels.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using george::ibpkernels::IbpKernel;
using std::cout;

namespace george {

// Eigen is column major and numpy is row major. Barf.
typedef Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor> > RowMajorMap;

class IbpHODLRSolverMatrix : public HODLR_Matrix {
public:
    IbpHODLRSolverMatrix (const unsigned int row, const unsigned int col): row_(row), col_(col) {

    };

    void set_data(const double* data){
        data_ = data;
        //cout<< "Set data " << data_[0] <<"\n";
    }

    double get_Matrix_Entry (const unsigned i, const unsigned j) {
        return data_[i * row_ + j];
    };

private:
    const double* data_;
    const unsigned int row_;
    const unsigned int col_;
};

class IbpSolver {

public:

    //
    // Allocation and deallocation.
    //
    IbpSolver (const unsigned int row, const unsigned int col, unsigned nLeaf = 10, double tol = 1e-10)
        : tol_(tol), nleaf_(nLeaf), row_(row), col_(col)
    {
        matrix_ = new IbpHODLRSolverMatrix(row, col);
        solver_ = NULL;
        status_ = SOLVER_OK;
        computed_ = 0;
    };
    ~IbpSolver () {
        if (solver_ != NULL) delete solver_;
        delete matrix_;
    };

    //
    // Accessors.
    //
    int get_status () const { return status_; };
    int get_computed () const { return computed_; };
    double get_log_determinant () const { return logdet_; };

    //
    // Pre-compute and factorize the kernel matrix.
    //
    int compute ( const double* data, const double* yerr,
                 unsigned int seed) {

        // It's not computed until it's computed...
        computed_ = 0;

         // Set matrix data
        matrix_->set_data(data);
        // Compute the diagonal elements.
        VectorXd diag(row_);
        for (unsigned int i = 0; i < row_; ++i) {
            diag[i] = yerr[i]*yerr[i];
            diag[i] += matrix_->get_Matrix_Entry(i, i);
        }

        // Set up the solver object.
        if (solver_ != NULL) delete solver_;
        solver_ = new HODLR_Tree<IbpHODLRSolverMatrix> (matrix_, row_, nleaf_);
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
    IbpHODLRSolverMatrix* matrix_;
    HODLR_Tree<IbpHODLRSolverMatrix>* solver_;
    MatrixXd x_;
    unsigned int row_;
    unsigned int col_;
};

};

#endif
