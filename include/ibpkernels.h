#ifndef _GEORGE_IBPKERNELS_H_
#define _GEORGE_IBPKERNELS_H_

#include <cmath>
#include <cfloat>
#include <vector>
#include <iostream>

using std::vector;
using std::cout;

namespace george {
namespace ibpkernels {


class IbpKernel {
public:
    IbpKernel (const double* data, const unsigned int row, const unsigned int col) : data_(data), row_(row), col_(col) {};
    ~IbpKernel () {
        delete data_;
    };
    double value (const int i, const int j) const {
        return data_[i*row_ + j];
    };

protected:
    unsigned int row_;
    unsigned int col_;
    const double* data_;
};
}; // namespace kernels
}; // namespace george

#endif
