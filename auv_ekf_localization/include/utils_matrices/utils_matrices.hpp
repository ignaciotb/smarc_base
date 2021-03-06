#ifndef UTILS_MATRICES_HPP
#define UTILS_MATRICES_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/operation_blocked.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include <boost/scoped_ptr.hpp>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/inverse_chi_squared.hpp>

namespace matrices{

    template<typename T>
    T matDeterminant(const boost::numeric::ublas::matrix<T>& mat_A){
        using namespace boost::numeric::ublas;
        matrix<T> mLu(mat_A);
        permutation_matrix<std::size_t> pivots(mat_A.size1());

        auto isSingular = lu_factorize(mLu, pivots);
        if (isSingular){
            return static_cast<T>(0);
        }

        T det = static_cast<T>(1);
        for (std::size_t i = 0; i < pivots.size(); ++i){
            if (pivots(i) != i){
                det *= static_cast<T>(-1);
            }
            det *= mLu(i, i);
        }
        return det;
    }

    template<typename T>
    bool InvertMatrix (const boost::numeric::ublas::matrix<T>& input, boost::numeric::ublas::matrix<T>& inverse) {
        using namespace boost::numeric::ublas;
        matrix<T> A(input);
        // Perform LU-factorization
        permutation_matrix<std::size_t> pm(A.size1());
        int res = lu_factorize(A,pm);
        if( res != 0 )
            return false;
        inverse.assign(identity_matrix<T>(A.size1()));
        lu_substitute(A, pm, inverse);
        return true;
    }

    template<typename T>
    boost::numeric::ublas::matrix<T> Cholesky(const boost::numeric::ublas::matrix<T>& mat_A){
        // TODO_NACHO: check for matrix conditions to use cholesky
        int n = mat_A.size1();
        boost::numeric::ublas::matrix<T> chol_triang(n, n);
        for(unsigned int i=0; i< mat_A.size1(); i++){
            for(unsigned int j=0; j< i + 1; j++){
                double s = 0;
                for(unsigned int k = 0; k<j; k++){
                    s += chol_triang(i * n + k) * chol_triang(j * n + k);
                }
                chol_triang(i * n + j) = (i = j)?
                            std::sqrt(mat_A(i * n + i) - s):
                            (1.0 / chol_triang(j * n + j) * (mat_A(i * n + j) - s));
            }
        }
        return chol_triang;
    }

    template<typename T>
    boost::numeric::ublas::matrix<T> matTriangDeterminant(const boost::numeric::ublas::matrix<T>& mat_A){
        int n = mat_A.size1();
        T det;
        for(unsigned int i=0; i< mat_A.size1(); i++){
            for(unsigned int j=0; j< mat_A.size2(); j++){
                det *= (i == j)? mat_A(i,j): 1;
            }
        }
        return det;
    }
}

#endif // UTILS_MATRICES_HPP
