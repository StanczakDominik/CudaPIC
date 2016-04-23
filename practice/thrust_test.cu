#include <thrust/sort.h>
#include <stdio.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>

int main()
{
    thrust::device_vector<int> A(3);
    thrust::device_vector<char> B(3);
    A[0] = 10; A[1] = 20; A[2] = 30;
    B[0] = 'x'; B[1] = 'y'; B[2] = 'z';

    thrust::zip_iterator first = thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin());
    thrust::zip_iterator last = thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end());

    thrust::maximum< tuple<int, char> > binary_op;
    thrust::tuple<int, char> init = first[0];
    cout << thrust::reduce(first, last, init, binary_op) << endl;
}
