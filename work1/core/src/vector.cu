#include "data.cuh"
#include "vector.cuh"
#include "vector_view.cuh"

template class Data<float>;
template class VectorView<float>;
template class Vector<float>;

template class Data<double>;
template class VectorView<double>;
template class Vector<double>;