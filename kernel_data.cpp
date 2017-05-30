#include "kernel_data.h"

namespace RKM
{

kernel_data::kernel_data(const std::size_t _n_sample, const std::size_t _n_feature)
{
    // Memory allocation
    x.reserve(_n_sample*_n_feature);
    t.reserve(_n_sample);
}

kernel_data::~kernel_data()
{
    clear_data();
}

void kernel_data::clear_data()
{
    x.clear();
    t.clear();
}

void kernel_data::x_append(const std::vector<double> _x)
{
    for (std::size_t i=0;i<_x.size();++i)
    {
        x.push_back(_x[i]);
    }
}

void kernel_data::t_append(const double _t)
{
    t.push_back(_t);
}

double kernel_data::get_data(const std::size_t i, const std::size_t j) const
{
    return x[i*get_n_feature()+j];
}

double kernel_data::get_label(const std::size_t i) const
{
    return(t[i]);
}

//void kernel_data::set_data(const int i, const int j, const double val)
//{
//    x[i*n_feature+j] = val;
//}
//
//void kernel_data::set_label(const int i, const double val)
//{
//    t[i] = val;
//}

std::size_t kernel_data::get_n_sample() const
{
    return t.size();
}

std::size_t kernel_data::get_n_feature() const
{
    std::size_t n_feature = x.size()/t.size();
    assert(t.size()*n_feature == x.size());
    return n_feature;
}

} // namespace rkm
