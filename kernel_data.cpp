#include "kernel_data.h"

namespace RKM
{

kernel_data::kernel_data(const size_t _n_sample, const size_t _n_feature)
{
    n_sample = _n_sample;
    n_feature = _n_feature;
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
    for (size_t i=0;i<_x.size();++i)
    {
        x.push_back(_x[i]);
    }
}

void kernel_data::t_append(const double _t)
{
    t.push_back(_t);
}

double kernel_data::get_data(const size_t i, const size_t j) const
{
    return x[i*n_feature+j];
}

double kernel_data::get_label(const size_t i) const
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

size_t kernel_data::get_n_sample() const
{
    //return t.size();
    return n_sample;
}

size_t kernel_data::get_n_feature() const
{
    //size_t n_feature = x.size()/t.size();
    //assert(t.size()*n_feature == x.size());
    return n_feature;
}

void kernel_data::scale_features()
{
    // Initialize max_x and min_x
    max_x.resize(n_feature, -1.0*std::numeric_limits<double>::infinity());
    min_x.resize(n_feature, +1.0*std::numeric_limits<double>::infinity());
    // Traversing the data to collect min and max
    for (size_t i=0;i<n_sample;++i)
    {
        for (size_t j=0;j<n_feature;++j)
        {
            size_t idx = i*n_feature+j;
            if (x[idx] < min_x[j])
            {
                min_x[j] = x[idx];
            }
            if (x[idx] > max_x[j])
            {
                max_x[j] = x[idx];
            }
        }
    }

    // check max_x - min_x > 0
    //for (size_t j=0;j<n_feature;++j)
    //{
    //    assert(max_x[j]>min_x[j]);
    //}

    // Scale the data
    // Normalization: x' = (x-min)/(max-min)
    for (size_t i=0;i<n_sample;++i)
    {
        for (size_t j=0;j<n_feature;++j)
        {
            size_t idx = i*n_feature+j;
            if (max_x[j] - min_x[j] == 0)
            {
                x[idx] = x[idx] - min_x[j];
            }
            else
            {
                x[idx] = (x[idx] - min_x[j]) / (max_x[j] - min_x[j]);
            }
            assert(x[idx] == x[idx]); // NaN check
        }
    }
} //void kernel_data::scale_features()

void kernel_data::scale_one_vector(std::vector<double>& x) const
{
    assert(x.size() == n_feature);
    for (size_t j=0;j<n_feature;++j)
    {
        if (max_x[j] - min_x[j] == 0)
        {
            x[j] = x[j] - min_x[j];
        }
        else
        {
            x[j] = (x[j] - min_x[j]) / (max_x[j] - min_x[j]);
        }
        assert(x[j] == x[j]); // NaN check
    }
}

void kernel_data::print_data(std::ostream& os) const
{
    for (size_t i=0;i<t.size();++i)
    {
        os<<get_label(i)<<" :";
        for (size_t j=0;j<n_feature;++j)
        {
            os<<" "<<get_data(i, j);
        }
        os<<"\n";
    }
}

void kernel_data::print_kernel(std::ostream& os) const
{
    os<<"Kernel type: "<<kern_type<<"\n";
    os<<"Gamma: "<<gamma<<"\n";
}

void kernel_data::set_kernel(const std::string& kernel_name)
{
    if (kernel_name.compare("Gaussian") == 0)
    {
        kern_type = GAUSSIAN;
    }
    else if (kernel_name.compare("Fast_Gaussian") == 0)
    {
        kern_type = FAST_GAUSSIAN;
    }
    else if (kernel_name.compare("Lookup_Gaussian") == 0)
    {
        kern_type = LOOKUP_GAUSSIAN;
        init_gaussian_loopup_table();
    }
}

void kernel_data::set_gamma(double _gamma)
{
    gamma = _gamma;
}

double kernel_data::kernel_one(size_t i, size_t j, size_t k) const
{
    return kernel_one(x[i*n_feature+k], x[j*n_feature+k]);
}

double kernel_data::kernel_one(double x_ik, double x_jk) const
{
    double res = 0;
    switch(kern_type)
    {
        case LOOKUP_GAUSSIAN:
        {
            double d = x_ik - x_jk;
            if (d<0) d = -d;
            size_t idx = (size_t)(d*tab_size);
            double ratio = d*tab_size - idx;
            if (ratio == 0)
            {
                res = lookup_tab[idx];
            }
            else
            {
                res = lookup_tab[idx]*(1-ratio) + lookup_tab[idx+1]*ratio;
            }
            break;
        }
        case GAUSSIAN:
        {
            double d = x_ik - x_jk;
            res = exp(-gamma*d*d);
            break;
        }
        case FAST_GAUSSIAN:
        {
            double d = x_ik - x_jk;
            double tmp = 1-gamma*d*d/16;  //1+x/n
            tmp *= tmp; // ^2
            tmp *= tmp; // ^4
            tmp *= tmp; // ^8
            tmp *= tmp; // ^16
            res = tmp;
            break;
        }
        case LINEAR:
        {
            res = x_ik * x_jk;
            break;
        }
    }
    return res;
}

double kernel_data::K(size_t i, size_t j, const std::vector<double>& v) const
{
    double res = 0;
    for (size_t k=0;k<n_feature;++k)
    {
        if (v[k]!=0)
            res += v[k]*kernel_one(x[i*n_feature+k], x[j*n_feature+k]);
    }
    return res;
}

double kernel_data::K(size_t i, const std::vector<double>& x_j, const std::vector<double>& v) const
{
    assert(x_j.size() == n_feature);
    double res = 0;
    for (size_t k=0;k<n_feature;++k)
    {
        if (v[k]!=0)
            res += v[k]*kernel_one(x[i*n_feature+k], x_j[k]);
    }
    return res;
}

const size_t kernel_data::tab_size = 10000;

void kernel_data::init_gaussian_loopup_table()
{
    lookup_tab = std::vector<double>(tab_size+1);
    for (size_t i=0;i<tab_size+1;++i)
    {
        double d = ((double)i)/tab_size;
        lookup_tab[i] = exp(-gamma*d*d);
    }
}

} // namespace rkm
