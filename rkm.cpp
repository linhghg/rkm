#include "rkm.h"

namespace RKM
{

    rkm::rkm(const std::string& input_file_name)
    {
        // all kinds of settings to be read
        std::string task_name;
        std::string kernel_name;
        size_t n_feature = 0;
        size_t n_sample = 0;
        double gamma;

        std::string str;
        // input file
        std::ifstream fin (input_file_name, std::ios_base::in);
        while (fin>>str)
        {
            fin>>delim(':');
            if (str.compare("train_file") == 0)
            {
                fin>>train_file;
            }
            else if (str.compare("test_file") == 0)
            {
                fin>>test_file;
            }
            else if (str.compare("verbose") == 0)
            {
                fin>>verbose;
            }
            else if (str.compare("task") == 0)
            {
                fin>>task_name;
            }
            else if (str.compare("Cp") == 0)
            {
                fin>>Cp;
            }
            else if (str.compare("Cn") == 0)
            {
                fin>>Cn;
            }
            else if (str.compare("eps") == 0)
            {
                fin>>eps;
            }
            else if (str.compare("kernel") == 0)
            {
                fin>>kernel_name;
            }
            else if (str.compare("gamma") == 0)
            {
                fin>>gamma;
            }
            else if (str.compare("tau") == 0)
            {
                fin>>tau;
            }
            else
            {
                fin>>str;
            }
        } // while (fin>>str)
        fin.close();

        if (verbose)
        {
            std::cout<<"========================\n";
            std::cout<<"Task: "<<task_name<<"...\n";
            std::cout<<"========================\n";
            std::cout<<"Reading training data file: "<<train_file<<"...\n";
        }

        // training data file
        fin.open(train_file, std::ios_base::in);
        // calc n_sample and n_feature
        while (std::getline(fin, str))
        {
            n_sample++;
            if (n_feature == 0)
            {
                std::stringstream ss(str);
                double temp;
                ss>>temp>>delim(':');
                while (ss>>temp)
                {
                    n_feature++;
                }
            }
        }
        fin.close();

        if (verbose)
        {
            std::cout<<"Found "<<n_sample<<" samples with "<<n_feature<<" features.\n";
            std::cout<<"Loading data to solver...\n";
        }

        // vector size allocation
        kd = new kernel_data(n_sample, n_feature);
        // set parameters
        kd -> set_kernel(kernel_name);
        kd -> set_gamma(gamma);

        // read data into the kernel_data
        fin.open(train_file, std::ios_base::in);
        std::vector<double> x(n_feature);
        double t;
        for (size_t i=0;i<n_sample;++i)
        {
            fin>>t>>delim(':');
            kd->t_append(t);
            for (size_t j=0;j<n_feature;++j)
            {
                fin>>x[j];
            }
            kd->x_append(x);
        }
        fin.close();

        if (verbose)
        {
            std::cout<<"Feature scaling...\n";
        }
        // Normalization
        kd->scale_features();

    } //rkm::rkm(const std::string& input_file_name)

    rkm::~rkm()
    {
        if (kd)
        {
            delete kd;
        }
    }

    bool rkm::is_in_I_up(size_t i) const
    {
        double yi = kd->get_label(i);
        if ( (yi > 0) && (alpha[i] < get_C(i)) )
            return true;
        else if ( (yi < 0) && (alpha[i] > 0) )
            return true;
        else
            return false;
    }

    bool rkm::is_in_I_low(size_t i) const
    {
        double yi = kd->get_label(i);
        if ( (yi < 0) && (alpha[i] < get_C(i)) )
            return true;
        else if ( (yi > 0) && (alpha[i] > 0) )
            return true;
        else
            return false;
    }

    double rkm::K(size_t i, size_t j) const
    {
        return kd->K(i, j, beta);
    }

    void rkm::solve()
    {
        size_t n_feature = kd->get_n_feature();
        size_t n_sample = kd->get_n_sample();

        // TBD
        // Initialization
        beta.resize(n_feature, 1.0);
        beta[1] = 0.0;
        beta[3] = 0.0;

        alpha.resize(n_sample, 0.0);
        alpha_status.resize(n_sample);
        for (size_t i=0;i<n_sample;++i)
        {
            update_alpha_status(i);
        }

        // TBD: shrinking heuristic

        // Gradient init
        std::vector<double> G(n_sample);
        std::vector<double> G_bar(n_sample);
        for (size_t i=0;i<n_sample;++i)
        {
            G[i] = -1;
            //G_bar[i] = 0;
        }
        //for (size_t i=0;i<n_sample;++i)
        //{
        //    if (!is_lower_bound(i))
        //    {
        //        for (size_t j=0;j<n_sample;++j)
        //        {
        //            G[i] += alpha[i]*K(i, j); // TBD: yi, yj
        //        }
        //        if (is_upper_bound(i))
        //        {
        //            for (size_t j=0;j<n_sample;++j)
        //            {
        //                G_bar[i] += get_C(i)*K(i, j);
        //            }
        //        }
        //    }
        //}

        // optimization step

        int iter = 0;
        int max_iter = n_sample*100;
        if (n_sample>std::numeric_limits<int>::max()/100)
        {
            max_iter = std::numeric_limits<int>::max();
        }
        if (max_iter < 10000000) max_iter = 10000000;
        int counter = min(n_sample,1000)+1;

        while(iter < max_iter)
        {
            // show progress, shrinking TBD
            if(--counter == 0)
            {
                counter = min(n_sample,1000);
                //if(shrinking) do_shrinking();
                std::cout<<".";
            }

            size_t i, j;
            if (!select_working_set(i, j))
                break;

            ++iter;

            // update alpha[i] and alpha[j], handle bounds carefully
            double C_i = get_C(i);
            double C_j = get_C(j);
            double y_i = kd->get_label(i);
            double y_j = kd->get_label(j);
            double old_alpha_i = alpha[i];
            double old_alpha_j = alpha[j];

            double a = K(i, i) + K(j, j) - 2*K(i, j);
            if (a <= 0)
                a = tau;
            double b = - y_i*G[i] + y_j*G[j];
            alpha[i] += y_i*b/a;
            alpha[j] -= y_j*b/a;
        } // while(iter < max_iter)

    } // void rkm::solve()

    bool rkm::select_working_set(size_t& i, size_t& j) const
    {
        // return i,j such that
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        size_t n_feature = kd->get_n_feature();
        double Gmax = -std::numeric_limits<double>::infinity();
        double Gmin = std::numeric_limits<double>::infinity();
        size_t i_idx = -1;
        size_t j_idx = -1;

        // select i
        for (size_t idx=0;idx<n_sample;++idx)
        {
            if (is_in_I_up(idx))
            {
                double obj_i = - kd->get_label(idx)*G[idx];
                if (obj_i >= Gmax)
                {
                    i_idx = idx;
                    Gmax = obj_i;
                }
            }
        }

        // select j
        double obj_j_min = std::numeric_limits<double>::infinity();
        for (size_t idx=0;idx<n_sample;++idx)
        {
            if (is_in_I_low(idx))
            {
                double tmp = - kd->get_label(idx)*G[idx];
                double b = Gmax - tmp;
                if (tmp <= Gmin)
                    Gmin = tmp;
                if (b > 0)
                {
                    double a = K(i_idx, i_idx) + K(idx, idx) - 2*K(i_idx, idx);
                    if (a <= 0)
                        a = tau;
                    double obj_j = -(b*b)/a;
                    if (obj_j <= obj_j_min)
                    {
                        obj_j_min = obj_j;
                        j_idx = idx;
                    }
                }
            }
        }

        if(Gmax-Gmin < eps || i_idx == -1 || j_idx == -1)
            return false;

        i = i_idx;
        j = j_idx;
        return true;
    } // int rkm::select_working_set(size_t& i, size_t& j) const

    double rkm::get_C(size_t i) const
    {
        return ((kd->get_label(i)>0)? Cp : Cn);
    }

    void rkm::update_alpha_status(size_t i)
    {
        if (alpha[i] >= get_C(i))
            alpha_status[i] = UPPER_BOUND;
        else if(alpha[i] <= 0)
            alpha_status[i] = LOWER_BOUND;
        else alpha_status[i] = FREE;
    }

} // namespace rkm
