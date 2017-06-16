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
            else if (str.compare("model_file") == 0)
            {
                fin>>model_file;
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

    double rkm::Y(size_t i) const
    {
        return kd->get_label(i);
    }

    double rkm::Q(size_t i, size_t j, size_t k) const
    {
        return (kd->kernel_one(i, j, k))*(kd->get_label(i))*(kd->get_label(j));
    }

    void rkm::solve()
    {
        double Cb = 1;
        if (verbose)
        {
            std::cout<<"Training started...\n";
        }
        size_t n_feature = kd->get_n_feature();
        size_t n_sample = kd->get_n_sample();

        // TBD
        // Initialization
        beta.resize(n_feature, 1.0);

        alpha.resize(n_sample, 0.0);
        alpha_status.resize(n_sample);
        for (size_t i=0;i<n_sample;++i)
        {
            update_alpha_status(i);
        }

        // TBD: shrinking heuristic

        // Gradient init
        Ga.resize(n_sample, -1.0);
        Gb.resize(n_feature, 0.0);
        Q_alpha.reserve(n_sample*n_feature);
        Q_alpha.resize(n_sample*n_feature, 0.0);
        //std::vector<double> G_bar(n_sample);
        //for (size_t i=0;i<n_sample;++i)
        //{
        //    G[i] = -1;
        //    G_bar[i] = 0;
        //}
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

        size_t iter = 0;
        size_t max_iter = n_sample*100;
        if (n_sample>std::numeric_limits<size_t>::max()/100)
        {
            max_iter = std::numeric_limits<size_t>::max();
        }
        if (max_iter < 10000000) max_iter = 10000000;
        size_t counter = (n_sample>1000 ? 1000 : n_sample)+1;

        while(iter < max_iter)
        {
            // show progress, shrinking TBD
            if(--counter == 0)
            {
                counter = (n_sample>1000 ? 1000 : n_sample);
                //if(shrinking) do_shrinking();
                std::cout<<".";
            }

            size_t i, j, k;
            if (!select_working_set(i, j))
                break;

            ++iter;

            for (size_t idx_k=0;idx_k<n_feature;++idx_k)
            {
                beta[idx_k] = 0;
                for (size_t idx_i=0;idx_i<n_sample;++idx_i)
                {
                    beta[idx_k] += (Cb-alpha[idx_i])*Q_alpha[idx_i*n_feature+idx_k];
                }
                if (beta[idx_k]<1)  // lower bound TBD
                {
                    beta[idx_k] = 1;
                }
                std::cout<<beta[idx_k]<<"\t";
            }

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
            double b = - y_i*Ga[i] + y_j*Ga[j];
            alpha[i] += y_i*b/a;
            alpha[j] -= y_j*b/a;

            // project alpha to the feasible region
            double sum = y_i*old_alpha_i + y_j*old_alpha_j;
            if (alpha[i] > C_i)
                alpha[i] = C_i;
            else if (alpha[i] < 0)
                alpha[i] = 0;
            alpha[j] = y_j*(sum - y_i*alpha[i]);
            if (alpha[j] > C_j)
                alpha[j] = C_j;
            else if (alpha[j] < 0)
                alpha[j] = 0;
            alpha[i] = y_i*(sum - y_j*alpha[j]);

            // update beta
            //if (select_working_feature(k))
            //{
            //    beta[k] = 1 - beta[k];
            //}

            // this section updates gradient
            double d_i = alpha[i] - old_alpha_i;
            double d_j = alpha[j] - old_alpha_j;
            // update Q_alpha
            for (size_t idx_i=0;idx_i<n_sample;++idx_i)
            {
                for (size_t idx_k=0;idx_k<n_feature;++idx_k)
                {
                    Q_alpha[idx_i*n_feature+idx_k] += d_i*Q(i, idx_i, idx_k) + d_j*Q(j, idx_i, idx_k);
                }
            }
            // update Ga
            for (size_t idx_i=0;idx_i<n_sample;++idx_i)
            {
                Ga[idx_i] = -1;
                for (size_t idx_k=0;idx_k<n_feature;++idx_k)
                {
                    Ga[idx_i] += beta[idx_k]*Q_alpha[idx_i*n_feature+idx_k];
                }
            }
            // update Gb
            for (size_t idx_k=0;idx_k<n_feature;++idx_k)
            {
                Gb[idx_k] = 0;
                for (size_t idx_i=0;idx_i<n_sample;++idx_i)
                {
                    Gb[idx_k] += alpha[idx_i]*Q_alpha[idx_i*n_feature+idx_k];
                }
                Gb[idx_k] *= 0.5;
                //std::cout<<Gb[idx_k]<<"\t";
            }
            std::cout<<"\n";

        } // while(iter < max_iter)

        if(iter >= max_iter)
        {
            std::cout<<"\nWARNING: reaching max number of iterations\n";
        }

        rho = calculate_rho();
        if (verbose)
        {
            std::cout<<"Training completed!\n";
        }

    } // void rkm::solve()

    double rkm::predict(const std::vector<double>& x) const
    {
        size_t n_sample = kd->get_n_sample();
        double res = -rho;
        for (size_t i=0;i<n_sample;++i)
        {
            if (alpha[i]>0) //non-zero, support vectors
            {
                res += alpha[i] * (kd->get_label(i)) * (kd->K(i, x, beta));
            }
        }
        return res;
    }

    bool rkm::select_working_set(size_t& i, size_t& j) const
    {
        // return i,j such that
        // i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
        // j: minimizes the decrease of obj value
        //    (if quadratic coefficeint <= 0, replace it with tau)
        //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

        //size_t n_feature = kd->get_n_feature();
        size_t n_sample = kd->get_n_sample();
        double Gmax = -std::numeric_limits<double>::infinity();
        double Gmin = std::numeric_limits<double>::infinity();
        size_t i_idx = n_sample;
        size_t j_idx = n_sample;

        // select i
        for (size_t idx=0;idx<n_sample;++idx)
        {
            if (is_in_I_up(idx))
            {
                double obj_i = - kd->get_label(idx)*Ga[idx];
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
                double tmp = - kd->get_label(idx)*Ga[idx];
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

        if(Gmax-Gmin < eps || i_idx >= n_sample || j_idx >= n_sample)
            return false;

        i = i_idx;
        j = j_idx;
        return true;
    } // int rkm::select_working_set(size_t& i, size_t& j) const

    bool rkm::select_working_feature(size_t& k) const
    {
        size_t n_feature = kd->get_n_feature();
        double Gmax = -std::numeric_limits<double>::infinity();
        size_t k_idx = n_feature;

        // select feature
        for (size_t idx=0;idx<n_feature;++idx)
        {
            double tmp = Gb[idx] * (2*beta[idx] - 1);
            if (tmp > Gmax)
            {
                Gmax = tmp;
                k_idx = idx;
            }
        }
        if (Gmax > 0 && k_idx < n_feature)
        {
            k = k_idx;
            return true;
        }
        else
            return false;
    }

    double rkm::calculate_rho() const
    {
        size_t n_sample = kd->get_n_sample();
        double r;
        size_t nr_free = 0;
        double ub = std::numeric_limits<double>::infinity();
        double lb = -std::numeric_limits<double>::infinity();
        double sum_free = 0;

        for(size_t i=0;i<n_sample;i++)
        {
            double y_i = kd->get_label(i);
            double yG = y_i*Ga[i];

            if(is_upper_bound(i))
            {
                if(y_i==-1)
                    ub = std::min(ub,yG);
                else
                    lb = std::max(lb,yG);
            }
            else if(is_lower_bound(i))
            {
                if(y_i==+1)
                    ub = std::min(ub,yG);
                else
                    lb = std::max(lb,yG);
            }
            else
            {
                ++nr_free;
                sum_free += yG;
            }
        }

        if(nr_free>0)
            r = sum_free/nr_free;
        else
            r = (ub+lb)/2;

        return r;
    }

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

    void rkm::test() const
    {
        if (verbose)
        {
            std::cout<<"\nTesting...\n\n";
        }
        std::string str;
        // testing data input file
        std::ifstream fin (test_file, std::ios_base::in);
        // calc n_sample and n_feature
        size_t n_sample_test = 0;
        size_t n_feature_test = 0;
        while (std::getline(fin, str))
        {
            n_sample_test++;
            if (n_feature_test == 0)
            {
                std::stringstream ss(str);
                double temp;
                ss>>temp>>delim(':');
                while (ss>>temp)
                {
                    n_feature_test++;
                }
            }
        }
        fin.close();

        if (verbose)
        {
            std::cout<<"Found "<<n_sample_test<<" samples with "<<n_feature_test<<" features.\n";
        }
        assert(n_feature_test == kd->get_n_feature());

        // Store target values and predicted values
        std::vector<double> target(n_sample_test);
        std::vector<double> predicted(n_sample_test);

        // read data into the kernel_data
        fin.open(test_file, std::ios_base::in);
        std::vector<double> x(n_feature_test);

        int err_pos = 0;
        int err_neg = 0;
        int n_pos = 0;
        int n_neg = 0;

        for (size_t i=0;i<n_sample_test;++i)
        {
            fin>>target[i]>>delim(':');
            for (size_t j=0;j<n_feature_test;++j)
            {
                fin>>x[j];
            }
            kd->scale_one_vector(x);
            predicted[i] = predict(x);
            if (target[i]>0)
            {
                ++n_pos;
                if (predicted[i]<0)
                    ++err_pos;
            }
            else
            {
                ++n_neg;
                if (predicted[i]>=0)
                    ++err_neg;
            }
            //std::cout<<target[i]<<" : "<<predicted[i]<<"\n";
        }
        fin.close();

        std::cout<<"========================\n";
        std::cout<<"RV #: "<<get_n_rv()<<"  RF #: "<<get_n_rf()<<"\n";
        std::cout<<"========================\n";
        std::cout<<"Positive Error: "<<err_pos*1.0/n_pos<<" ("<<err_pos<<", "<<n_pos<<")"<<"\n";
        std::cout<<"Negative Error: "<<err_neg*1.0/n_neg<<" ("<<err_neg<<", "<<n_neg<<")"<<"\n";
        std::cout<<"Overall  Error: "<<(err_pos+err_neg)*1.0/n_sample_test<<"\n";
        std::cout<<"========================\n";

    } //rkm::rkm(const std::string& input_file_name)

    int rkm::get_n_rv() const
    {
        size_t n_sample = kd->get_n_sample();
        int n_rv = 0;
        for (size_t i=0;i<n_sample;++i)
        {
            if (alpha[i] > 0)
            {
                ++n_rv;
            }
        }
        return n_rv;
    }

    int rkm::get_n_rf() const
    {
        size_t n_feature = kd->get_n_feature();
        int n_rf = 0;
        for (size_t i=0;i<n_feature;++i)
        {
            if (beta[i] > 0)
            {
                ++n_rf;
            }
        }
        return n_rf;
    }

    void rkm::write_model_file() const
    {
        size_t n_sample = kd->get_n_sample();
        size_t n_feature = kd->get_n_feature();

        std::ofstream fout (model_file, std::ios_base::out);
        for (size_t i=0;i<n_sample;++i)
        {
            fout<<alpha[i]<<" ";
        }
        fout<<"\n";
        for (size_t j=0;j<n_feature;++j)
        {
            fout<<beta[j]<<" ";
        }
        fout<<"\n";
        fout.close();
    }

    void rkm::read_model_file(const std::string& model_file_name)
    {
        size_t n_sample = kd->get_n_sample();
        size_t n_feature = kd->get_n_feature();

        alpha.resize(n_sample, 0.0);
        beta.resize(n_feature, 0.0);

        std::ifstream fin (model_file_name, std::ios_base::in);
        for (size_t i=0;i<n_sample;++i)
        {
            fin>>alpha[i];
        }
        for (size_t j=0;j<n_feature;++j)
        {
            fin>>beta[j];
        }
        fin.close();
        rho = calculate_rho();
    }

    void rkm::read_model_file()
    {
        read_model_file(model_file);
    }

} // namespace rkm
