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

        eq_precomputed = false;

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
            else if (str.compare("Cb") == 0)
            {
                fin>>Cb;
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
            else if (str.compare("precompute_eq") == 0)
            {
                fin>>eq_file;
                eq_precomputed = true;
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
        kd -> set_gamma(gamma); // this has to be prior to set_kernel for lookup_gaussian
        kd -> set_kernel(kernel_name);

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

        // Pre-compute e^T times Q_t
        if (eq_precomputed)
        {
            eQ = std::vector<double>(n_sample*n_feature, 0.0);
            std::ifstream f_eq (eq_file, std::ios_base::in);
            if (f_eq.good()) // eQ pre-computed and saved, file exists
            {
                for (size_t i=0;i<n_sample;++i)
                {
                    for (size_t j=0;j<n_feature;++j)
                    {
                        f_eq>>eQ[i*n_feature+j];
                    }
                }
                f_eq.close();
            }
            else // file does not exist, compute and save to file
            {
                f_eq.close();
                for (size_t i=0;i<n_sample;++i)
                {
                    for (size_t j=0;j<n_sample;++j)
                    {
                        for (size_t t=0;t<n_feature;++t)
                        {
                            eQ[i*n_feature+j] += Q(i, j, t);
                        }
                    }
                }
                // write to file
                std::ofstream f_eq_out(eq_file, std::ios_base::out);
                for (size_t i=0;i<n_sample;++i)
                {
                    for (size_t j=0;j<n_feature;++j)
                    {
                        f_eq_out<<eQ[i*n_feature+j]<<" ";
                    }
                    f_eq_out<<"\n";
                }
                f_eq_out.close();
            }
        }

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
        if (kd->get_label(i) != kd->get_label(j))
        {
            return (- kd->kernel_one(i, j, k));
        }
        else
        {
            return (kd->kernel_one(i, j, k));
        }
    }

    void rkm::solve()
    {
        if (verbose)
        {
            std::cout<<"Training started...\n";
        }
        size_t n_feature = kd->get_n_feature();
        size_t n_sample = kd->get_n_sample();

        // TBD
        // Initialization
        beta.resize(n_feature, 1.0/n_feature);
        double sum_beta = 1.0;

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
            std::cout<<iter<<": ";
            if(--counter == 0)
            {
                counter = (n_sample>1000 ? 1000 : n_sample);
                //if(shrinking) do_shrinking();
                std::cout<<".";
            }

            size_t i, j;
            if (!select_working_set(i, j))
                break;

            ++iter;

            // update beta
            sum_beta = 0;
            for (size_t idx_k=0;idx_k<n_feature;++idx_k)
            {
                beta[idx_k] = 0;
                for (size_t idx_i=0;idx_i<n_sample;++idx_i)
                {
                    beta[idx_k] += (Cb-alpha[idx_i])*Q_alpha[idx_i*n_feature+idx_k];
                }
                if (beta[idx_k]<1.0/n_feature)  // lower bound TBD
                {
                    beta[idx_k] = 1.0/n_feature;
                }
                sum_beta += beta[idx_k];
                //std::cout<<beta[idx_k]<<"\t";
            }

            // update alpha[i] and alpha[j], handle bounds carefully
            double C_i = get_C(i);
            double C_j = get_C(j);
            double y_i = kd->get_label(i);
            double y_j = kd->get_label(j);
            double old_alpha_i = alpha[i];
            double old_alpha_j = alpha[j];

            //double a = K(i, i) + K(j, j) - 2*K(i, j);
            double a = 2*sum_beta - 2*K(i, j);
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
            //for (size_t idx_k=0;idx_k<n_feature;++idx_k)
            //{
            //    Gb[idx_k] = 0;
            //    for (size_t idx_i=0;idx_i<n_sample;++idx_i)
            //    {
            //        Gb[idx_k] += alpha[idx_i]*Q_alpha[idx_i*n_feature+idx_k];
            //    }
            //    Gb[idx_k] *= 0.5;
            //    //std::cout<<Gb[idx_k]<<"\t";
            //}
            //std::cout<<"\n";

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

    bool rkm::select_working_set(size_t& i, size_t& j, double& delta_obj) const
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
                    // for gaussian kernel and 1/F initialization
                    double a = 2*sum_of_beta() - 2*K(i_idx, idx);
                    //double a = K(i_idx, i_idx) + K(idx, idx) - 2*K(i_idx, idx);
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

        delta_obj = Gmax - Gmin;
        if(delta_obj < eps || i_idx >= n_sample || j_idx >= n_sample)
            return false;

        i = i_idx;
        j = j_idx;
        return true;
    } // int rkm::select_working_set(size_t& i, size_t& j, double& delta_obj) const

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
                    // for gaussian kernel and 1/F initialization
                    double a = 2*sum_of_beta() - 2*K(i_idx, idx);
                    //double a = K(i_idx, i_idx) + K(idx, idx) - 2*K(i_idx, idx);
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
        std::cout<<(Gmax-Gmin)<<"\n";

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

    double rkm::sum_of_beta() const
    {
        double sum = 0;
        size_t n_feature = kd->get_n_feature();
        for (size_t i=0;i<n_feature;++i)
        {
            sum += beta[i];
        }
        return sum;
    }

    void rkm::solve_no_fs()
    {
        if (verbose)
        {
            std::cout<<"Training started...\n";
        }
        size_t n_feature = kd->get_n_feature();
        size_t n_sample = kd->get_n_sample();

        // TBD
        // Initialization
        beta.resize(n_feature, 1.0/n_feature);

        alpha.resize(n_sample, 0.0);
        alpha_status.resize(n_sample);
        for (size_t i=0;i<n_sample;++i)
        {
            update_alpha_status(i);
        }

        // Gradient init
        Ga.resize(n_sample, -1.0);
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
            std::cout<<iter<<": ";
            if(--counter == 0)
            {
                counter = (n_sample>1000 ? 1000 : n_sample);
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

            // for gaussian kernel and 1/F initialization
            //double a = K(i, i) + K(j, j) - 2*K(i, j);
            double a = 2*sum_of_beta() - 2*K(i, j);
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

            // this section updates gradient
            double d_i = alpha[i] - old_alpha_i;
            double d_j = alpha[j] - old_alpha_j;

            // update Ga
            for (size_t idx_i=0;idx_i<n_sample;++idx_i)
            {
                double y_idx = kd->get_label(idx_i);
                Ga[idx_i] += d_i*y_i*y_idx*K(idx_i, i) + d_j*y_j*y_idx*K(idx_i, j);
            }

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

    } // void rkm::solve_no_fs()

    void rkm::solve_iter()
    {
        if (verbose)
        {
            std::cout<<"Training started...\n";
        }
        RKM::timer run_time;
        size_t n_feature = kd->get_n_feature();
        size_t n_sample = kd->get_n_sample();
        double bmax = 1.0/n_feature;
        double bmin = 0;

        // Initialization
        beta.resize(n_feature, 1.0/n_feature);
        double sum_beta = 1.0;
        double obj = 0.0;
        double d_obj = 0.0;
        double prev_obj = 0.0;

        int max_ext_iter = 100;
        int ext_iter = 0;

        while (1) // alpha and beta ext iteration
        {
            if (verbose)
            {
                std::cout<<"\tUpdating alpha...\n";
            }
            // alpha iteration
            run_time.start();
            alpha = std::vector<double>(n_sample, 0.0);
            alpha_status.resize(n_sample);
            for (size_t i=0;i<n_sample;++i)
            {
                update_alpha_status(i);
            }

            // Gradient init
            //Ga.resize(n_sample, -1.0);
            Ga = std::vector<double>(n_sample, -1.0);

            prev_obj = obj;
            obj = 0;

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
                    //std::cout<<".";
                    std::cout<<"\t\t"<<iter<<": "<<d_obj<<"\n";
                }

                size_t i, j;
                if (!select_working_set(i, j, d_obj))
                    break;

                obj -= d_obj;
                ++iter;

                // update alpha[i] and alpha[j], handle bounds carefully
                double C_i = get_C(i);
                double C_j = get_C(j);
                double y_i = kd->get_label(i);
                double y_j = kd->get_label(j);
                double old_alpha_i = alpha[i];
                double old_alpha_j = alpha[j];

                // for gaussian kernel and 1/F initialization
                //double a = K(i, i) + K(j, j) - 2*K(i, j);
                double a = 2*sum_of_beta() - 2*K(i, j);
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

                // this section updates gradient
                double d_i = alpha[i] - old_alpha_i;
                double d_j = alpha[j] - old_alpha_j;

                // update Ga
                for (size_t idx_i=0;idx_i<n_sample;++idx_i)
                {
                    double y_idx = kd->get_label(idx_i);
                    Ga[idx_i] += d_i*y_i*y_idx*K(idx_i, i) + d_j*y_j*y_idx*K(idx_i, j);
                }

            } // while(iter < max_iter)

            if(iter >= max_iter)
            {
                std::cout<<"\nWARNING: reaching max number of iterations\n";
            }
            run_time.stop("\t#### Alpha iteration run time");

            if (prev_obj - obj < eps)
                break;
            else
                std::cout<<"\tEXT iteration improvement: "<<prev_obj - obj<<"\n";

            if (verbose)
            {
                std::cout<<"\tUpdating beta...\n";
            }
            run_time.start();

            // update beta
            double beta_d_obj = 0;
            std::vector<double> new_beta(n_feature);
            double sum = 0;
            for (size_t idx_k=0;idx_k<n_feature;++idx_k)
            {
                if (idx_k/100 == idx_k*1.0/100)
                    std::cout<<"\t\tFeature: "<< idx_k<<"\n";
                new_beta[idx_k] = 0;
                if (eq_precomputed)
                {
                    double eQa = 0;
                    double aQa = 0;
                    for (size_t idx_i=0;idx_i<n_sample;++idx_i)
                    {
                        if (alpha[idx_i] > 0)
                        {
                            eQa += eQ[idx_i*n_feature+idx_k]*alpha[idx_i];
                        }
                        for (size_t idx_j=0;idx_j<n_sample;++idx_j)
                        {
                            if (alpha[idx_j]>0)
                            {
                                aQa += alpha[idx_i]*Q(idx_i, idx_j, idx_k)*alpha[idx_j];
                            }
                        }
                    }
                    new_beta[idx_k] = Cb*eQa - aQa;
                }
                else
                {
                    for (size_t idx_i=0;idx_i<n_sample;++idx_i)
                    {
                        double tmp = 0;
                        for (size_t idx_j=0;idx_j<n_sample;++idx_j)
                        {
                            if (alpha[idx_j]>0)
                            {
                                tmp += Q(idx_i, idx_j, idx_k)*alpha[idx_j];
                            }
                        }
                        new_beta[idx_k] += (Cb - alpha[idx_i])*tmp;
                        //new_beta[idx_k] += (Cb)*tmp;
                        //new_beta[idx_k] += (alpha[idx_i])*tmp;
                    }
                }
                if (new_beta[idx_k]<0)
                {
                    new_beta[idx_k] = 0;
                }
                sum += new_beta[idx_k];
                //std::cout<<new_beta[idx_k]<<"\t";
                //beta_d_obj -= new_beta[idx_k]*(new_beta[idx_k]/2.0-temp);
                //beta_d_obj += beta[idx_k]*(beta[idx_k]/2.0-temp);
            }
            for (size_t idx_k=0;idx_k<n_feature;++idx_k)
            {
                new_beta[idx_k] /= sum;
                //std::cout<<new_beta[idx_k]<<"\t";
            }
            //std::cout<<"\n";
            //size_t i, j;
            //if (!select_working_set(i, j, obj))
            //{
            //    break;
            //}
            //std::cout<<"\t\tOBJ: "<<obj<<"\n";
            //if (beta_d_obj<eps)
            //{
            //    break;
            //}
            //else
            //{
            //    beta = new_beta;
            //}
            if (++ext_iter > max_ext_iter)
            {
                std::cout<<"\nWARNING: reaching max number of EXT iterations\n";
                break;
            }
            beta = new_beta;
            run_time.stop("\t#### Beta iteration run time");
            //std::cout<<"\t\t"<<obj<<"\n";
        } // while (1), alpha and beta exterior loop

        rho = calculate_rho();
        if (verbose)
        {
            std::cout<<"Training completed!\n";
        }

    } // void rkm::solve()

} // namespace rkm
