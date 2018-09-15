#ifndef STAN_MCMC_VAR_EXPERIMENTAL_ADAPTATION_HPP
#define STAN_MCMC_VAR_EXPERIMENTAL_ADAPTATION_HPP

#include <stan/math/mix/mat.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>
#include <vector>
#include <iostream>

namespace stan {

  namespace mcmc {

    template <typename Model>
    struct log_prob_wrapper {
      const Model& model_;
      log_prob_wrapper(const Model& model) : model_(model) {}
      
      template <typename T>
      T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& q) const {
	return model_.template log_prob<true, true, T>(const_cast<Eigen::Matrix<T, Eigen::Dynamic, 1>& >(q), &std::cout);
      }
    };

    class var_experimental_adaptation: public windowed_adaptation {
    protected:
      std::vector<Eigen::MatrixXd> hessians;
    public:
      explicit var_experimental_adaptation(int n)
        : windowed_adaptation("variance"), estimator_(n) {}

      template<typename Model>
      bool learn_variance(const Model& model, Eigen::VectorXd& var, const Eigen::VectorXd& q) {
        if (adaptation_window()) {
          estimator_.add_sample(q);
	  double lp;
	  Eigen::VectorXd grad;
	  Eigen::MatrixXd hessian;
	  stan::math::hessian(log_prob_wrapper<Model>(model), q, lp, grad, hessian);
	  hessians.push_back(hessian);
	}

        if (end_adaptation_window()) {
          compute_next_window();

          estimator_.sample_variance(var);

          double n = static_cast<double>(estimator_.num_samples());
          var = (n / (n + 5.0)) * var
                + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());

          estimator_.restart();
	  Eigen::MatrixXd avg = hessians[0];
	  for(auto& hess : hessians)
	    avg += hess;
	  avg = avg / hessians.size();

	  for (int i = 0; i < var.size(); ++i) {
	    std::cout << -1.0 / avg(i, i) << " " << var(i) << std::endl;
	    var(i) = -1.0 / avg(i, i);
	  }
	  
          ++adapt_window_counter_;

	  hessians.clear();
          return true;
        }

        ++adapt_window_counter_;
        return false;
      }

    protected:
      stan::math::welford_var_estimator estimator_;
    };

  }  // mcmc

}  // stan
#endif
