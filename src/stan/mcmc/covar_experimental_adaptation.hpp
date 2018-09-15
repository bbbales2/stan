#ifndef STAN_MCMC_COVAR_EXPERIMENTAL_ADAPTATION_HPP
#define STAN_MCMC_COVAR_EXPERIMENTAL_ADAPTATION_HPP

#include <stan/math/mix/mat.hpp>
#include <stan/mcmc/windowed_adaptation.hpp>
#include <vector>

namespace stan {

  namespace mcmc {

    template <typename Model>
    struct log_prob_wrapper_covar {
      const Model& model_;
      log_prob_wrapper_covar(const Model& model) : model_(model) {}
      
      template <typename T>
      T operator()(const Eigen::Matrix<T, Eigen::Dynamic, 1>& q) const {
	return model_.template log_prob<true, true, T>(const_cast<Eigen::Matrix<T, Eigen::Dynamic, 1>& >(q), &std::cout);
      }
    };
    
    class covar_experimental_adaptation: public windowed_adaptation {
    protected:
      std::vector<Eigen::MatrixXd> hessians;
    public:
      explicit covar_experimental_adaptation(int n)
        : windowed_adaptation("covariance"), estimator_(n) {}

      template<typename Model>
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
        if (adaptation_window()) {
          estimator_.add_sample(q);
	  double lp;
	  Eigen::VectorXd grad;
	  Eigen::MatrixXd hessian;
	  stan::math::hessian(log_prob_wrapper_covar<Model>(model), q, lp, grad, hessian);
	  hessians.push_back(hessian);
	}

        if (end_adaptation_window()) {
          compute_next_window();

          estimator_.sample_covariance(covar);

          double n = static_cast<double>(estimator_.num_samples());
          covar = (n / (n + 5.0)) * covar
            + 1e-3 * (5.0 / (n + 5.0))
            * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

          estimator_.restart();
	  Eigen::MatrixXd avg = hessians[0];
	  for(auto& hess : hessians)
	    avg += hess;
	  avg = avg / hessians.size();

	  Eigen::MatrixXd inverse = -avg.inverse();

	  std::cout << inverse << std::endl << "----" << std::endl << covar << std::endl;

	  for (int i = 0; i < covar.size(); ++i) {
	    covar(i) = inverse(i);
	  }

          ++adapt_window_counter_;

	  hessians.clear();
          return true;
        }

        ++adapt_window_counter_;
        return false;
      }

    protected:
      stan::math::welford_covar_estimator estimator_;
    };

  }  // mcmc

}  // stan

#endif
