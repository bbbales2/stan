#ifndef STAN_MCMC_HMC_NUTS_ADAPT_EXPERIMENTAL_DENSE_E_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_ADAPT_EXPERIMENTAL_DENSE_E_NUTS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/stepsize_covar_experimental_adapter.hpp>
#include <stan/mcmc/hmc/nuts/dense_e_nuts.hpp>

namespace stan {
  namespace mcmc {
    /**
     * The No-U-Turn sampler (NUTS) with multinomial sampling
     * with a Gaussian-Euclidean disintegration and adaptive
     * dense metric and adaptive step size
     */
    template <class Model, class BaseRNG>
    class adapt_experimental_dense_e_nuts : public dense_e_nuts<Model, BaseRNG>,
					    public stepsize_covar_experimental_adapter {
    protected:
      const Model& model_;
    public:
      adapt_experimental_dense_e_nuts(const Model& model, int which_adaptation, BaseRNG& rng)
        : model_(model), dense_e_nuts<Model, BaseRNG>(model, rng),
        stepsize_covar_experimental_adapter(model.num_params_r(), which_adaptation) {}

      ~adapt_experimental_dense_e_nuts() {}

      sample
      transition(sample& init_sample, callbacks::logger& logger) {
	/*if (this->adapt_flag_) {
	  this->nom_epsilon_ = this->covar_adaptation_.learn_stepsize(model_,
								      this->z_.inv_e_metric_,
								      this->z_.q) / 2.0;
								      }*/

        sample s = dense_e_nuts<Model, BaseRNG>::transition(init_sample,
                                                            logger);

	//std::cout << "divergent: " << this->divergent_ << std::endl;

        if (this->adapt_flag_) {
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                    s.accept_stat());
	  //std::cout << "nom_epsilon_: " << this->nom_epsilon_ << std::endl;

	  double stability_limit = 0.0;
          bool update = this->covar_adaptation_.learn_covariance(model_,
								 this->z_.inv_e_metric_,
								 this->z_.q,
								 stability_limit);

          if (update) {
            this->init_stepsize(logger);
	    //10 * this->nom_epsilon_
            this->stepsize_adaptation_.set_mu(log(stability_limit));
            this->stepsize_adaptation_.restart();
          }
        }
        return s;
      }

      void disengage_adaptation() {
        base_adapter::disengage_adaptation();
        this->stepsize_adaptation_.complete_adaptation(this->nom_epsilon_);
      }
    };

  }  // mcmc
}  // stan
#endif
