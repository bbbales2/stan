#ifndef STAN_MCMC_STEPSIZE_VAR_EXPERIMENTAL_ADAPTER_HPP
#define STAN_MCMC_STEPSIZE_VAR_EXPERIMENTAL_ADAPTER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/var_experimental_adaptation.hpp>

namespace stan {
  namespace mcmc {

    class stepsize_var_experimental_adapter: public base_adapter {
    public:
      explicit stepsize_var_experimental_adapter(int n, int which_adaptation)
        : var_adaptation_(n, which_adaptation) {
      }

      stepsize_adaptation& get_stepsize_adaptation() {
        return stepsize_adaptation_;
      }

      var_experimental_adaptation& get_var_adaptation() {
        return var_adaptation_;
      }

      void set_window_params(unsigned int num_warmup,
                             unsigned int init_buffer,
                             unsigned int term_buffer,
                             unsigned int base_window,
                             callbacks::logger& logger) {
        var_adaptation_.set_window_params(num_warmup,
                                          init_buffer,
                                          term_buffer,
                                          base_window,
                                          logger);
      }


    protected:
      stepsize_adaptation stepsize_adaptation_;
      var_experimental_adaptation var_adaptation_;
    };

  }  // mcmc
}  // stan
#endif
