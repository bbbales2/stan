#ifndef STAN_MCMC_HMC_INTEGRATORS_IMPL_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_IMPL_LEAPFROG_HPP

#include <Eigen/Dense>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {
  namespace mcmc {

    template <typename Hamiltonian>
    class impl_leapfrog: public base_leapfrog<Hamiltonian> {
    public:
      impl_leapfrog(): base_leapfrog<Hamiltonian>(),
                       max_num_fixed_point_(10),
                       fixed_point_threshold_(1e-8) {}

      void begin_update_p(typename Hamiltonian::PointType& z,
                          Hamiltonian& hamiltonian,
                          double epsilon,
                          callbacks::logger& logger) {
        hat_phi(z, hamiltonian, epsilon, logger);
        hat_tau(z, hamiltonian, epsilon, this->max_num_fixed_point_,
                logger);
      }

      void update_q(typename Hamiltonian::PointType& z,
                    Hamiltonian& hamiltonian,
                    double epsilon,
                    callbacks::logger& logger) {
        // hat{T} = dT/dp * d/dq
	for(int m = 0; m < 4; m++) {
	  typename Hamiltonian::PointType wz = z;
	  double inner_epsilon = epsilon * std::pow(0.5, m);
	  try {
	    Eigen::VectorXd q_init = wz.q + 0.5 * inner_epsilon * hamiltonian.dtau_dp(wz);
	    Eigen::VectorXd delta_q(wz.q.size());
	    
	    for (int i = 0; i <= m; ++i) {
	      for (int n = 0; n < this->max_num_fixed_point_; ++n) {
		delta_q = wz.q;
		wz.q.noalias() = q_init + 0.5 * inner_epsilon * hamiltonian.dtau_dp(wz);
		hamiltonian.update_metric(wz, logger);
		
		delta_q -= wz.q;
		if (delta_q.cwiseAbs().maxCoeff() < this->fixed_point_threshold_)
		  break;
	      }

	      std::cout << "integrated to: " << (i + 1) * inner_epsilon << std::endl;
	    }

	    z = wz;

	    break;
	  } catch (...) {
	    std::cout << "inner_epsilon " << inner_epsilon << " failed" << std::endl;
	    if(m == 3) {
	      throw;
	    }
	  }
	}
	
	hamiltonian.update_gradients(z, logger);
      }

      void end_update_p(typename Hamiltonian::PointType& z,
                        Hamiltonian& hamiltonian,
                        double epsilon,
                        callbacks::logger& logger) {
        hat_tau(z, hamiltonian, epsilon, 1, logger);
        hat_phi(z, hamiltonian, epsilon, logger);
      }

      // hat{phi} = dphi/dq * d/dp
      void hat_phi(typename Hamiltonian::PointType& z,
                   Hamiltonian& hamiltonian,
                   double epsilon,
                   callbacks::logger& logger) {
        z.p -= epsilon * hamiltonian.dphi_dq(z, logger);
      }

      // hat{tau} = dtau/dq * d/dp
      void hat_tau(typename Hamiltonian::PointType& z,
                   Hamiltonian& hamiltonian,
                   double epsilon,
                   int num_fixed_point,
                   callbacks::logger& logger) {
        Eigen::VectorXd p_init = z.p;
        Eigen::VectorXd delta_p(z.p.size());

        for (int n = 0; n < num_fixed_point; ++n) {
          delta_p = z.p;
          z.p.noalias() = p_init - epsilon * hamiltonian.dtau_dq(z, logger);
          delta_p -= z.p;
          if (delta_p.cwiseAbs().maxCoeff() < this->fixed_point_threshold_)
            break;
        }
      }

      int max_num_fixed_point() {
        return this->max_num_fixed_point_;
      }

      void set_max_num_fixed_point(int n) {
        if (n > 0) this->max_num_fixed_point_ = n;
      }

      double fixed_point_threshold() {
        return this->fixed_point_threshold_;
      }

      void set_fixed_point_threshold(double t) {
        if (t > 0) this->fixed_point_threshold_ = t;
      }

    private:
      int max_num_fixed_point_;
      double fixed_point_threshold_;
    };

  }  // mcmc
}  // stan

#endif
