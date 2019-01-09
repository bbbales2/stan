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
    public:
      explicit covar_experimental_adaptation(int n)
        : windowed_adaptation("covariance"), estimator_(n) {}

      
      template<typename Model>
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
        if (adaptation_window()) {
          estimator_.add_sample(q);
	}

        if (end_adaptation_window()) {
          compute_next_window();
	  
          estimator_.sample_covariance(covar);

          double n = static_cast<double>(estimator_.num_samples());
          covar = (n / (n + 5.0)) * covar
            + 1e-3 * (5.0 / (n + 5.0))
            * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

          estimator_.restart();

          //std::cout << adapt_window_counter_ << std::endl;

	  auto Ax = [&](const Eigen::VectorXd& x) {
	    double lp;
	    Eigen::VectorXd grad1;
	    Eigen::VectorXd grad2;
	    Eigen::VectorXd Ax;
	    //stan::math::hessian_times_vector(log_prob_wrapper_covar<Model>(model), q, x, lp, Ax);
	    double dx = 1e-5;
	    Eigen::VectorXd dr = x * (dx / x.norm());
	    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q + dr / 2.0, lp, grad1);
	    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q - dr / 2.0, lp, grad2);
	    Ax = (grad1 - grad2) / dx;
	    return Ax;
	  };
  
	  int N = q.size();
	  int M = (q.size() < 10) ? q.size() : 10;

	  Eigen::MatrixXd vs = Eigen::MatrixXd::Zero(N, M);
	  Eigen::VectorXd a = Eigen::VectorXd::Zero(M);
	  Eigen::VectorXd b = Eigen::VectorXd::Zero(M);

	  Eigen::VectorXd r = Eigen::VectorXd::Random(N);
	  vs.block(0, 0, N, 1) = r / r.norm();

	  r = Ax(vs.block(0, 0, N, 1));
	  a(0) = (vs.block(0, 0, N, 1).transpose() * r)(0, 0);
	  r = r - a(0) * vs.block(0, 0, N, 1);
	  b(0) = r.norm();
	  for(int j = 1; j < M; j++) {
	    vs.block(0, j, N, 1) = r / b(j - 1);
	    r = Ax(vs.block(0, j, N, 1));
	    r = r - vs.block(0, j - 1, N, 1) * b(j - 1);
	    a(j) = (vs.block(0, j, N, 1).transpose() * r)(0, 0);
	    r = r - vs.block(0, j, N, 1) * a(j);
	    r = r - vs * (vs.transpose() * r);
	    b(j) = r.norm();
	  }

	  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(M, M);
	  T(0, 0) = a(0);
	  for(int m = 1; m < M; m++) {
	    T(m, m) = a(m);
	    T(m, m - 1) = b(m - 1);
	    T(m - 1, m) = b(m - 1);
	  }

          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp1(T);
	  
	  //double lp;
	  //Eigen::VectorXd grad;
	  //Eigen::MatrixXd hessian;
	  //stan::math::hessian(log_prob_wrapper_covar<Model>(model), q, lp, grad, hessian);
	  
          //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(hessian);

	  //std::cout << "Approx: " << decomp1.eigenvalues()(0) << std::endl;
	  //std::cout << "Exact: " << decomp2.eigenvalues()(0) << std::endl;

	  int K = (N < 50) ? 1 : 4;
	  
	  Eigen::VectorXd es = decomp1.eigenvalues().block(0, 0, K, 1);
	  Eigen::MatrixXd vs2 = vs * decomp1.eigenvectors().block(0, 0, M, K);

	  double etail = decomp1.eigenvalues()(K);

	  Eigen::VectorXd es2 = Eigen::VectorXd::Zero(N);
	  for(int i = 0; i < N; i++) {
	    if(i < K) {
	      es2(i) = es(i);
	    } else {
	      es2(i) = etail;
	    }
	  }

	  Eigen::MatrixXd Vnull = Eigen::MatrixXd::Random(N, N - K);

	  Vnull = (Vnull - vs2 * (vs2.transpose() * Vnull)).eval();

	  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);

	  V.block(0, 0, N, K) = vs2;
	  V.block(0, K, N, N - K) = stan::math::qr_thin_Q(Vnull);

	  //Eigen::MatrixXd A = V * es2.asDiagonal() * V.transpose();
	  Eigen::MatrixXd Ainv = V * es2.array().inverse().matrix().asDiagonal() * V.transpose();

	  /*std::cout << A << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << hessian << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -covar.inverse() << std::endl;
	  std::cout << "----****" << std::endl;

	  std::cout << Ainv * hessian << std::endl;
	  std::cout << "----!!!!" << std::endl;

	  std::cout << -A.inverse() << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -Ainv << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -hessian.inverse() << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << covar << std::endl;
	  std::cout << "----xxxx" << std::endl;*/
	  
	  Eigen::MatrixXd negative_inverse = -Ainv;

          for (int i = 0; i < covar.size(); ++i) {
            covar(i) = negative_inverse(i);
	  }

          ++adapt_window_counter_;

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
