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
      explicit covar_experimental_adaptation(int n, int adapt_experimental)
        : windowed_adaptation("covariance"), estimator_(n), K_(adapt_experimental) {}

      template<typename Model>
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
        if (adaptation_window()) {
          estimator_.add_sample(q);
	}

        if (end_adaptation_window()) {
          compute_next_window();

	  /*estimator_.sample_covariance(covar);

          double n = static_cast<double>(estimator_.num_samples());
          covar = (n / (n + 5.0)) * covar
            + 1e-3 * (5.0 / (n + 5.0))
            * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

	  estimator_.restart();*/
	  
	  Eigen::VectorXd var;
          estimator_.sample_variance(var);

          double n = static_cast<double>(estimator_.num_samples());
          var = (n / (n + 5.0)) * var
                + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());

	  estimator_.restart();

	  Eigen::MatrixXd L = var.array().sqrt().matrix().asDiagonal();//covar.llt().matrixL()
	  //Eigen::MatrixXd Linv = var.array().inverse().sqrt().matrix().asDiagonal();

	  //std::cout << L << std::endl;
	  //std::cout << "**" << std::endl;
	  //std::cout << Linv << std::endl;
	  //std::cout << "**" << std::endl;
	  
          //std::cout << adapt_window_counter_ << std::endl;

	  auto Ax = [&](const Eigen::VectorXd& x) {
	    double lp;
	    Eigen::VectorXd grad1;
	    Eigen::VectorXd grad2;
	    Eigen::VectorXd Ax;
	    //stan::math::hessian_times_vector(log_prob_wrapper_covar<Model>(model), q, x, lp, Ax);
	    double dx = 1e-5;
	    Eigen::VectorXd dr = L * x * dx;
	    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q + dr / 2.0, lp, grad1);
	    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q - dr / 2.0, lp, grad2);
	    Ax = L.transpose() * (grad1 - grad2) / dx;
	    return Ax;
	  };
  
	  int N = q.size();
	  int M = (q.size() < 100) ? q.size() : 100;

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
	  
	  /*double lp;
	  Eigen::VectorXd grad;
	  Eigen::MatrixXd hessian;
	  stan::math::hessian(log_prob_wrapper_covar<Model>(model), q, lp, grad, hessian);
	  
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(L.transpose() * hessian * L);*/

	  //std::cout << "Approx: " << decomp1.eigenvalues()(0) << std::endl;
	  //std::cout << "Exact: " << decomp2.eigenvalues()(0) << std::endl;
	  
	  int K = K_;

	  if(K >= M || K > 25) {
	    throw std::runtime_error("Can't adapt more than min(15, number_params - 1) Lanczos vectors");
	  }
	  
	  Eigen::VectorXd es = decomp1.eigenvalues().block(0, 0, K, 1);
	  Eigen::MatrixXd vs2 = vs * decomp1.eigenvectors().block(0, 0, M, K);

	  Eigen::MatrixXd Vnull = Eigen::MatrixXd::Random(N, N - K);
	  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
	  Vnull = (Vnull - vs2 * (vs2.transpose() * Vnull)).eval();
	  V.block(0, 0, N, K) = vs2;
	  V.block(0, K, N, N - K) = stan::math::qr_thin_Q(Vnull);

	  //for(int i = 0; i < K; i++) {
	  //  std::cout << std::setprecision(3) << es(i) << " : " << vs2.block(0, i, N, 1).transpose() << std::endl;
	  //}
	  //std::cout << "----" << std::endl;

	  /*std::cout << (V.transpose() * V).block(0, 0, 5, 5) << std::endl;
	  std::cout << (V.transpose() * V).diagonal().transpose() << std::endl;
	  std::cout << "off diagonals: " << (V.transpose() * V).array().abs().sum() - N << std::endl;*/

	  double etail = decomp1.eigenvalues()(K);

	  //std::cout << etail << std::endl;
	  
	  Eigen::LLT< Eigen::MatrixXd > LLT = covar.llt();
	  
	  Eigen::VectorXd es2 = Eigen::VectorXd::Zero(N);
	  for(int i = 0; i < N; i++) {
	    if(i < K) {
	      es2(i) = es(i);
	    } else {
	      es2(i) = etail;////-1.0;-(V.block(0, i, N, 1).transpose() * LLT.solve(V.block(0, i, N, 1)))(0, 0);
	    }
	  }

	  //Eigen::MatrixXd A = L.transpose() * V * es2.asDiagonal() * V.transpose() * L;
	  //Eigen::MatrixXd Ainv = Linv * V * es2.array().inverse().matrix().asDiagonal() * V.transpose() * Linv.transpose();
	  Eigen::MatrixXd Ainv = L * V * es2.array().inverse().matrix().asDiagonal() * V.transpose() * L.transpose();

	  /*std::cout << A << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << hessian << std::endl;
	  std::cout << "----****" << std::endl;*/

	  //std::cout << Ainv * hessian << std::endl;
	  //std::cout << "----!!!!" << std::endl;

	  /*std::cout << -A.inverse() << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -Ainv << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -hessian.inverse() << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << covar << std::endl;
	  std::cout << "----xxxx" << std::endl;*/
	  
          for (int i = 0; i < covar.size(); ++i) {
            covar(i) = -Ainv(i);
	  }

          ++adapt_window_counter_;

          return true;
        }

        ++adapt_window_counter_;
        return false;
      }

    protected:
      stan::math::welford_var_estimator estimator_;
      int K_;
    };

  }  // mcmc

}  // stan

#endif
