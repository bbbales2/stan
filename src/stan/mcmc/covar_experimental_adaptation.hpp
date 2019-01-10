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

	  Eigen::VectorXd var;// = Eigen::VectorXd::Zero(q.size());
	  //for(int i = 0; i < q.size(); i++)
	  //  var(i) = 1;
	  
          estimator_.sample_variance(var);

          double n = static_cast<double>(estimator_.num_samples());
          var = (n / (n + 5.0)) * var
                + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());

	  estimator_.restart();

          //std::cout << adapt_window_counter_ << std::endl;

	  auto CAx = [&](const Eigen::VectorXd& x) {
	    double lp;
	    Eigen::VectorXd grad1;
	    Eigen::VectorXd grad2;
	    Eigen::VectorXd Ax;
	    //stan::math::hessian_times_vector(log_prob_wrapper_covar<Model>(model), q, x, lp, Ax);
	    double dx = 1e-5;
	    Eigen::VectorXd dr = x * (dx / x.norm());
	    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q + dr / 2.0, lp, grad1);
	    stan::math::gradient(log_prob_wrapper_covar<Model>(model), q - dr / 2.0, lp, grad2);
	    Ax = (grad1 - grad2) / dx;//var.asDiagonal() *
	    return Ax;
	  };
  
	  int N = q.size();
	  int M = (q.size() < 10) ? q.size() : 10;

	  Eigen::MatrixXd vs = Eigen::MatrixXd::Zero(N, M);
	  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(M, M);
	  Eigen::VectorXd r = Eigen::VectorXd::Random(N);
	  vs.block(0, 0, N, 1) = r / r.norm();
	  for(int j = 0; j < M; j++) {
	    Eigen::VectorXd w = CAx(vs.block(0, j, N, 1));
	    for(int i = 0; i <= j; i++) {
	      H(i, j) = (vs.block(0, i, N, 1).transpose() * w)(0, 0);
	      w = w - H(i, j) * vs.block(0, i, N, 1);
	    }
	    if(j < M - 1) {
	      H(j + 1, j) = w.norm();
	      vs.block(0, j + 1, N, 1) = w / H(j + 1, j);
	    }
	  }

          Eigen::EigenSolver<Eigen::MatrixXd> decomp1(H);

	  std::vector<int> idxs(N);
	  for(int i = 0; i < N; i++) { idxs[i] = i; }
	  std::sort(idxs.begin(), idxs.end(), [&](int i, int j) {
	      return decomp1.eigenvalues()(i).real() < decomp1.eigenvalues()(j).real();
	    });

	  std::cout << "Approx: " << decomp1.eigenvalues()(idxs[0]).real() << std::endl;

	  double lp;
	  Eigen::VectorXd grad;
	  Eigen::MatrixXd hessian;
	  stan::math::hessian(log_prob_wrapper_covar<Model>(model), q, lp, grad, hessian);
	  
          Eigen::EigenSolver<Eigen::MatrixXd> decomp2(hessian);//var.asDiagonal() * 

	  std::vector<int> idxs2(N);
	  for(int i = 0; i < N; i++) { idxs2[i] = i; }
	  std::sort(idxs2.begin(), idxs2.end(), [&](int i, int j) {
	      return decomp2.eigenvalues()(i).real() < decomp2.eigenvalues()(j).real();
	    });

	  std::cout << "Exact: " << decomp2.eigenvalues()(idxs2[0]).real() << std::endl;
	  
	  int K = (N < 50) ? 2 : 4;
	  
	  //Eigen::VectorXd es = decomp1.eigenvalues().block(0, 0, K, 1);
	  Eigen::MatrixXd vs2 = Eigen::MatrixXd::Zero(N, K);
	  for(int i = 0; i < K; i++) {
	    vs2.block(0, i, N, 1) = vs * decomp1.eigenvectors().block(0, idxs[i], M, 1).real();
	    std::cout << "v" << i << " : " << vs2.block(0, i, N, 1).transpose() << std::endl;
	  }

	  Eigen::MatrixXd Vnull = Eigen::MatrixXd::Random(N, N - K);
	  Vnull = (Vnull - vs2 * (vs2.transpose() * Vnull)).eval();

	  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
	  V.block(0, 0, N, K) = vs2;
	  V.block(0, K, N, N - K) = stan::math::qr_thin_Q(Vnull);

	  double etail = decomp1.eigenvalues()(idxs[K]).real();

	  std::cout << decomp1.eigenvalues().real().transpose() << std::endl;

	  std::cout << "tail: " << etail << std::endl;

	  Eigen::VectorXd es2 = Eigen::VectorXd::Zero(N);
	  for(int i = 0; i < N; i++) {
	    if(i < K) {
	      es2(i) = decomp1.eigenvalues()(idxs[i]).real();
	    } else {
	      es2(i) = -(V.block(0, i, N, 1).transpose() * var.asDiagonal().inverse() * V.block(0, i, N, 1))(0, 0);
	    }
	  }

	  std::cout << "es2" << std::endl;
	  std::cout << es2.transpose() << std::endl;
	  std::cout << "----" << std::endl;

	  std::cout << V.transpose() * V << std::endl;
	  std::cout << "----" << std::endl;

	  Eigen::MatrixXd A = V * es2.asDiagonal() * V.transpose();
	  Eigen::MatrixXd Ainv = V * es2.array().inverse().matrix().asDiagonal() * V.transpose();

	  std::cout << "var:" << std::endl;
	  std::cout << var.transpose() << std::endl;
	  std::cout << "Arnoldi approx:" << std::endl;
	  std::cout << vs * H * vs.transpose() << std::endl;
	  std::cout << "Approx:" << std::endl;
	  std::cout << A << std::endl;
	  std::cout << "Exact:" << std::endl;
	  std::cout << hessian << std::endl;
	  std::cout << "----****" << std::endl;

	  std::cout << -((-A).llt().solve(hessian)) << std::endl;
	  std::cout << "----!!!!" << std::endl;

	  std::cout << -Ainv << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -hessian.inverse() << std::endl;
	  std::cout << "----xxxx" << std::endl;
	  
	  Eigen::MatrixXd negative_inverse = -Ainv;//hessian.inverse();//Ainv;

          for (int i = 0; i < covar.size(); ++i) {
            covar(i) = negative_inverse(i);
	  }

	  /*for(int i = 0; i < covar.size(); i++) {
	    covar(i) = 0;
	  }

	  for(int i = 0; i < var.size(); i++) {
	    covar(i, i) = var(i);
	    }*/

          ++adapt_window_counter_;

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
