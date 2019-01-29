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
      explicit covar_experimental_adaptation(int n,
					     int lanczos_iterations,
					     int approximation_rank,
					     bool endpoint_only)
        : windowed_adaptation("covariance"), estimator_(n),
	  lanczos_iterations_(lanczos_iterations),
	  approximation_rank_(approximation_rank),
	  endpoint_only_(endpoint_only) {
      }

      template<typename Model>
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
	auto Ax = [&](const Eigen::MatrixXd& L, const Eigen::VectorXd& x) {
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

	/*Eigen::VectorXd r = Eigen::VectorXd::Random(q.size());
	double timestep = 0.0;
	r = r / r.norm();
	for(int i = 0; i < 5; i++) {
	  Eigen::VectorXd rr = Ax(inv_metric_L, r);
	  timestep = 1 / sqrt(-rr.transpose().dot(r) / r.norm());
	  r = rr / rr.norm();
	}*/

	if (adaptation_window()) {
          estimator_.add_sample(q);

	  qs2_.push_back(q);
	  if(!endpoint_only_ || (endpoint_only_ && end_adaptation_window())) {
	    qs_.push_back(q);
	  }
	}

        if (end_adaptation_window()) {
          compute_next_window();

	  /*estimator_.sample_covariance(covar);

          double n = static_cast<double>(estimator_.num_samples());
          covar = (n / (n + 5.0)) * covar
            + 1e-3 * (5.0 / (n + 5.0))
            * Eigen::MatrixXd::Identity(covar.rows(), covar.cols());

	  estimator_.restart();

	  Eigen::MatrixXd L_covar = covar.llt().matrixL();*/
	  
	  Eigen::VectorXd var;
          estimator_.sample_variance(var);

          double n = static_cast<double>(estimator_.num_samples());
          var = (n / (n + 5.0)) * var
                + 1e-3 * (5.0 / (n + 5.0)) * Eigen::VectorXd::Ones(var.size());

	  estimator_.restart();

	  Eigen::MatrixXd L_covar = var.array().sqrt().matrix().asDiagonal();
	  //Eigen::MatrixXd Linv = var.array().inverse().sqrt().matrix().asDiagonal();

	  //std::cout << L << std::endl;
	  //std::cout << "**" << std::endl;
	  //std::cout << Linv << std::endl;
	  //std::cout << "**" << std::endl;
	  
          std::cout << adapt_window_counter_ << std::endl;

	  int N = q.size();
	  int M = std::min((N - 1) / 2, approximation_rank_ + 1);

	  Eigen::MatrixXd H_avg = Eigen::MatrixXd::Zero(N, N);

	  //std::cout << "-----------     " << qs_.size() << std::endl;
	  
	  for(auto&& q : qs_) {
	    Eigen::MatrixXd vs = Eigen::MatrixXd::Zero(N, N);
	    Eigen::VectorXd a = Eigen::VectorXd::Zero(N);
	    Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

	    Eigen::VectorXd r = Eigen::VectorXd::Random(N);
	    vs.block(0, 0, N, 1) = r / r.norm();

	    r = Ax(L_covar, vs.block(0, 0, N, 1));
	    a(0) = (vs.block(0, 0, N, 1).transpose() * r)(0, 0);
	    r = r - a(0) * vs.block(0, 0, N, 1);
	    b(0) = r.norm();
	    int j = 1;
	    for(; j < N; j++) {
	      vs.block(0, j, N, 1) = r / b(j - 1);
	      r = Ax(L_covar, vs.block(0, j, N, 1));
	      r = r - vs.block(0, j - 1, N, 1) * b(j - 1);
	      a(j) = (vs.block(0, j, N, 1).transpose() * r)(0, 0);
	      r = r - vs.block(0, j, N, 1) * a(j);
	      r = r - vs * (vs.transpose() * r);
	      b(j) = r.norm();

	      if(j >= 2 * M) {
		Eigen::MatrixXd T = Eigen::MatrixXd::Zero(j + 1, j + 1);
		T(0, 0) = a(0);
		for(int m = 1; m < j + 1; m++) {
		  T(m, m) = a(m);
		  T(m, m - 1) = b(m - 1);
		  T(m - 1, m) = b(m - 1);
		}

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> T_decomp(T);
		double res = 0.0;
		for(int i = 0; i < M; i++) {
		  res = std::max(res, std::abs(b(j) * T_decomp.eigenvectors()(j, i)));
		  res = std::max(res, std::abs(b(j) * T_decomp.eigenvectors()(j, j - i)));
		}

		if(res < 1e-10) {
		  break;
		}
	      }
	    }

	    int J = std::min(N, j + 1);
	    
	    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(J, J);
	    T(0, 0) = a(0);
	    for(int m = 1; m < J; m++) {
	      T(m, m) = a(m);
	      T(m, m - 1) = b(m - 1);
	      T(m - 1, m) = b(m - 1);
	    }
	    //std::cout << "j : " << j << std::endl;
	    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> T_decomp(H_avg); 

	    H_avg = H_avg + vs.block(0, 0, N, J) * T * vs.block(0, 0, N, J).transpose();
	  }

	  H_avg = H_avg / qs_.size();

	  //std::cout << H_avg << std::endl;
	  
          Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp1(H_avg);
	  
	  /*double lp;
	  Eigen::VectorXd grad;
	  Eigen::MatrixXd hessian;
	  stan::math::hessian(log_prob_wrapper_covar<Model>(model), q, lp, grad, hessian);

	  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(hessian);*/

          /*Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(L_covar.transpose() * hessian * L_covar);

	  std::cout << "Approx: " << decomp1.eigenvalues().transpose() << std::endl;
	  std::cout << "Exact: " << decomp2.eigenvalues().transpose() << std::endl;*/
	  
	  int K = approximation_rank_;

	  if(K > (N - 1) / 2) {
	    throw std::runtime_error("Approximation rank must be less than or equal to the (number of parameters - 1) / 2");
	  }
	  
	  Eigen::VectorXd es2 = Eigen::VectorXd::Zero(N);
	  Eigen::MatrixXd vs2 = Eigen::MatrixXd::Zero(N, 2 * K);

	  //std::cout << decomp1.eigenvalues()(K) << " ==" << std::endl;
	  //std::cout << decomp1.eigenvalues()(N - K - 1) << " ++" << std::endl;
	  
	  for(int i = 0; i < K; i++) {
	    es2(i) = -std::abs(decomp1.eigenvalues()(i));
	    if(decomp1.eigenvalues()((N - K) + i) < 0) {
	      es2(K + i) = -std::abs(decomp1.eigenvalues()((N - K) + i));//std::abs(decomp1.eigenvalues()((N - K) + i) / decomp1.eigenvalues()(N - K - 1)); // / decomp1.eigenvalues()(N - K - 1));
	    } else {
	      es2(K + i) = -std::abs(decomp1.eigenvalues()(K));//
	      /// decomp1.eigenvalues()(K)std::abs(decomp1.eigenvalues()(K));//std::sqrt( * decomp1.eigenvalues()(N - K - 1)));
	    }
	    vs2.block(0, i, N, 1) = decomp1.eigenvectors().block(0, i, N, 1);//vs *
	    vs2.block(0, K + i, N, 1) = decomp1.eigenvectors().block(0, (N - K) + i, N, 1);
	  }

	  for(int i = 2 * K; i < N; i++) {
	    //es2(i) = -std::abs(decomp1.eigenvalues()(K));//std::sqrt( * ));
	    es2(i) = -std::sqrt(std::abs(decomp1.eigenvalues()(N - K - 1) * decomp1.eigenvalues()(K)));//std::abs(decomp1.eigenvalues()(K));
	  }

	  //std::cout << decomp1.eigenvalues().transpose() << std::endl;
	  //std::cout << es2.transpose() << std::endl;

	  Eigen::MatrixXd Vnull = Eigen::MatrixXd::Random(N, N - 2 * K);
	  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
	  Vnull = (Vnull - vs2 * (vs2.transpose() * Vnull)).eval();
	  V.block(0, 0, N, 2 * K) = vs2;
	  V.block(0, 2 * K, N, N - 2 * K) = stan::math::qr_thin_Q(Vnull);
	  
	  /*for(int i = 0; i < K; i++) {
	    std::cout << std::setprecision(3) << es(i) << " : " << vs2.block(0, i, 4, 1).transpose() << std::endl;
	  }
	  std::cout << "----" << std::endl;*/

	  /*std::cout << (V.transpose() * V) << std::endl;
	  std::cout << (V.transpose() * V).diagonal().transpose() << std::endl;
	  std::cout << "off diagonals: " << (V.transpose() * V).array().abs().sum() - N << std::endl;*/

	  //double etail = decomp1.eigenvalues()(K);

	  //std::cout << etail << std::endl;
	  
	  /*Eigen::VectorXd es2 = Eigen::VectorXd::Zero(N);
	  for(int i = 0; i < N; i++) {
	    if(i < K) {
	      es2(i) = es(i);
	    } else {
	      es2(i) = etail;////-1.0;-(V.block(0, i, N, 1).transpose() * LLT.solve(V.block(0, i, N, 1)))(0, 0);
	    }
	    }*/

	  //Eigen::MatrixXd A = L.transpose() * V * es2.asDiagonal() * V.transpose() * L;
	  //Eigen::MatrixXd Ainv = Linv * V * es2.array().inverse().matrix().asDiagonal() * V.transpose() * Linv.transpose();//-hessian.inverse()
	  Eigen::MatrixXd Ainv = L_covar * V * es2.array().inverse().matrix().asDiagonal() * V.transpose() * L_covar.transpose();

	  //std::cout << Ainv << std::endl;
	  //std::cout << Ainv * hessian << std::endl;
	  //std::cout << "+++++" << std::endl;
	  //Eigen::MatrixXd L2 = (-Ainv).llt().matrixL();
	  //std::cout << L2 << std::endl;

	  /*Eigen::MatrixXd qs = Eigen::MatrixXd::Zero(N, int(qs2_.size()));
	  for(int i = 0; i < qs2_.size(); i++) {
	    qs.block(0, i, N, 1) = qs2_[i];
	  }
	  Eigen::MatrixXd samples = decomp2.eigenvectors().transpose() * qs;
	  Eigen::VectorXd mean = Eigen::VectorXd::Zero(N);
	  Eigen::VectorXd var2 = Eigen::VectorXd::Zero(N);

	  for(int i = 0; i < samples.cols(); i++) {
	    for(int j = 0; j < samples.rows(); j++) {
	      mean(j) += samples(j, i);
	      //std::cout << samples.block(0, i, N, 1).transpose() << " " << qs.block(0, i, N, 1).transpose() << std::endl;
	    }
	  }

	  mean = mean / samples.cols();

	  for(int i = 0; i < samples.cols(); i++) {
	    for(int j = 0; j < samples.rows(); j++) {
	      var2(j) += std::pow(samples(j, i) - mean(j), 2);
	    }
	  }

	  var2 = var2 / (samples.cols() - 1);

	  //std::cout << "====1 " << var.transpose() << std::endl;
	  //std::cout << "====2 " << var2.transpose() << std::endl;

	  Eigen::MatrixXd covar2 = (decomp2.eigenvectors() * var2.asDiagonal() * decomp2.eigenvectors().transpose());*/

	  /*std::cout << A << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << hessian << std::endl;
	  std::cout << "----****" << std::endl;*/

	  //std::cout << Ainv * hessian << std::endl;
	  //std::cout << covar2 * hessian << std::endl;
	  //std::cout << "----!!!!" << std::endl;

	  /*std::cout << -A.inverse() << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -Ainv << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << -hessian.inverse() << std::endl;
	  std::cout << "----" << std::endl;
	  std::cout << covar << std::endl;
	  std::cout << "----xxxx" << std::endl;*/

	  //std::cout << Ainv << std::endl;

          for (int i = 0; i < covar.size(); ++i) {
            covar(i) = -Ainv(i);
	  }

          ++adapt_window_counter_;
	  qs_.clear();
	  qs2_.clear();
	  
          return true;
        }

        ++adapt_window_counter_;
        return false;
      }

    protected:
      stan::math::welford_var_estimator estimator_;
      int lanczos_iterations_;
      int approximation_rank_;
      bool endpoint_only_;
      std::vector< Eigen::VectorXd > qs_;
      std::vector< Eigen::VectorXd > qs2_;
    };

  }  // mcmc

}  // stan

#endif
