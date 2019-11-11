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
      explicit var_experimental_adaptation(int n,
					   int which_adaptation)
        : windowed_adaptation("variance"),
	  which_adaptation_(which_adaptation) {}

      template<typename Model>
      Eigen::VectorXd Ax(const Model& model,
			 const Eigen::MatrixXd& L,
			 const Eigen::VectorXd& q,
			 const Eigen::VectorXd& x) {
	double lp;
	Eigen::VectorXd grad1;
	Eigen::VectorXd grad2;
	//stan::math::hessian_times_vector(log_prob_wrapper_covar<Model>(model), q, x, lp, Ax);
	double dx = 1e-5;
	Eigen::VectorXd dr = L * x * dx;
	stan::math::gradient(log_prob_wrapper_covar<Model>(model), q + dr / 2.0, lp, grad1);
	stan::math::gradient(log_prob_wrapper_covar<Model>(model), q - dr / 2.0, lp, grad2);
	return L.transpose() * (grad1 - grad2) / dx;
      }

      template<typename Model>
      int lanczos(int Nev,
		  const Model& model,
		  const Eigen::MatrixXd& L,
		  const Eigen::VectorXd& q,
		  Eigen::VectorXd& eigenvalues,
		  Eigen::MatrixXd& eigenvectors) {
	int N = q.size();

	Eigen::MatrixXd vs = Eigen::MatrixXd::Zero(N, N);
	Eigen::VectorXd a = Eigen::VectorXd::Zero(N);
	Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

	Eigen::VectorXd r = Eigen::VectorXd::Random(N);
	vs.block(0, 0, N, 1) = r / r.norm();

	r = Ax(model, L, q, vs.block(0, 0, N, 1));
	a(0) = (vs.block(0, 0, N, 1).transpose() * r)(0, 0);
	r = r - a(0) * vs.block(0, 0, N, 1);
	b(0) = r.norm();
	int j = 1;
	for(; j < N; j++) {
	  vs.block(0, j, N, 1) = r / b(j - 1);
	  r = Ax(model, L, q, vs.block(0, j, N, 1));
	  r = r - vs.block(0, j - 1, N, 1) * b(j - 1);
	  a(j) = (vs.block(0, j, N, 1).transpose() * r)(0, 0);
	  r = r - vs.block(0, j, N, 1) * a(j);
	  r = r - vs * (vs.transpose() * r);
	  b(j) = r.norm();
	    
	  if(j >= Nev) {
	    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(j + 1, j + 1);
	    T(0, 0) = a(0);
	    for(int m = 1; m < j + 1; m++) {
	      T(m, m) = a(m);
	      T(m, m - 1) = b(m - 1);
	      T(m - 1, m) = b(m - 1);
	    }
	    
	    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> T_decomp(T);
	    double res = 0.0;
	    for(int i = 0; i < Nev; i++) {
	      res = std::max(res, std::abs(b(j) * T_decomp.eigenvectors()(j, i)));
	    }

	    if(res < 1e-10) {
	      break;
	    }
	  }
	}

	int M = std::min(N, j + 1);
	Eigen::MatrixXd T = Eigen::MatrixXd::Zero(M, M);
	T(0, 0) = a(0);
	for(int m = 1; m < M; m++) {
	  T(m, m) = a(m);
	  T(m, m - 1) = b(m - 1);
	  T(m - 1, m) = b(m - 1);
	}

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> T_decomp(T);
	eigenvalues = T_decomp.eigenvalues().block(0, 0, Nev, 1);
	eigenvectors = vs.block(0, 0, N, M) * T_decomp.eigenvectors().block(0, 0, M, Nev);

	return j;
      }

      Eigen::MatrixXd covariance(const Eigen::MatrixXd& Y) {
	Eigen::MatrixXd centered = Y.colwise() - Y.rowwise().mean();
	return centered * centered.transpose() / std::max(centered.cols() - 1.0, 1.0);
      }

      MultiNormalInvWishart diagonal_metric(int N,
				      const Eigen::MatrixXd& cov) {
	Eigen::VectorXd var = cov.diagonal();
	return MultiNormalInvWishart(N + 5, ((N / (N + 5.0)) * var
					     + 1e-3 * (5.0 / (N + 5.0)) * Eigen::VectorXd::Ones(var.size())).asDiagonal());
      }

      MultiNormalInvWishart dense_metric(int N,
					 const Eigen::MatrixXd& cov) {
	return MultiNormalInvWishart(N + 5, (N / (N + 5.0)) * cov +
				     1e-3 * (5.0 / (N + 5.0)) * Eigen::MatrixXd::Identity(cov.rows(), cov.cols()));
      }

      template<typename Model>
      MultiNormalInvWishart low_rank_metric(int Nev,
					    const Model& model,
					    const Eigen::MatrixXd& L,
					    const Eigen::MatrixXd& qs) {
	int N = qs.rows();
	Eigen::MatrixXd avg = Eigen::MatrixXd::Zero(N, N);

	if(Nev >= N) {
	  throw std::runtime_error("Approximation must be less than the number of parameters");
	}

	if(qs.cols() > 1) {
	  throw std::runtime_error("The averaging isn't working right now");
	}

	//for(int i = 0; i < qs.cols(); i++) {
	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;

	lanczos(Nev + 1, model, L, qs.block(0, 0, N, 1), eigenvalues, eigenvectors);

	avg = avg + eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
	//}

	//avg = avg / qs.cols();

	//Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> avg_decomp(avg);

	//Eigen::VectorXd eigenvalues = avg_decomp.eigenvalues().block(0, 0, Nev + 1, 1);
	//Eigen::MatrixXd eigenvectors = avg_decomp.eigenvectors().block(0, 0, N, Nev + 1);

	Eigen::VectorXd abs_inv_eigenvalues = eigenvalues.array().abs().inverse().matrix();
	Eigen::VectorXd approx_e = abs_inv_eigenvalues.block(0, 0, Nev, 1).array() - abs_inv_eigenvalues(Nev);
	Eigen::MatrixXd approx = eigenvectors.block(0, 0, N, Nev) *
	  approx_e.asDiagonal() *
	  eigenvectors.block(0, 0, N, Nev).transpose();
	
	for(int i = 0; i < approx.rows(); i++)
	  approx(i, i) = approx(i, i) + abs_inv_eigenvalues(Nev);
	
	return MultiNormalInvWishart(N, L * approx * L.transpose());
      }

      /*template<typename Model>
      double stability_limit(const Model& model, const Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
	Eigen::MatrixXd L = covar.llt().matrixL();
	Eigen::VectorXd r = Eigen::VectorXd::Random(q.size());
	double stepsize = 0.0;
	r = r / r.norm();
	for(int i = 0; i < 5; i++) {
	  Eigen::VectorXd rr = Ax(L, q, r);
	  stepsize = 2.0 / sqrt(-rr.transpose().dot(r) / r.norm());
	  r = rr / rr.norm();
	}
	
	std::cout << "stability limit sample: " << stability_limit << std::endl;
	return stepsize;
	}*/

      template<typename Model>
      double top_eigenvalue(const Model& model, const Eigen::MatrixXd& metric, const Eigen::VectorXd& q) {
	Eigen::MatrixXd L = metric.llt().matrixL();
	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;

	lanczos(1, model, L, q, eigenvalues, eigenvectors);

	//std::cout << q.transpose() << std::endl;
	//std::cout << "top: " << eigenvalues(0) << std::endl;
	
	return eigenvalues(0);
      }

      double bottom_eigenvalue_estimate(const Eigen::MatrixXd& metric, const Eigen::MatrixXd& covar) {
	Eigen::MatrixXd L = metric.llt().matrixL();
	Eigen::MatrixXd S = L.template triangularView<Eigen::Lower>().
	  solve(L.template triangularView<Eigen::Lower>().solve(covar).transpose()).transpose();
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp(S);

	//std::cout << "bottom: " << decomp.eigenvalues().transpose() << std::endl;
	double eval = -1.0 / decomp.eigenvalues()(decomp.eigenvalues().size() - 1);
	
	//std::cout << "bottom: " << eval << std::endl;

	return eval;
      }

      template<typename Model>
      Eigen::MatrixXd hessian(const Model& model, const Eigen::VectorXd& q) {
	double lp;
	Eigen::VectorXd grad;
	Eigen::MatrixXd hessian_tmp;
	stan::math::hessian(log_prob_wrapper_covar<Model>(model), q, lp, grad, hessian_tmp);
	return hessian_tmp;
      }

      template<typename Model>
      bool learn_variance(const Model& model, Eigen::VectorXd& var, const Eigen::VectorXd& q, double& stability_limit) {
	if (adaptation_window()) {
	  qs_.push_back(q);
	}

        if (end_adaptation_window()) {
          compute_next_window();

	  int N = q.size();
	  int M = qs_.size();

	  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(N, M);
	  std::vector<int> idxs(M);
	  for(int i = 0; i < qs_.size(); i++)
	    idxs[i] = i;

	  std::random_shuffle(idxs.begin(), idxs.end());
	  for(int i = 0; i < qs_.size(); i++)
	    Y.block(0, i, N, 1) = qs_[idxs[i]];

	  /*Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(hessian);*/

	  try {
	    std::string best_metric;
	    for(auto state : { "selection", "refinement" }) {
	      Eigen::MatrixXd Ytrain;
	      Eigen::MatrixXd Ytest;

	      if(state == "selection") {
		int Ntest;
		Ntest = int(0.2 * Y.cols());
		if(Ntest < 5) {
		  Ntest = 5;
		}

		if(Y.cols() < 10) {
		  throw std::runtime_error("Each warmup stage must have at least 10 samples");
		}
	      
		std::cout << "train: " << Y.cols() - Ntest << ", test: " << Ntest << std::endl;
		Ytrain = Y.block(0, 0, N, Y.cols() - Ntest);
		Ytest = Y.block(0, Ytrain.cols(), N, Ntest);
	      } else {
		Ytrain = Y;
	      }

	      Eigen::MatrixXd cov = covariance(Ytrain);
	      Eigen::MatrixXd cov_test = covariance(Ytest);

	      Eigen::MatrixXd D = cov.diagonal().array().sqrt().matrix().asDiagonal();

	      //std::cout << "am I being used?" << std::endl;

	      std::map<std::string, MultiNormalInvWishart> inv_metrics;
	      inv_metrics["diagonal"] = diagonal_metric(Ytrain.cols(), cov);
	      inv_metrics["rank 1"] = MultiNormalInvWishart(Ytrain.rows(), low_rank_metric(1, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1)).mean().diagonal().asDiagonal());
	      if(N > 2)
		inv_metrics["rank 2"] = MultiNormalInvWishart(Ytrain.rows(), low_rank_metric(2, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1)).mean().diagonal().asDiagonal());
	      if(N > 4)
		inv_metrics["rank 4"] = MultiNormalInvWishart(Ytrain.rows(), low_rank_metric(4, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1)).mean().diagonal().asDiagonal());
	      if(N > 8)
		inv_metrics["rank 8"] = MultiNormalInvWishart(Ytrain.rows(), low_rank_metric(8, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1)).mean().diagonal().asDiagonal());
	    
	      if(state == "selection") {
		double best_score = -1.0;
	  
		int which = 1; // which == 0 corresponds to switching adaptation
		for(const auto& it : inv_metrics) {
		  double low_eigenvalue = bottom_eigenvalue_estimate(it.second.mean(), cov_test);
		  Eigen::VectorXd c = Eigen::VectorXd::Zero(std::min(5, int(Ytest.cols())));
		  for(int i = 0; i < c.size(); i++) {

		    /*double lp;
		      Eigen::VectorXd grad;
		      Eigen::MatrixXd hessian;
		      stan::math::hessian(log_prob_wrapper_covar<Model>(model), Ytest.block(0, i, N, 1), lp, grad, hessian);
		    
		      std::cout << "q: " << q.transpose() << std::endl;
		      std::cout << hessian << std::endl;
		      std::cout << "-----" << std::endl;*/

		    double high_eigenvalue = top_eigenvalue(model, it.second.mean(), Ytest.block(0, i, N, 1));
		    c(i) = std::sqrt(high_eigenvalue / low_eigenvalue);
		  }
		  std::sort(c.data(), c.data() + c.size());
		  double min = c.minCoeff();
		  double max = c.maxCoeff();
		  double median = c(c.size() / 2);
	      
		  std::cout << "adapt: " << adapt_window_counter_ << ", which: " << which << ", min: " << min << ", median: " << median << ", max: " << max << ", " << it.first << std::endl;
		  if(best_score < 0.0 || max < best_score) {
		    best_score = max;
		    best_metric = it.first;
		  }

		  which += 1;
		}
	      } else {
		if(which_adaptation_ == 0) {
		  last_metric_ = inv_metrics[best_metric];
		  std::cout << "picked: " << adapt_window_counter_ << ", name: " << best_metric << std::endl;
		} else {
		  if(which_adaptation_ - 1 < inv_metrics.size()) {
		    auto it = inv_metrics.begin();
		    std::advance(it, which_adaptation_ - 1);
		    last_metric_ = it->second;
		    std::cout << "picked: " << adapt_window_counter_ << ", name: " << it->first << std::endl;
		  } else {
		    throw std::runtime_error("which_adaptation_ too high");
		  }
		}

		stability_limit = 1e300;
		for(int i = 0; i < std::min(5, int(Y.cols())); i++) {
		  stability_limit = std::min(stability_limit,
					     2 / std::sqrt(std::abs(top_eigenvalue(model, last_metric_.mean(), Y.block(0, Y.cols() - i - 1, N, 1)))));
		}
	      }
	    }

	    var = last_metric_.mean().diagonal();
	  } catch(const std::exception& e) {
	    std::cout << e.what() << std::endl;
	    std::cout << "Exception while using experimental adaptation, falling back to diagonal" << std::endl;
	    if(Y.cols() > 1) {
	      Eigen::MatrixXd cov = covariance(Y);
	      var = diagonal_metric(M, cov).mean().diagonal();
	    } else {
	      var = Eigen::VectorXd::Ones(q.size());
	    }
	  }

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

          ++adapt_window_counter_;
	  qs_.clear();
	  
          return true;
        }
	
        ++adapt_window_counter_;
        return false;
      }

    protected:
      int which_adaptation_;
      MultiNormalInvWishart last_metric_;
      std::vector< Eigen::VectorXd > qs_;
    };

  }  // mcmc

}  // stan
#endif
