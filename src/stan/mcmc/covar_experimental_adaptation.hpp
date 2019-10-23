#ifndef STAN_MCMC_COVAR_EXPERIMENTAL_ADAPTATION_HPP
#define STAN_MCMC_COVAR_EXPERIMENTAL_ADAPTATION_HPP

#include <stan/math/rev/mat.hpp>
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

    class MultiNormalInvWishart {
    private:
      int k_;
      int v_;
      Eigen::VectorXd mu_;
      Eigen::MatrixXd Lambda_;

    public:
      MultiNormalInvWishart() :
	k_(0), v_(0), mu_(0), Lambda_(0, 0) {
      }
      
      MultiNormalInvWishart(int k, int n, const Eigen::VectorXd& mu, const Eigen::MatrixXd& cov) :
	k_(k), v_(n + mu.rows() + 1), mu_(mu), Lambda_(n * cov) {
	if(mu.rows() != cov.rows()) {
	  throw std::runtime_error("Mean must have same number of rows as covariance");
	}

	if(cov.cols() != cov.rows()) {
	  throw std::runtime_error("Covariance matrix must be square");
	}
      }

      MultiNormalInvWishart(int n, const Eigen::MatrixXd& cov) :
	MultiNormalInvWishart(0, n, Eigen::VectorXd::Zero(cov.rows()), cov) {
      }

      void update(const Eigen::MatrixXd& Y) {
	int N = Y.rows();
	int M = Y.cols();

	Eigen::VectorXd Ymean = Y.rowwise().mean();
	Eigen::MatrixXd Y_minus_Ymean = Y.colwise() - Ymean;
	Eigen::MatrixXd Y_minus_mu_ = Y.colwise() - mu_;

	//std::cout << "M1: " << std::endl << mean() << std::endl;
	
	Lambda_ = Lambda_ + Y_minus_Ymean * Y_minus_Ymean.transpose() + (k_ * M / double(k_ + M)) * Y_minus_mu_ * Y_minus_mu_.transpose();

	mu_ = (k_ / double(k_ + M)) * mu_ + (M / double(k_ + M)) * Ymean;
	k_ = k_ + M;
	v_ = v_ + M;

	//std::cout << "M2: " << std::endl << mean() << std::endl;
      }

      Eigen::MatrixXd mean() const {
	return Lambda_ / (v_ - Lambda_.cols() - 1.0);
      }
    };
    
    class covar_experimental_adaptation: public windowed_adaptation {
    public:
      explicit covar_experimental_adaptation(int n,
					     int which_adaptation)
        : windowed_adaptation("covariance"),
	  which_adaptation_(which_adaptation) {
      }

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

      Eigen::MatrixXd ledoit_wolf_2004(const Eigen::MatrixXd& Y) {
	Eigen::MatrixXd centered = Y.colwise() - Y.rowwise().mean();
	int Neff = std::max(centered.cols() - 1.0, 1.0);
	Eigen::MatrixXd S = centered * centered.transpose() / Neff;

	double mn = 0.0;
	for(int i = 0; i < S.cols(); ++i)
	  mn += S(i, i) / S.cols();

	double d2 = (S - Eigen::MatrixXd::Identity(S.rows(), S.cols()) * mn).squaredNorm();

	double b2bar = 0.0;
	for(int k = 0; k < centered.cols(); ++k) {
	  Eigen::MatrixXd outer = centered.block(0, k, centered.rows(), 1) * centered.block(0, k, centered.rows(), 1).transpose();
	  b2bar += (outer - S).squaredNorm() / (Neff * Neff);
	}

	double b2 = std::min(b2bar, d2);

	double a2 = d2 - b2;

	Eigen::MatrixXd Sstar = Eigen::MatrixXd::Identity(S.rows(), S.cols()) * (b2 / d2) * mn + (a2 / d2) * S;
	
	return Sstar;
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

      MultiNormalInvWishart lw_2004_metric(const Eigen::MatrixXd& Y,
					   const Eigen::MatrixXd& L,
					   const Eigen::MatrixXd& Linv) {
	return MultiNormalInvWishart(Y.cols(), L * ledoit_wolf_2004(Linv * Y) * L.transpose());
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

	Eigen::VectorXd eigenvalues;
	Eigen::MatrixXd eigenvectors;

	lanczos(Nev + 1, model, L, qs.block(0, 0, N, 1), eigenvalues, eigenvectors);
	avg = avg + eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();

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
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, const Eigen::VectorXd& q, double& stability_limit) {
	if (adaptation_window()) {
	  qs_.push_back(q);
	}

        if (end_adaptation_window()) {
          compute_next_window();

	  int N = q.size(); // N parameters
	  int M = qs_.size(); // M draws

	  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(N, M);
	  std::vector<int> idxs(M);
	  for(int i = 0; i < qs_.size(); i++)
	    idxs[i] = i;

	  std::random_shuffle(idxs.begin(), idxs.end());
	  for(int i = 0; i < qs_.size(); i++)
	    Y.block(0, i, N, 1) = qs_[idxs[i]];

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
	      Eigen::MatrixXd Dinv = D.diagonal().array().inverse().matrix().asDiagonal();
	      
	      Eigen::MatrixXd cov_test_lw = D * ledoit_wolf_2004(Dinv * Ytest) * D;

	      std::map<std::string, MultiNormalInvWishart> inv_metrics;
	      inv_metrics["diagonal"] = diagonal_metric(Ytrain.cols(), cov);
	      inv_metrics["dense"] = dense_metric(Ytrain.cols(), cov);
	      inv_metrics["lw2004"] = lw_2004_metric(Ytrain, D, Dinv);
	      inv_metrics["rank 1"] = low_rank_metric(1, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1));
	      if(N > 2)
		inv_metrics["rank 2"] = low_rank_metric(2, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1));
	      if(N > 4)
		inv_metrics["rank 4"] = low_rank_metric(4, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1));
	      if(N > 8)
		inv_metrics["rank 8"] = low_rank_metric(8, model, D, Ytrain.block(0, Ytrain.cols() - 1, N, 1));

	      inv_metrics["rank 1 + wishart"] = inv_metrics["rank 1"];
	      inv_metrics["rank 1 + wishart"].update(Ytrain);

	      if(N > 2) {
		inv_metrics["rank 2 + wishart"] = inv_metrics["rank 2"];
		inv_metrics["rank 2 + wishart"].update(Ytrain);
	      }
	    
	      if(N > 4) {
		inv_metrics["rank 4 + wishart"] = inv_metrics["rank 4"];
		inv_metrics["rank 4 + wishart"].update(Ytrain);
	      }
	    
	      if(N > 8) {
		inv_metrics["rank 8 + wishart"] = inv_metrics["rank 8"];
		inv_metrics["rank 8 + wishart"].update(Ytrain);
	      }

	      if(state == "selection") {
		double best_score = -1.0;

		int which = 1; // which == 0 corresponds to switching adaptation
		for(const auto& it : inv_metrics) {
		  double low_eigenvalue = bottom_eigenvalue_estimate(it.second.mean(), cov_test_lw);
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

	    covar = last_metric_.mean();
	  } catch(const std::exception& e) {
	    std::cout << e.what() << std::endl;
	    std::cout << "Exception while using experimental adaptation, falling back to diagonal" << std::endl;
	    Eigen::MatrixXd cov = covariance(Y);
	    covar = diagonal_metric(M, cov).mean();
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
