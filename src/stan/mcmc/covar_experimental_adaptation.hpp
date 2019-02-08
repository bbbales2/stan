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
	  first_call_(true),
	  lanczos_iterations_(lanczos_iterations),
	  approximation_rank_(approximation_rank),
	  endpoint_only_(endpoint_only),
	  last_metric_n_(0) {
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
	return centered * centered.transpose() / std::max(centered.rows() - 1.0, 1.0);
      }

      Eigen::MatrixXd diagonal_metric(int N,
				      const Eigen::MatrixXd& cov) {
	Eigen::VectorXd var = cov.diagonal();
	return ((N / (N + 5.0)) * var
		+ 1e-3 * (5.0 / (N + 5.0)) * Eigen::VectorXd::Ones(var.size())).asDiagonal();
      }

      Eigen::MatrixXd dense_metric(int N,
				   const Eigen::MatrixXd& cov) {
	return (N / (N + 5.0)) * cov +
	  1e-3 * (5.0 / (N + 5.0)) * Eigen::MatrixXd::Identity(cov.rows(), cov.cols());
      }

      Eigen::MatrixXd sparse_metric(int N,
				    const Eigen::MatrixXd& cov,
				    const Eigen::MatrixXd& sparsity) {
	Eigen::MatrixXd dense = dense_metric(N, cov);

	return (dense.inverse().array() * sparsity.array()).matrix().inverse();
      }

      template<typename Model>
      Eigen::MatrixXd low_rank_metric(int Nev,
				      const Model& model,
				      const Eigen::MatrixXd& L,
				      const Eigen::MatrixXd& qs,
				      const Eigen::MatrixXd& Y) {
	int N = qs.rows();
	Eigen::MatrixXd avg = Eigen::MatrixXd::Zero(N, N);

	if(Nev >= N) {
	  throw std::runtime_error("Approximation must be less than the number of parameters");
	}

	for(int i = 0; i < qs.cols(); i++) {
	  Eigen::VectorXd eigenvalues;
	  Eigen::MatrixXd eigenvectors;

	  lanczos(Nev + 1, model, L, qs.block(0, i, N, 1), eigenvalues, eigenvectors);

	  avg = avg + eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
	}

	avg = avg / qs.cols();

	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> avg_decomp(avg);

	Eigen::VectorXd eigenvalues = avg_decomp.eigenvalues().block(0, 0, Nev + 1, 1);
	Eigen::MatrixXd eigenvectors = avg_decomp.eigenvectors().block(0, 0, N, Nev + 1);
	
	Eigen::VectorXd abs_inv_eigenvalues = eigenvalues.array().abs().inverse().matrix();
	Eigen::VectorXd approx_e = abs_inv_eigenvalues.block(0, 0, Nev, 1).array() - abs_inv_eigenvalues(Nev);
	Eigen::MatrixXd approx = eigenvectors.block(0, 0, N, Nev) *
	  approx_e.asDiagonal() *
	  eigenvectors.block(0, 0, N, Nev).transpose();
	
	for(int i = 0; i < approx.rows(); i++)
	  approx(i, i) = approx(i, i) + abs_inv_eigenvalues(Nev);
	
	return L * approx * L.transpose();
      }

      Eigen::MatrixXd wishart_metric(int n0,
				     const Eigen::MatrixXd& prior,
				     const Eigen::MatrixXd& Y) {
	int N = Y.rows();
	Eigen::VectorXd u0 = Eigen::VectorXd::Zero(N);
	double M = Y.cols();
	double k0 = 0;
	double v0 = n0 + N + 1;
	Eigen::MatrixXd Lambda0 = n0 * prior;

	Eigen::VectorXd Ymean = Y.rowwise().mean();
	Eigen::VectorXd un = (k0 / (k0 + M)) * u0 + (M / (k0 + M)) * Ymean;
	double kn = k0 + M;
	double vn = v0 + M;

	Eigen::MatrixXd Y_minus_Ymean = Y.colwise() - Ymean;
	Eigen::MatrixXd Y_minus_u0 = Y.colwise() - u0;

	return Lambda0 + Y_minus_Ymean * Y_minus_Ymean.transpose() + (k0 * M / (k0 + M)) * Y_minus_u0 * Y_minus_u0.transpose();
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

	//std::cout << decomp.eigenvalues().transpose() << std::endl;
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
      double condition_number_estimate(const Model& model, const Eigen::MatrixXd& metric,
				       const Eigen::MatrixXd& covar, const Eigen::VectorXd& q) {
	return top_eigenvalue(model, metric, q) / bottom_eigenvalue_estimate(metric, covar);
      }

      template<typename Model>
      bool learn_covariance(const Model& model, Eigen::MatrixXd& covar, const Eigen::VectorXd& q, double& stability_limit) {
	if (adaptation_window()) {
          estimator_.add_sample(q);

	  qs_.push_back(q);
	}

        if (end_adaptation_window()) {
          compute_next_window();

	  int N = q.size();
	  int M = qs_.size();

	  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(N, M);
	  std::vector<int> idxs(N);
	  for(int i = 0; i < qs_.size(); i++)
	    idxs[i] = i;

	  std::random_shuffle(idxs.begin(), idxs.end());

	  //for(auto state : { "selection", "refinement" }) {
	  for(int i = 0; i < qs_.size(); i++)
	    Y.block(0, i, N, 1) = qs_[idxs[i]];

	  Eigen::MatrixXd Ytrain = Y.block(0, 0, N, int(0.8 * Y.cols()));
	  Eigen::MatrixXd Ytest = Y.block(0, Ytrain.cols(), N, Y.cols() - Ytrain.cols());

	  Eigen::MatrixXd cov = covariance(Ytrain);
	  Eigen::MatrixXd cov_test = covariance(Ytest);
	  //std::cout << "cov: " << cov << std::endl;
	  //std::cout << "dense: " << dense_metric(Y.cols(), cov) << std::endl;

	  //Eigen::MatrixXd cov_hessian = -hessian(model, q).inverse();
	  
	  if(first_call_) {
	    sparsity_ = hessian(model, q);
	
	    //std::cout << sparsity_.diagonal().transpose() << std::endl;
	    for(int i = 0; i < sparsity_.size(); i++) {
	      //sparsity_(i) = (std::abs(sparsity_(i)) > cutoff) ? sparsity_(i) : 0.0;
	      sparsity_(i) = (std::abs(sparsity_(i)) < 1e-300) ? 1.0 : 0.0;
	    }
	    sparsity_ = ((sparsity_ + sparsity_.transpose()) / 2.0).eval();
	    for(int i = 0; i < sparsity_.rows(); i++)
	      sparsity_(i, i) = 1.0;
	  }

	  std::map<std::string, Eigen::MatrixXd> metrics;
	  metrics["diagonal"] = diagonal_metric(Ytrain.cols(), cov);
	  metrics["dense"] = dense_metric(Ytrain.cols(), cov);
	  //metrics["sparse"] = sparse_metric(Y.cols(), cov, sparsity_);
	  Eigen::MatrixXd D = cov.diagonal().array().sqrt().matrix().asDiagonal();
	  metrics["rank 1"] =
	    low_rank_metric(1, model, D,
			    Ytrain.block(0, Ytrain.cols() - lanczos_iterations_, N, lanczos_iterations_), Ytrain);
	  metrics["rank 0.03 * N"] =
	    low_rank_metric(1 + int((N - 1) * 0.03), model, D,
			    Ytrain.block(0, Ytrain.cols() - lanczos_iterations_, N, lanczos_iterations_), Ytrain);
	  metrics["rank 0.3 * N"] =
	    low_rank_metric(1 + int((N - 1) * 0.3), model, D,
			    Ytrain.block(0, Ytrain.cols() - lanczos_iterations_, N, lanczos_iterations_), Ytrain);
	  metrics["rank 1 + wishart"] = wishart_metric(N, metrics["rank 1"], Ytrain);
	  metrics["rank 0.03 * N + wishart"] = wishart_metric(N, metrics["rank 0.03 * N"], Ytrain);
	  metrics["rank 0.3 * N + wishart"] = wishart_metric(N, metrics["rank 0.3 * N"], Ytrain);
	  if(!first_call_) {
	    metrics["last metric"] = last_metric_;
	    metrics["last metric + wishart"] = wishart_metric(last_metric_n_, metrics["last metric"], Ytrain);
	  }
	  /*std::cout << "sparsity: " << std::endl <<
	    sparsity_ << std::endl <<
	    "------" << std::endl;*/

	  //std::cout << sparsity_.diagonal().transpose() << std::endl;
	  //std::cout << metrics["sparse metric"].diagonal().transpose() << std::endl;

	  double best_score = -1.0;
	  
	  for(const auto& it : metrics) {
	    Eigen::VectorXd c = Eigen::VectorXd::Zero(std::min(7, int(Ytest.cols())));
	    //Eigen::VectorXd c2 = Eigen::VectorXd::Zero(lanczos_iterations_);
	    for(int i = 0; i < c.size(); i++) {
	      c(i) = std::sqrt(condition_number_estimate(model, it.second,
							 cov_test, Ytest.block(0, Ytest.cols() - i - 1, N, 1)));
	      //c2(i) = std::sqrt(condition_number_estimate(model, it.second,
	      //					 cov_hessian, qs_[qs_.size() - i - 1]));
	    }
	    std::sort(c.data(), c.data() + c.size());
	    //std::cout << "metric: " << std::endl <<
	    //  it.second.block(0, 0, it.second.rows(), it.second.cols()) << std::endl <<
	    //  "------" << std::endl;
	    /*std::cout << "hessian inverse: " << std::endl <<
	      hessian(model, q).inverse() << std::endl <<
	      "------" << std::endl;*/
	    double min = c.minCoeff();
	    double max = c.maxCoeff();
	    double median = c(c.size() / 2);

	    std::map<int, std::string> choose_metric;
	    choose_metric[1] = "dense";
	    choose_metric[2] = "diagonal";
	    choose_metric[3] = "last metric";
	    choose_metric[4] = "last metric + wishart";
	    choose_metric[5] = "rank 0.03 * N";
	    choose_metric[6] = "rank 0.03 * N + wishart";
	    choose_metric[7] = "rank 0.3 * N";
	    choose_metric[8] = "rank 0.3 * N + wishart";
	    choose_metric[9] = "rank 1";
	    choose_metric[10] = "rank 1 + wishart";

	    //std::cout << c.transpose() << std::endl;
	    std::cout << "adapt: " << adapt_window_counter_ << ", min: " << min << ", median: " << median << ", max: " << max << ", " << it.first << std::endl;
	    //std::cout << "adapt (full hessian): " << adapt_window_counter_ << ", " << it.first << ", min: " << c2.minCoeff() << ", max: " << c2.maxCoeff();
	    if(best_score < 0.0 || median < best_score) {
	      if(approximation_rank_ == 0 || (adapt_window_counter_ < 400 && approximation_rank_ > 0) || (adapt_window_counter_ > 400 && approximation_rank_ > 0 && choose_metric[approximation_rank_] == it.first)) {
		best_score = median;
		last_metric_ = it.second;

		//if(approximation_rank_ > 0 && adapt_window_counter_ > 400) {
		  std::cout << "picking: " << it.first << std::endl;
		  //}

		if(it.first == "diagonal") {
		  last_metric_n_ = M;
		} else if(it.first == "dense") {
		  last_metric_n_ = M;
		} else if(it.first == "rank 1") {
		  last_metric_n_ = N;
		} else if(it.first == "rank 0.03 * N") {
		  last_metric_n_ = N;
		} else if(it.first == "rank 0.3 * N") {
		  last_metric_n_ = N;
		} else if(it.first == "rank 1 + wishart") {
		  last_metric_n_ = N + M;
		} else if(it.first == "rank 0.03 * N + wishart") {
		  last_metric_n_ = N + M;
		} else if(it.first == "rank 0.3 * N + wishart") {
		  last_metric_n_ = N + M;
		} else if(it.first == "last metric") {
		  last_metric_n_ = last_metric_n_;
		} else if(it.first == "last metric + wishart") {
		  last_metric_n_ = last_metric_n_ + M;
		}
	      }
	    }
	  }

	  stability_limit = 1e300;
	  for(int i = 0; i < std::min(7, int(Ytest.cols())); i++) {
	    stability_limit = std::min(stability_limit,
				       2 / std::sqrt(std::abs(top_eigenvalue(model, last_metric_, Ytest.block(0, Ytest.cols() - i - 1, N, 1)))));
	  }

	  /*double lp;
	  Eigen::VectorXd grad;
	  Eigen::MatrixXd hessian;
	  stan::math::hessian(log_prob_wrapper_covar<Model>(model), q, lp, grad, hessian);

	  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(hessian);*/

          /*Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomp2(L_covar.transpose() * hessian * L_covar);

	  std::cout << "Approx: " << decomp1.eigenvalues().transpose() << std::endl;
	  std::cout << "Exact: " << decomp2.eigenvalues().transpose() << std::endl;*/

	  /*int K = approximation_rank_;

	  if(K > N - 1) {
	    throw std::runtime_error("Approximation rank must be less than the number of parameters");
	  }
	  
	  Eigen::VectorXd es2 = Eigen::VectorXd::Zero(N);
	  Eigen::MatrixXd vs2 = Eigen::MatrixXd::Zero(N, K);*/

	  //std::cout << decomp1.eigenvalues()(K) << " ==" << std::endl;
	  //std::cout << decomp1.eigenvalues()(N - K - 1) << " ++" << std::endl;

	  /*std::cout << decomp1.eigenvalues().transpose() << std::endl;
	  
	  for(int i = 0; i < K; i++) {
	    es2(i) = -std::abs(decomp1.eigenvalues()(i));
	    vs2.block(0, i, N, 1) = decomp1.eigenvectors().block(0, i, N, 1);//vs *
	  }

	  for(int i = K; i < N; i++) {	
	    es2(i) = -std::abs(decomp1.eigenvalues()(K));//std::sqrt( * ));
	    }*/

	  //std::cout << decomp1.eigenvalues().transpose() << std::endl;
	  //std::cout << es2.transpose() << std::endl;

	  /*Eigen::MatrixXd Vnull = Eigen::MatrixXd::Random(N, N - K);
	  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(N, N);
	  Vnull = (Vnull - vs2 * (vs2.transpose() * Vnull)).eval();
	  V.block(0, 0, N, K) = vs2;
	  V.block(0, K, N, N - K) = stan::math::qr_thin_Q(Vnull);*/
	  
	  /*for(int i = 0; i < K; i++) {
	    std::cout << std::setprecision(3) << es(i) << " : " << vs2.block(0, i, 4, 1).transpose() << std::endl;
	  }
	  std::cout << "----" << std::endl;*/

	  //std::cout << es2 << std::endl;
	  //std::cout << (V.transpose() * V) << std::endl;
	  /*std::cout << (V.transpose() * V).diagonal().transpose() << std::endl;
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
	  //Eigen::MatrixXd negative_Ainv = -L_covar * V * es2.array().inverse().matrix().asDiagonal() * V.transpose() * L_covar.transpose();

	  //std::cout << "negative Ainv" << std::endl;
	  //std::cout << negative_Ainv << std::endl;
	  //std::cout << "----" << std::endl;
	  
	  /*Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(N, int(qs_.size()));
	  for(int i = 0; i < qs_.size(); i++) {
	    Y.block(0, i, N, 1) = qs_[i];
	  }
	  Eigen::VectorXd Ymean = Eigen::VectorXd::Zero(N);

	  for(int i = 0; i < Y.cols(); i++) {
	    for(int j = 0; j < Y.rows(); j++) {
	      Ymean(j) += Y(j, i);
	    }
	  }
	  
	  Ymean = Ymean / Y.cols();
	  
	  //std::cout << Ymean << std::endl;
				
	  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(N);
	  double M = Y.cols();
	  double k0 = 0;
	  double n0 = N;
	  double v0 = n0 + N + 1;
	  Eigen::MatrixXd Lambda0 = n0 * negative_Ainv;

	  //std::cout << n0 << " " << k0 << " " << v0 << std::endl;
  
	  Eigen::VectorXd un = (k0 / (k0 + M)) * u0 + (M / (k0 + M)) * Ymean;
	  double kn = k0 + M;
	  double vn = v0 + M;
	  Eigen::MatrixXd Y_minus_Ymean = Y;
	  Eigen::MatrixXd Y_minus_u0 = Y;
	  for(int i = 0; i < Y.cols(); i++) {
	    Y_minus_Ymean.block(0, i, N, 1) -= Ymean;
	    Y_minus_u0.block(0, i, N, 1) -= u0;
	    }*/

	  //std::cout << "M: " << M << std::endl;
	  //std::cout << kn << " " << vn << std::endl;
	  //std::cout << Y_minus_Ymean.block(0, 0, N, 1) << std::endl;
	  //std::cout << Y_minus_u0.block(0, 0, N, 1) << std::endl;
	  //Eigen::MatrixXd Lambdan = Lambda0 + Y_minus_Ymean * Y_minus_Ymean.transpose() + (k0 * M / (k0 + M)) * Y_minus_u0 * Y_minus_u0.transpose();
	  //std::cout << Lambdan << std::endl;
	  //std::cout << Y_minus_Ymean * Y_minus_Ymean.transpose() << std::endl;
	  //std::cout << (k0 * M / (k0 + M)) * Y_minus_u0 * Y_minus_u0.transpose() << std::endl;
	  //Eigen::MatrixXd wishart_mean = Lambdan / (vn - N - 1);
	  //if(endpoint_only_) {
	  //  wishart_mean = negative_Ainv;
	  //}

	  //riwish(vn, Lambdan)

	  //std::cout << Ainv << std::endl;
	  //std::cout << negative_Ainv * hessian << std::endl;
	  //std::cout << wishart_mean * hessian << std::endl;
	  //std::cout << "+++++" << std::endl;
	  //Eigen::MatrixXd L2 = (-Ainv).llt().matrixL();
	  //std::cout << L2 << std::endl;

	  /*Eigen::MatrixXd qs = Eigen::MatrixXd::Zero(N, int(qs_.size()));
	  for(int i = 0; i < qs_.size(); i++) {
	    qs.block(0, i, N, 1) = qs_[i];
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
            covar(i) = last_metric_(i);//-Ainv(i);
	  }

          ++adapt_window_counter_;
	  qs_.clear();
	  first_call_ = false;
	  
          return true;
        }

        ++adapt_window_counter_;
        return false;
      }

    protected:
      stan::math::welford_var_estimator estimator_;
      bool first_call_;
      int lanczos_iterations_;
      int approximation_rank_;
      bool endpoint_only_;
      Eigen::MatrixXd sparsity_;
      Eigen::MatrixXd last_metric_;
      int last_metric_n_;
      std::vector< Eigen::VectorXd > qs_;
    };

  }  // mcmc

}  // stan

#endif
