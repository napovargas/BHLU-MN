#include <RcppArmadillo.h>
#include <cmath>
#include <limits>
#include <time.h>
#include <algorithm>
#include <fstream>
#define ARMA_USE_SUPERLU 1

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins("cpp11")]]

arma::uvec mysetdiff(arma::uvec& x, arma::uvec& y){
  
  x = unique(x);
  y = unique(y);
  for (size_t j = 0; j < y.n_elem; j++) {
    arma::uvec q1 = arma::find(x == y[j]);
    x.shed_row(q1(0));
  }
  
  return x;
}

arma::uvec std_setdiff(arma::uvec& x, arma::uvec& y) {
  
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  
  return arma::conv_to<arma::uvec>::from(out);
}

// [[Rcpp::export]]
arma::mat ReshapeMat(arma::vec a, arma::uword p, arma::uword n, arma::uvec estim, arma::uvec last){
  arma::mat   B     = zeros(n, p);
  arma::vec   X     = zeros(1, 1);
  arma::uvec  idi   = regspace<uvec>(0, p - 2);
  arma::uvec  rowi  = zeros<uvec>(1, 1);
  for(uword i = 0; i < n; i++){
    rowi(0)         = i;
    X(0)            = std::max(1 - accu(a.elem(idi)), 0.0);
    B(rowi, estim)  = a.elem(idi).t();
    B(rowi, last) = X;
    idi             = idi + ones<uvec>(p - 1, 1)*(p - 1); 
  }
  return(B);
}

// Positive Gaussian distribution 
double rpnorm(double mean, double dev){
  const double  A     = 1.136717791056118;
  const double  pi    = M_PI;
  double        var   = dev*dev;
  double        v     = std::numeric_limits<double>::quiet_NaN();
  double        mA    = (1 - A*A)/A*dev;
  double        mC    = dev*std::sqrt(pi/2.0);
  double        a     = 0;
  double        z     = 0;
  double        rho   = 0;
  double        r     = 0;
  double        u     = 0;
  double        g     = 0;
  
  while(std::isnan(v)){
    if (mean < mA){
      a 	= (-mean + sqrt(mean*mean + 4.0*var))/2/var;
      z 	= -log(1.0 - runif(1)[0])/a;
      rho = exp(-(z - mean)*(z - mean)/2.0/var - a*(mean - z + a*var/2.0));
    }
    
    else if(mean <= 0){
      z 	= std::abs(rnorm(1)[0])*dev + mean;
      rho = (z >= 0.0)?1.0:0.0;
    }
    
    else if(mean < mC){
      r 	= (runif(1)[0] < mean/(mean + std::sqrt(pi/2.0)*dev))?1.0:0.0;
      u 	= runif(1)[0]*mean;
      g 	= std::abs(rnorm(1)[0]*dev) + mean;
      z 	= r*u + (1.0 - r)*g;
      rho = r*exp(-(z - mean)*(z - mean)/2.0/var) + (1.0 - r);
    }
    
    else {
      z 	= rnorm(1)[0]*dev + mean;
      rho = (z >= 0)?1.0:0.0;
    }
    
    if(runif(1)[0] < rho){
      v = z;
    }
  }
  return (v);
}

arma::vec Dirichlet(arma::vec alpha){
  arma::uword k = alpha.n_rows;
  arma::vec   y = zeros(k);
  arma::vec   x = zeros(k);
  for(uword j = 0; j < k; j++){
    y(j) = rgamma(1, alpha(j), 1)[0];
  }
  for(uword i = 0; i < k; i++){
    x(i) = y(i)/arma::accu(y);
  }
  return(x);
}

arma::mat Wishart(int const& nu, arma::mat const& V){
  
  int m = V.n_rows;
  mat T = zeros(m,m); 
  mat R = zeros(m,m);
  for(int i = 0; i < m; i++) {
    T(i,i) = sqrt(rchisq(1,nu-i)[0]); 
  }
  
  for(int j = 0; j < m; j++) {  
    for(int i = j+1; i < m; i++) {    
      T(i,j) = rnorm(1)[0]; 
    }}
  
  mat C = trans(T)*chol(V); 
  
  return R = trans(C)*C;
}

arma::mat InverseWishart(int const& nu, arma::mat const& V){ 
  
  int m   = V.n_rows;
  mat T   = zeros(m,m);
  mat IR  = zeros(m,m);
  for(int i = 0; i < m; i++) {
    T(i,i) = sqrt(rchisq(1,nu-i)[0]);
  }
  
  for(int j = 0; j < m; j++) {  
    for(int i = j+1; i < m; i++) {    
      T(i,j) = randn(1)[0]; 
    }}
  
  mat C   = trans(T)*chol(V);
  mat CI  = solve(trimatu(C),eye(m,m));  
  
  return IR = CI*trans(CI);
}


/* Truncated Gaussian  distribution */
/* Adapted from Matlab Code provided by Nicolas Dobigeon */
double TruncGauss(double x_old, double mu, double sigma, double mum, double mup){
  double      sigma2;
  int         accept;
  int         compt;
  double      x;
  double      z;
  double      d0;
  sigma2 		= sigma*sigma;
  mup 			= (mup - mu)/sigma;
  mum 			= (mum - mu)/sigma;
  
  accept 		= 0;
  compt 		= 0;
  x 			  = x_old;
  while ((accept == 0) && (compt < 200)) {
    compt++;
    z = runif(1)[0]*(mup - mum) + mum;
    if (0.0 < mum) {
      d0 = exp((mum * mum - z * z) / 2.0);
    } else if (mup < 0.0) {
      d0 = exp((mup * mup - z * z) / 2.0);
    } else {
      d0 = exp(-(z * z) / 2.0);
    }
    
    if (runif(1)[0] < d0) {
      x 		  = z;
      accept 	= 1;
    }
  }
  
  return (x * sqrt(sigma2) + mu) * (double)accept + x_old * (1.0 - (double)accept);
}

/* Metropolis step for truncated Gaussian */
/* Adapted from Matlab Code provided by Nicolas Dobigeon */
double TruncGaussMH(double X, double Mu, double Sigma, double Mum, double Mup){
  
  double    Mu_new;
  double    Mup_new;
  double    Z;
  double    Y;
  bool      b0;
  double    delta;
  Mu_new 	= Mu - Mum;
  Mup_new = Mup - Mum;
  if (Mu < Mup) {
    Z 		= rpnorm(Mu_new, Sigma);
  } else {
    delta 	= Mu_new - Mup_new;
    Mu_new 	= -delta;
    Z 		= rpnorm(Mu_new, Sigma);
    Z 		= -(Z - Mup_new);
  }
  
  Z += Mum;
  if ((Z <= Mup) && (Z >= Mum)) {
    b0 = true;
  } else {
    b0 = false;
  }
  
  Y = Z * (double)b0 + X*(double)!b0;
  return(Y);
}

/*     Sampling from Gaussian truncated on a simplex     */
/* Adapted from Matlab Code provided by Nicolas Dobigeon */
// [[Rcpp::export]]
vec MVGSimplex(vec S, vec const& Mean, mat const& Var){
  vec Mu            = Mean;
  uword p           = Mu.n_elem;
  uword j           = 0;
  vec Mu_sv         = zeros(p);
  vec Var_sv        = zeros(p);
  vec Sd_sv         = zeros(p);
  vec Sj;
  vec Muj;
  vec Rv(p - 1);
  uvec P            = linspace<uvec>(1, p, p);
  uvec shuffledP    = shuffle(P);
  mat vecSigma      = zeros(p - 1, p);
  mat Rm            = zeros(p, p);
  cube matSigma     = zeros(p - 1, p - 1, p);
  
  for(uword r = 0;r < p; r++){
    Rm = Var;
    Rm.shed_row(r);
    Rv = Rm.col(r);
    Rm.shed_col(r);
    matSigma.slice(r) = inv(Rm);
    vecSigma.col(r) = Rv;
  } 
  for(uvec::iterator jit = shuffledP.begin(); jit != shuffledP.end(); jit++){
    j               	= (int)*jit - 1;
    Sj              	= S;
    Sj.shed_row(j);
    Muj             	= Mu;
    Muj.shed_row(j);
    Mu_sv(j)        	= (Mu(j) + (trans(vecSigma.col(j))*matSigma.slice(j)*(Sj - Muj)))[0];
    Var_sv(j) 			  = (Var(j, j) - (trans(vecSigma.col(j))*matSigma.slice(j)*vecSigma.col(j)))[0];
    Sd_sv(j) 			    = sqrt(std::abs(Var_sv(j)));
    S(j) 				      = TruncGaussMH(S(j), Mu_sv(j), Sd_sv(j), 0.0, (1 - sum(S) + S(j)));
    
  }
  return(S);
}

arma::cube makeZ(arma::cube Y, arma::vec mp, arma::uword n, arma::uword q, arma::uword r){
  arma::cube Z  = zeros(r, q, n);
  arma::mat  Mp = repmat(mp, 1, q);
  for(uword i = 0; i < n; i++){
    Z.slice(i) = Y.slice(i) - Mp;
  }
  return(Z);
}

arma::cube myreshape(arma::mat A, arma::uword n, arma::uword q, arma::uword r){
  arma::cube X         = zeros(r, q, n);
  arma::uvec idq       = regspace<uvec>(0, q - 1);
  for(uword i = 0; i < n; i++){
    X.slice(i) = A.cols(idq);
    idq        = idq + ones<uvec>(q, 1)*q;
  }
  return(X);
}

arma::sp_mat removecol(arma::sp_mat A, arma::uword first, arma::uword last, arma::uword irow, arma::uword n){
  arma::sp_mat tmp1;
  arma::sp_mat tmp2;
  arma::sp_mat X;
  if(first == 0){
    X = A.submat(irow, last, irow, n - 1);
  } else if (last == (n - 1)){
    X = A.submat(irow, 0, irow, first);
  } else {
    tmp1 = A.submat(irow, 0, irow, first);
    tmp2 = A.submat(irow, last, irow, n - 1);
    X    = join_horiz(tmp1, tmp2);
  }
  return(X);
}

// [[Rcpp::export]]
List BHLU_MN(arma::cube Y, arma::mat M, arma::uword nIter, arma::uword burn, arma::uword thin){
  arma::uword     p           = M.n_cols;
  arma::uword     r           = M.n_rows;
  arma::uword     n           = Y.n_slices;
  arma::uword     q           = Y.n_cols;
  double          nu          = r + 2.0;
  double          gamma       = q + 2.0;
  double          pct         = 0;
  double          ttime       = 0;
  clock_t start;
  clock_t end;
  arma::uvec      spps        = regspace<uvec>(0, p - 1);
  arma::uvec      tmp         = zeros<uvec>(p);
  arma::uvec      estim       = zeros<uvec>(p - 1);
  arma::uvec      last        = zeros<uvec>(1);
  arma::uvec      idx         = regspace<uvec>(0, p - 2);
  arma::uvec      notidx;
  arma::uvec      allidx      = regspace<uvec>(0, n*q*(p - 1) - 1);
  //arma::uvec      keep        = regspace<uvec>(burn, thin, nIter);
  arma::vec       one         = ones<vec>(p - 1);
  arma::mat       M_p;
  arma::mat       mp; 
  arma::mat       Q;
  arma::sp_mat    Qtilde;
  arma::mat       Omega       = eye(r, r);
  arma::mat       OmegaInv    = inv_sympd(Omega);
  arma::mat       Phi         = eye(q, q);
  arma::mat       PhiInv      = inv_sympd(Phi);
  arma::mat       R           = zeros(q, q);
  arma::mat       V           = zeros(r, r);
  arma::mat       var         = zeros(p - 1, p - 1);
  arma::mat       mu          = zeros(p - 1);
  arma::mat       w;
  arma::mat       Tmp;
  arma::mat       ThetaE      = zeros(n*q, p);
  arma::vec       thetahat    = randu(n*q*(p - 1));
  arma::vec       s           = ones(r, 1);
  arma::vec       a           = ones(q, 1);
  arma::vec       thetatmp;
  arma::cube      Z           = zeros(r, q, n);
  arma::cube      Theta       = zeros(r, q, n);
  arma::cube      Mu          = zeros(r, q, n);
  arma::cube      E           = zeros(r, q, n);
  arma::sp_mat    SigmaInv;
  arma::mat       C;
  arma::sp_mat    Vinv        = speye(n*q*(p - 1), n*q*(p - 1))*0.01;
  arma::cube      StoreTheta  = zeros(nIter, p, n*q);
  arma::mat       StoreOmega  = zeros(nIter, r*r);
  arma::mat       StorePhi    = zeros(nIter, q*q);
  arma::mat       StoreS      = zeros(nIter, r);
  arma::mat       StoreA      = zeros(nIter, q);
  /*arma::cube      KeepTheta;
  arma::mat       KeepOmega;
  arma::mat       KeepPhi;
  arma::mat       KeepS;
  arma::mat       KeepA;*/
  List Out;
  
  start = clock();
  for(uword iter = 0; iter < nIter; iter++){
    spps                = regspace<uvec>(0, p - 1);
    tmp                 = shuffle(spps);
    last                = tmp(0);
    estim               = std_setdiff(spps, last);
    M_p                 = M.cols(estim);
    mp                  = M.cols(last);
    Q                   = M_p - mp*one.t();
    Qtilde              = kron(eye(q*n, q*n), Q);
    SigmaInv            = kron(eye(n, n), kron(PhiInv, OmegaInv));
    C                   = Qtilde.t()*SigmaInv*Qtilde + Vinv;
    Z                   = makeZ(Y, mp, n, q, r);
    w                   = Qtilde.t()*SigmaInv*vectorise(Z);
    allidx              = regspace<uvec>(0, n*q*(p - 1) - 1);
    idx                 = regspace<uvec>(0, p - 2);
    R                   = zeros(r, r);
    V                   = zeros(q, q);
    for(uword i = 0; i < (n*q); i++){
      notidx            = std_setdiff(allidx, idx);
      var               = inv_sympd(C(idx, idx));
      mu                = var*(w(idx) - C(idx, notidx)*thetahat(notidx));
      if(p == 2){
        thetatmp        = thetahat(idx);
        thetahat(idx(0))= TruncGauss(thetatmp(0), mu(0), sqrt(var(0, 0)), 0.0, 1.0);
      } else {
        thetahat(idx)   = MVGSimplex(thetahat(idx), mu, var);
      }
      idx               = idx + ones<uvec>(p - 1, 1)*(p - 1);
    }
    Tmp                 = reshape(thetahat, n*q, p - 1);
    ThetaE.cols(estim)  = Tmp;
    ThetaE.cols(last)   = ones(n*q) - sum(Tmp, 1);
    Mu                  = myreshape(M*ThetaE.t(), n, q, r);
    E                   = Y - Mu;
    for(uword i = 0; i < n; i++){
      R                 = R + E.slice(i)*PhiInv*E.slice(i).t();
      V                 = V + E.slice(i).t()*OmegaInv*E.slice(i);
    }
    Omega               = InverseWishart(nu + r + n*q - 1, inv_sympd(R + 2*nu*diagmat(s)));
    OmegaInv            = inv_sympd(Omega);
    Phi                 = InverseWishart(gamma + q + n*r - 1, inv_sympd(V + 2*gamma*diagmat(a)));
    PhiInv              = inv_sympd(Phi);
    for(uword k = 0; k < r; k++){
      s(k)              = 1/randg(1, distr_param((nu + r)/2, 1/(nu*OmegaInv(k, k) + 1/100)))[0];
    }
    for(uword j = 0; j < q; j++){
      a(j)              = 1/randg(1, distr_param((gamma + q)/2, 1/(gamma*PhiInv(j, j) + 1/100)))[0];
    }
    StoreTheta(span(iter), span::all, span::all)  = ThetaE.t();
    StoreOmega.row(iter)                          = vectorise(Omega).t();
    StorePhi.row(iter)                            = vectorise(Phi).t();
    StoreS.row(iter)                              = s.t();
    StoreA.row(iter)                              = a.t();
    if(iter % 200 == 0){
      Rcpp::Rcout.precision(2);
      Rcpp::checkUserInterrupt();
      pct = (double)iter/(double)nIter;
      Rcpp::Rcout << " Iteration " << iter + 1 << "/" << nIter << " [" 
                  << std::fixed << (pct*100.00) << "%] "<< std::endl;
    }
  }
  end = clock();
  ttime = ((double) (end - start)) / CLOCKS_PER_SEC;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << " Wrapping up! " << std::endl;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << nIter << " iterations in " << ttime << " seconds" << std::endl;
  
  //KeepTheta    = StoreTheta.rows(keep);
  //KeepOmega    = StoreOmega.rows(keep);
  //KeepPhi      = StorePhi.rows(keep);

  Out["Theta"] = StoreTheta;
  Out["Omega"] = StoreOmega;
  Out["Phi"]   = StorePhi;
  Out["S"]     = StoreS;
  Out["A"]     = StoreA;
  Out["Time"]  = ttime;
  return(Out);
}

// [[Rcpp::export]]
List BHLU_MNI(arma::cube Y, arma::mat M, arma::uword nIter, arma::uword burn, arma::uword thin){
  arma::uword     p           = M.n_cols;
  arma::uword     r           = M.n_rows;
  arma::uword     n           = Y.n_slices;
  arma::uword     q           = Y.n_cols;
  double          nu          = r + 2.0;
  double          gamma       = q + 2.0;
  double          pct         = 0;
  double          ttime       = 0;
  clock_t start;
  clock_t end;
  arma::uvec      spps        = regspace<uvec>(0, p - 1);
  arma::uvec      tmp         = zeros<uvec>(p);
  arma::uvec      estim       = zeros<uvec>(p - 1);
  arma::uvec      last        = zeros<uvec>(1);
  arma::uvec      idx         = regspace<uvec>(0, p - 2);
  arma::uvec      notidx;
  arma::uvec      idi;
  arma::uvec      idq;
  //arma::uvec      keep        = regspace<uvec>(burn, thin, nIter);
  arma::vec       one         = ones<vec>(p - 1);
  arma::mat       M_p;
  arma::mat       mp; 
  arma::mat       Q;
  arma::sp_mat    Qtilde;
  arma::mat       Omega       = eye(r, r);
  arma::mat       OmegaInv    = inv_sympd(Omega);
  arma::mat       Phi         = eye(q, q);
  arma::mat       PhiInv      = inv_sympd(Phi);
  arma::mat       R           = zeros(q, q);
  arma::mat       V           = zeros(r, r);
  arma::mat       var         = zeros(p - 1, p - 1);
  arma::mat       mu          = zeros(p - 1);
  arma::mat       w;
  arma::mat       Tmp;
  arma::mat       ThetaE      = randu(n*q, p);
  arma::vec       thetahat    = randu(n*q*(p - 1));
  arma::vec       thetarep    = randu(q*(p - 1));
  arma::vec       s           = ones(r, 1);
  arma::vec       a           = ones(q, 1);
  arma::vec       thetatmp;
  arma::cube      Z           = zeros(r, q, n);
  arma::cube      Theta       = zeros(r, q, n);
  arma::cube      Mu          = zeros(r, q, n);
  arma::cube      E           = zeros(r, q, n);
  arma::mat       SigmaInv;
  arma::mat       C;
  arma::mat       mpMat;
  arma::sp_mat    Vinv        = speye(q*(p - 1), q*(p - 1))*0.01;
  arma::cube      StoreTheta  = zeros(nIter, p, n*q);
  arma::mat       StoreOmega  = zeros(nIter, r*r);
  arma::mat       StorePhi    = zeros(nIter, q*q);
  arma::mat       StoreS      = zeros(nIter, r);
  arma::mat       StoreA      = zeros(nIter, q);
  arma::mat       saveVar(n, p - 1);
  arma::mat       saveMu(n, p - 1);
  /*arma::cube      KeepTheta;
  arma::mat       KeepOmega;
  arma::mat       KeepPhi;
  arma::mat       KeepS;
  arma::mat       KeepA;*/
  List Out;
  
  start = clock();
  for(uword iter = 0; iter < nIter; iter++){
    spps                = regspace<uvec>(0, p - 1);
    tmp                 = shuffle(spps);
    last                = tmp(0);
    estim               = std_setdiff(spps, last);
    M_p                 = M.cols(estim);
    mp                  = M.cols(last);
    Q                   = M_p - mp*one.t();
    Qtilde              = kron(eye(q, q), Q);
    SigmaInv            = kron(PhiInv, OmegaInv);
    C                   = Qtilde.t()*SigmaInv*Qtilde + Vinv;
    idq                 = regspace<uvec>(0, q*(p - 1) - 1);
    idx                 = regspace<uvec>(0, q*(p - 1) - 1);
    R                   = zeros(r, r);
    V                   = zeros(q, q);
    mpMat               = repmat(mp, 1, q);
    thetahat            = vectorise(ThetaE.cols(estim).t());
    for(uword i = 0; i < n; i++){
      w                 = Qtilde.t()*SigmaInv*vectorise(Y.slice(i) - mpMat);
      idi               = regspace<uvec>(0, p - 2);
      thetarep          = thetahat(idq);
      for(uword j = 0; j < q; j++){
        notidx          = std_setdiff(idx, idi);
        var             = inv_sympd(C(idi, idi));
        mu              = var*(w(idi) - C(idi, notidx)*thetarep(notidx));
        if(p == 2){
          thetatmp          = thetarep(idi);
          thetarep(idi(0))  = TruncGauss(thetatmp(0), mu(0), sqrt(var(0, 0)), 0.0, 1.0);
        } else {
          thetatmp          = thetarep(idi);
          thetarep(idi)     = MVGSimplex(thetatmp, mu, var);
        }
        idi             = idi + ones<uvec>(p - 1, 1)*(p - 1);
      }
      thetahat(idq)     = thetarep;
      idq               = idq + ones<uvec>(q*(p - 1), 1)*(q*(p - 1));
    }
    ThetaE              = ReshapeMat(thetahat, p, q*n, estim, last);
    Mu                  = myreshape(M*ThetaE.t(), n, q, r);
    E                   = Y - Mu;
    for(uword i = 0; i < n; i++){
      R                 = R + E.slice(i)*PhiInv*E.slice(i).t();
      V                 = V + E.slice(i).t()*OmegaInv*E.slice(i);
    }
    Omega               = InverseWishart(nu + r + n*q - 1, inv_sympd(R + 2*nu*diagmat(s)));
    OmegaInv            = inv_sympd(Omega);
    Phi                 = InverseWishart(gamma + q + n*r - 1, inv_sympd(V + 2*gamma*diagmat(a)));
    PhiInv              = inv_sympd(Phi);
    for(uword k = 0; k < r; k++){
      s(k)              = 1/randg(1, distr_param((nu + r)/2, 1/(nu*OmegaInv(k, k) + 1/100)))[0];
    }
    for(uword j = 0; j < q; j++){
      a(j)              = 1/randg(1, distr_param((gamma + q)/2, 1/(gamma*PhiInv(j, j) + 1/100)))[0];
    }
    StoreTheta(span(iter), span::all, span::all)  = ThetaE.t();
    StoreOmega.row(iter)                          = vectorise(Omega).t();
    StorePhi.row(iter)                            = vectorise(Phi).t();
    StoreS.row(iter)                              = s.t();
    StoreA.row(iter)                              = a.t();
    if(iter % 200 == 0){
      Rcpp::Rcout.precision(2);
      Rcpp::checkUserInterrupt();
      pct = (double)iter/(double)nIter;
      Rcpp::Rcout << " Iteration " << iter + 1 << "/" << nIter << " [" 
                  << std::fixed << (pct*100.00) << "%] "<< std::endl;
    }
  }
  end = clock();
  //thetahat.save("ThetaHat.txt", arma_ascii);
  ttime = ((double) (end - start)) / CLOCKS_PER_SEC;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << " Wrapping up! " << std::endl;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << nIter << " iterations in " << ttime << " seconds" << std::endl;
  
  //KeepTheta    = StoreTheta.rows(keep);
  //KeepOmega    = StoreOmega.rows(keep);
  //KeepPhi      = StorePhi.rows(keep);
  
  Out["Theta"] = StoreTheta;
  Out["Omega"] = StoreOmega;
  Out["Phi"]   = StorePhi;
  Out["S"]     = StoreS;
  Out["A"]     = StoreA;
  Out["Time"]  = ttime;
  return(Out);
}

// [[Rcpp::export]]
List BHLU_MNNRM(arma::cube Y, arma::mat M, arma::mat Ginv, arma::uword nIter, arma::uword burn, arma::uword thin){
  arma::uword     p           = M.n_cols;
  arma::uword     r           = M.n_rows;
  arma::uword     n           = Y.n_slices;
  arma::uword     q           = Y.n_cols;
  arma::uword     to_r, from_r, to_q, from_q;
  double          nu          = r + 2.0;
  double          gamma       = q + 2.0;
  double          pct         = 0;
  double          ttime       = 0;
  clock_t start;
  clock_t end;
  arma::uvec      spps        = regspace<uvec>(0, p - 1);
  arma::uvec      tmp         = zeros<uvec>(p);
  arma::uvec      estim       = zeros<uvec>(p - 1);
  arma::uvec      last        = zeros<uvec>(1);
  arma::uvec      idx         = regspace<uvec>(0, p - 2);
  arma::uvec      notidx;
  arma::uvec      allidx      = regspace<uvec>(0, n*q*(p - 1) - 1);
  arma::vec       one         = ones<vec>(p - 1);
  arma::mat       M_p;
  arma::mat       mp; 
  arma::mat       Q;
  arma::sp_mat    Qtilde;
  arma::mat       Omega       = eye(r, r);
  arma::mat       OmegaInv    = inv_sympd(Omega);
  arma::mat       Phi         = eye(q, q);
  arma::mat       PhiInv      = inv_sympd(Phi);
  arma::mat       R           = zeros(q, q);
  arma::mat       V           = zeros(r, r);
  arma::mat       var         = zeros(p - 1, p - 1);
  arma::mat       mu          = zeros(p - 1);
  arma::mat       w;
  arma::mat       Tmp;
  arma::mat       ThetaE      = zeros(n*q, p);
  arma::vec       thetahat    = randu(n*q*(p - 1));
  arma::vec       s           = ones(r, 1);
  arma::vec       a           = ones(q, 1);
  arma::vec       thetatmp;
  arma::cube      Z           = zeros(r, q, n);
  arma::cube      Theta       = zeros(r, q, n);
  arma::cube      Mu          = zeros(r, q, n);
  arma::cube      E           = zeros(r, q, n);
  arma::mat       LE          = zeros(r, q*n);
  arma::mat       WE          = zeros(q, r*n);
  arma::sp_mat    SigmaInv;
  arma::mat       C;
  arma::sp_mat    Vinv        = speye(n*q*(p - 1), n*q*(p - 1))*0.1;
  arma::cube      StoreTheta  = zeros(nIter, p, n*q);
  arma::mat       StoreOmega  = zeros(nIter, r*r);
  arma::mat       StorePhi    = zeros(nIter, q*q);
  arma::mat       StoreS      = zeros(nIter, r);
  arma::mat       StoreA      = zeros(nIter, q);
  List Out;
  
  start = clock();
  for(uword iter = 0; iter < nIter; iter++){
    spps                = regspace<uvec>(0, p - 1);
    tmp                 = shuffle(spps);
    last                = tmp(0);
    estim               = std_setdiff(spps, last);
    M_p                 = M.cols(estim);
    mp                  = M.cols(last);
    Q                   = M_p - mp*one.t();
    Qtilde              = kron(eye(q*n, q*n), Q);
    SigmaInv            = kron(Ginv, kron(PhiInv, OmegaInv));
    C                   = Qtilde.t()*SigmaInv*Qtilde + Vinv;
    Z                   = makeZ(Y, mp, n, q, r);
    w                   = Qtilde.t()*SigmaInv*vectorise(Z);
    allidx              = regspace<uvec>(0, n*q*(p - 1) - 1);
    idx                 = regspace<uvec>(0, p - 2);
    R                   = zeros(r, r);
    V                   = zeros(q, q);
    for(uword i = 0; i < (n*q); i++){
      notidx            = std_setdiff(allidx, idx);
      var               = inv_sympd(C(idx, idx));
      mu                = var*(w(idx) - C(idx, notidx)*thetahat(notidx));
      if(p == 2){
        thetatmp        = thetahat(idx);
        thetahat(idx(0))= TruncGauss(thetatmp(0), mu(0), sqrt(var(0, 0)), 0.0, 1.0);
      } else {
        thetahat(idx)   = MVGSimplex(thetahat(idx), mu, var);
      }
      idx               = idx + ones<uvec>(p - 1, 1)*(p - 1);
    }
    Tmp                 = reshape(thetahat, n*q, p - 1);
    ThetaE.cols(estim)  = Tmp;
    ThetaE.cols(last)   = ones(n*q) - sum(Tmp, 1);
    Mu                  = myreshape(M*ThetaE.t(), n, q, r);
    E                   = Y - Mu;
    from_r              = 0;
    to_r                = q - 1;
    from_q              = 0;
    to_q                = r - 1;
    for(uword i = 0; i < n; i++){
      LE.cols(from_r, to_r) = E.slice(i);
      WE.cols(from_q, to_q) = E.slice(i).t();
      to_r              = to_r + q;
      from_r            = from_r + q;
      to_q              = to_q + r;
      from_q            = from_q + r;
    }
	R                   = LE*kron(Ginv, PhiInv)*LE.t();
	V                   = WE*kron(Ginv, OmegaInv)*WE.t();
    Omega               = InverseWishart(nu + r + n*q - 1, inv_sympd(R + 2*nu*diagmat(s)));
    OmegaInv            = inv_sympd(Omega);
    Phi                 = InverseWishart(gamma + q + n*r - 1, inv_sympd(V + 2*gamma*diagmat(a)));
    PhiInv              = inv_sympd(Phi);
    for(uword k = 0; k < r; k++){
      s(k)              = 1/randg(1, distr_param((nu + r)/2, 1/(nu*OmegaInv(k, k) + 1/100)))[0];
    }
    for(uword j = 0; j < q; j++){
      a(j)              = 1/randg(1, distr_param((gamma + q)/2, 1/(gamma*PhiInv(j, j) + 1/100)))[0];
    }
    StoreTheta(span(iter), span::all, span::all)  = ThetaE.t();
    StoreOmega.row(iter)                          = vectorise(Omega).t();
    StorePhi.row(iter)                            = vectorise(Phi).t();
    StoreS.row(iter)                              = s.t();
    StoreA.row(iter)                              = a.t();
    if(iter % 200 == 0){
      Rcpp::Rcout.precision(2);
      Rcpp::checkUserInterrupt();
      pct = (double)iter/(double)nIter;
      Rcpp::Rcout << " Iteration " << iter + 1 << "/" << nIter << " [" 
                  << std::fixed << (pct*100.00) << "%] "<< std::endl;
    }
  }
  end = clock();
  ttime = ((double) (end - start)) / CLOCKS_PER_SEC;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << " Wrapping up! " << std::endl;
  Rcpp::Rcout << "                                               " << std::endl;
  Rcpp::Rcout << nIter << " iterations in " << ttime << " seconds" << std::endl;
  
  //KeepTheta    = StoreTheta.rows(keep);
  //KeepOmega    = StoreOmega.rows(keep);
  //KeepPhi      = StorePhi.rows(keep);
  
  Out["Theta"] = StoreTheta;
  Out["Omega"] = StoreOmega;
  Out["Phi"]   = StorePhi;
  Out["S"]     = StoreS;
  Out["A"]     = StoreA;
  Out["Time"]  = ttime;
  return(Out);
}
