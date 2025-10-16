%% Part C Fisher LDA classifier (ROC + min P(error))
clearvars -except X y_true; close all; clc; rng(7,'twister');

% Priors (only for evaluating P(error))
P0 = 0.65; P1 = 0.35;

%% Generate/reuse the same 10K dataset
need_data = ~(exist('X','var')==1 && exist('y_true','var')==1 ...
    && size(X,2)==3 && numel(y_true)==size(X,1));
if need_data
    m0_true = [-1/2; -1/2; -1/2];
    C0_true = [ 1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
    m1_true = [1;1;1];
    C1_true = [ 1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];
    N = 10000;
    y = rand(N,1) >= P0;
    N0 = sum(y==0); N1 = sum(y==1);
    X = [mvnrnd(m0_true',C0_true,N0); mvnrnd(m1_true',C1_true,N1)];
    y_true = [zeros(N0,1); ones(N1,1)];
else
    m0_true = [-1/2; -1/2; -1/2];
    C0_true = [ 1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1];
    m1_true = [1;1;1];
    C1_true = [ 1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];
end

idx0 = (y_true==0); idx1 = ~idx0; n0 = sum(idx0); n1 = sum(idx1);
X0 = X(idx0,:); X1 = X(idx1,:);

%% --- Estimate class means/covariances (equal weights)
m0hat = mean(X0,1)'; m1hat = mean(X1,1)';
C0hat = cov(X0, 1); C1hat = cov(X1, 1); % ML covariance

% Fisher LDA direction: w = (C0hat + C1hat)^(-1) (m1hat - m0hat)
Sw = C0hat + C1hat;
dm = (m1hat - m0hat);
w = Sw \ dm; % more stable than inv(Sw)*dm

% 1-D scores and threshold sweep
z = X * w;
zmin = min(z) - 5*std(z); zmax = max(z) + 5*std(z);
tau = [-Inf, linspace(zmin, zmax, 500), +Inf];

TPR = zeros(size(tau));
FPR = zeros(size(tau));
Perr = zeros(size(tau));

for k = 1:numel(tau)
    D = z > tau(k);
    TPR(k) = mean(D(idx1)==1);
    FPR(k) = mean(D(idx0)==1);
    Perr(k)= FPR(k)*P0 + (1-TPR(k))*P1;
end

% Min-error point and AUC
[Perr_min, kmin] = min(Perr);
tau_min = tau(kmin);
TPR_min = TPR(kmin); FPR_min = FPR(kmin);
[Fs,ord] = sort(FPR); Ts = TPR(ord); AUC_lda = trapz(Fs, Ts);

fprintf('Fisher LDA: min P(error) = %.6f (%.2f%%) at tau=%.6f\n', Perr_min, 100*Perr_min, tau_min);
fprintf('At min-error: TPR=%.4f, FPR=%.4f\n', TPR_min, FPR_min);
fprintf('AUC (LDA) = %.4f\n', AUC_lda);

% --- Plot ROC with min-error marker (export as lda_roc_min.png)
figure; hold on; grid on; axis([0 1 0 1]); axis square;
plot(FPR, TPR, 'm-', 'LineWidth', 1.7);
plot([0 1],[0 1], 'k--');
scatter(FPR_min, TPR_min, 70, 'md', 'filled');
xlabel('False Positive Rate P(D=1 | L=0; \tau)');
ylabel('True Positive Rate P(D=1 | L=1; \tau)');
title('Fisher LDA ROC with minimum P(error) marker');
legend({'LDA ROC','Chance','Min-P(error)'}, 'Location','southEast');

% --- OPTIONAL: overlay with Optimal and Naive Bayes (export as roc_compare_all.png)
do_compare = true;
if do_compare
    % Optimal LLR
    lp0 = log_mvnpdf(X, m0_true', C0_true);
    lp1 = log_mvnpdf(X, m1_true', C1_true);
    llr_opt = lp1 - lp0;

    % Naive Bayes (=I) LLR (use sample means for fairness)
    d1 = sum((X - m1hat').^2, 2);
    d0 = sum((X - m0hat').^2, 2);
    llr_nb = -0.5*(d1 - d0);

    % Common grid
    Lmin = min([llr_opt; llr_nb; z]) - 8;
    Lmax = max([llr_opt; llr_nb; z]) + 8;
    lg = [-Inf, linspace(Lmin, Lmax, 500), +Inf];

    [TPR_opt,FPR_opt] = sweep_llr(llr_opt, lg, idx1, idx0, n1, n0);
    [TPR_nb, FPR_nb ] = sweep_llr(llr_nb, lg, idx1, idx0, n1, n0);

    figure; hold on; grid on; axis([0 1 0 1]); axis square;
    plot(FPR_opt, TPR_opt, 'b-', 'LineWidth', 1.6);
    plot(FPR_nb, TPR_nb, 'r-', 'LineWidth', 1.6);
    plot(FPR, TPR, 'm-', 'LineWidth', 1.6);
    plot([0 1],[0 1], 'k--');
    legend({'Optimal (true \Sigma)','Naive Bayes (\Sigma=I)','Fisher LDA','Chance'}, 'Location','southEast');
    xlabel('False Positive Rate'); ylabel('True Positive Rate');
    title('ROC comparison: Optimal vs Naive Bayes vs Fisher LDA');
end

%% Helpers
function [TPR,FPR] = sweep_llr(score, log_gamma, idx1, idx0, n1, n0)
TPR = zeros(size(log_gamma)); FPR = TPR;
for k = 1:numel(log_gamma)
    D = score > log_gamma(k);
    TPR(k) = sum(D(idx1)==1)/n1;
    FPR(k) = sum(D(idx0)==1)/n0;
end
end

function lp = log_mvnpdf(X, mu, Sigma)
[U,p] = chol(Sigma);
if p>0, error('Covariance not positive definite.'); end
d = size(X,2);
XC = bsxfun(@minus, X, mu);
sol = U \ XC';
q = sum(sol.^2,1)'; logdetS = 2*sum(log(diag(U)));
lp = -0.5*(q + d*log(2*pi) + logdetS);
end
