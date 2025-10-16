function nb_and_comparison_plots()
% ---------------------------------------------------------------
% Two separate plots:
% (Fig 1) Naive Bayes only (C0=C1=I) with MAP & min-error markers
% (Fig 2) Comparison overlay: True ERM (quadratic) vs Naive Bayes
% ---------------------------------------------------------------

clear; clc; close all; rng default;

%% 1) Generate ONE dataset (use exact prior counts for stability)
N = 10000; P0 = 0.65; P1 = 0.35;
N0 = round(P0*N); N1 = N - N0;

m0 = [-0.5; -0.5; -0.5];
C0 = [ 1 -0.5 0.3; -0.5 1 -0.5; 0.3 -0.5 1 ];
m1 = [1; 1; 1];
C1 = [ 1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1 ];

r0 = mvnrnd(m0, C0, N0);
r1 = mvnrnd(m1, C1, N1);
X = [r0; r1];
L = [zeros(N0,1); ones(N1,1)];
idx0 = (L==0); idx1 = (L==1);

%% 2) Build LLRs for both models

% --- True ERM (quadratic discriminant) ---
C0inv = inv(C0); C1inv = inv(C1);
logdetC0 = 2*sum(log(diag(chol(C0,'lower'))));
logdetC1 = 2*sum(log(diag(chol(C1,'lower'))));
Xm0 = X - m0.'; Xm1 = X - m1.';
q0 = sum((Xm0*C0inv).*Xm0, 2);
q1 = sum((Xm1*C1inv).*Xm1, 2);
llr_true = -0.5*(q1 - q0) - 0.5*(logdetC1 - logdetC0);

% --- Naive Bayes (identity covariances) ---
d0sq = sum((X - m0.').^2, 2);
d1sq = sum((X - m1.').^2, 2);
llr_nb = -0.5*(d1sq - d0sq);

%% 3) ROC arrays
[FPR_t,TPR_t,Perr_t,gamma_t] = roc_from_llr(llr_true, idx0, idx1, P0, P1);
[FPR_n,TPR_n,Perr_n,gamma_n] = roc_from_llr(llr_nb, idx0, idx1, P0, P1);

% AUCs
auc_true = trapz(FPR_t,TPR_t);
auc_nb = trapz(FPR_n,TPR_n);

% MAP & min-error indices
gamma_map = P0/P1; tmap = log(gamma_map);
[~,kmap_t] = min(abs(log(gamma_t)-tmap));
[~,kmap_n] = min(abs(log(gamma_n)-tmap));

[~,kmin_t] = min(Perr_t);
[~,kmin_n] = min(Perr_n);

%% 4) FIGURE 1: Naive Bayes only
figure(1); clf; hold on; grid on;
plot(FPR_n,TPR_n,'b-','LineWidth',1.6,'DisplayName','ROC (Naive Bayes)');
plot([0 1],[0 1],'k--','LineWidth',0.7,'DisplayName','Chance');
plot(FPR_n(kmap_n),TPR_n(kmap_n),'ro','MarkerFaceColor','r','DisplayName','MAP (NB)');
plot(FPR_n(kmin_n),TPR_n(kmin_n),'p','MarkerEdgeColor','k','MarkerFaceColor','y',...
     'MarkerSize',10,'DisplayName','Min P_e (NB)');
xlabel('False Positive Rate P(D=1 | L=0 ; \gamma)');
ylabel('True Positive Rate P(D=1 | L=1 ; \gamma)');
title('Naive Bayes ROC (C_0=C_1=I) with MAP & Min-Error points');
legend('Location','SouthEast'); axis([0 1 0 1]);

%% 5) FIGURE 2: Comparison overlay
figure(2); clf; hold on; grid on;
plot(FPR_t,TPR_t,'m-','LineWidth',1.6,'DisplayName','True ERM ROC');
plot(FPR_n,TPR_n,'b-','LineWidth',1.6,'DisplayName','Naive Bayes ROC');
plot([0 1],[0 1],'k--','LineWidth',0.7,'DisplayName','Chance');
plot(FPR_t(kmap_t),TPR_t(kmap_t),'mo','MarkerFaceColor','m','DisplayName','True MAP');
plot(FPR_n(kmap_n),TPR_n(kmap_n),'bo','MarkerFaceColor','b','DisplayName','NB MAP');
plot(FPR_t(kmin_t),TPR_t(kmin_t),'mp','MarkerFaceColor','y','MarkerSize',10,'DisplayName','True Min P_e');
plot(FPR_n(kmin_n),TPR_n(kmin_n),'bp','MarkerFaceColor','y','MarkerSize',10,'DisplayName','NB Min P_e');
xlabel('False Positive Rate P(D=1 | L=0 ; \gamma)');
ylabel('True Positive Rate P(D=1 | L=1 ; \gamma)');
title('Comparison: True ERM vs Naive Bayes (same dataset)');
legend('Location','SouthEast'); axis([0 1 0 1]);

%% 6) Print comparison table
fprintf('\n==== COMPARISON (same dataset) ====\n');
fprintf('AUC True = %.6f | AUC NB = %.6f | dAUC = %.6f\n', auc_true, auc_nb, auc_nb-auc_true);
fprintf('\nMAP (gamma=P0/P1=%.6f)\n', gamma_map);
fprintf(' True: FPR=%.6f, TPR=%.6f, P_e=%.7f\n', FPR_t(kmap_t), TPR_t(kmap_t), Perr_t(kmap_t));
fprintf(' NB  : FPR=%.6f, TPR=%.6f, P_e=%.7f\n', FPR_n(kmap_n), TPR_n(kmap_n), Perr_n(kmap_n));
fprintf('\nMinimum empirical P_e\n');
fprintf(' True: gamma=%.9f, FPR=%.6f, TPR=%.6f, P_e=%.7f\n', gamma_t(kmin_t), FPR_t(kmin_t), TPR_t(kmin_t), Perr_t(kmin_t));
fprintf(' NB  : gamma=%.9f, FPR=%.6f, TPR=%.6f, P_e=%.7f\n', gamma_n(kmin_n), FPR_n(kmin_n), TPR_n(kmin_n), Perr_n(kmin_n));
fprintf(' P_e(min) difference (NB - True) = %.7f\n', Perr_n(kmin_n) - Perr_t(kmin_t));
fprintf('====================================\n');

% Optional: save figures
% saveas(figure(1), 'nb_only_roc.png');
% saveas(figure(2), 'comparison_true_vs_nb_roc.png');

end

% ---------- helper: empirical ROC from an LLR vector -------------
function [FPR,TPR,Perr,gamma_grid] = roc_from_llr(llr, idx0, idx1, P0, P1)
t_grid = [-Inf; sort(llr); +Inf];
gamma_grid = exp(t_grid);
TPR = zeros(numel(t_grid),1); FPR = TPR;
for k = 1:numel(t_grid)
    D = llr > t_grid(k);
    TPR(k) = mean(D(idx1)==1);
    FPR(k) = mean(D(idx0)==1);
end
Perr = FPR*P0 + (1-TPR)*P1;
end
