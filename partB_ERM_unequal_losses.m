% =========================== PART B (ERM) ================================
clear; clc; close all; rng(42);

% ----- Setup (same distribution as Part A; change if you wish) ----------
K = 4; d = 2; N = 10000;
priors = 0.25 * ones(1,K);

mu = [ -2  2;
       -3 -1;
        2 -2;
        3  2];

Sigma(:,:,1) = [1.0  0.3;  0.3 1.0];
Sigma(:,:,2) = [0.8 -0.2; -0.2 1.2];
Sigma(:,:,3) = [1.2  0.0;  0.0 0.5];
Sigma(:,:,4) = [0.6  0.25; 0.25 0.8];

% ----- Loss matrix (costly to misclassify true class 4) ------------------
Lambda = [0 1 1 100;
          1 0 1 100;
          1 1 0 100;
          3 3 3 0];

% ------------------------ Sampling X and labels L ------------------------
u = rand(N,1); edges = [0 cumsum(priors)];
L = zeros(N,1);
for j = 1:K
    L(u > edges(j) & u <= edges(j+1)) = j;
end
X = zeros(N,d);
for j = 1:K
    idx = (L == j);
    nj = sum(idx);
    R = chol(Sigma(:,:,j)); % Sigma = R'R
    X(idx,:) = randn(nj,d)*R + mu(j,:);
end

% -------------------- Posteriors via stable log-sum-exp ------------------
loglike = zeros(N,K);
for j = 1:K
    loglike(:,j) = log_gauss_pdf(X, mu(j,:), Sigma(:,:,j)) + log(priors(j));
end
loglike = loglike - max(loglike,[],2); % stabilize
num = exp(loglike);
post = num ./ sum(num,2); % NxK, rows sum to 1

% -------------------------- ERM decision rule ----------------------------
% Risk R_i(x) = sum_j lambda_{ij} P(L=j|x)
risk = post * Lambda.'; % NxK
[~, D_erm] = min(risk, [], 2);

% ------------------ Confusion (counts) and average risk ------------------
C_counts = confusion_counts(L, D_erm, K); % rows=decision i, cols=true j
colN = sum(C_counts,1);
C_prob = C_counts ./ colN; % P(D=i | L=j)

overall_err = mean(D_erm ~= L);
loss_vec = arrayfun(@(i,l) Lambda(i,l), D_erm, L);
avg_risk = mean(loss_vec);

% ---------------------- Visualization (required style) -------------------
figure('Color','w'); hold on; grid on; box on;
shapes = {'.','o','^','s'}; sz = 18;
for j = 1:K
    idx_true = (L==j);
    idx_correct = idx_true & (D_erm==j);
    idx_incorrect = idx_true & (D_erm~=j);
    if any(idx_correct)
        scatter(X(idx_correct,1), X(idx_correct,2), sz, 'g', 'filled', 'Marker', shapes{j});
    end
    if any(idx_incorrect)
        scatter(X(idx_incorrect,1), X(idx_incorrect,2), sz, 'r', 'filled', 'Marker', shapes{j});
    end
end
title('ERM (unequal-loss) classification: shapes=true class, color=correctness');
xlabel('x_1'); ylabel('x_2'); axis equal;
h1 = scatter(nan,nan,sz,'g','filled','Marker','o','DisplayName','Correct');
h2 = scatter(nan,nan,sz,'r','filled','Marker','o','DisplayName','Incorrect');
legend([h1 h2],'Location','bestoutside'); set(gca,'FontSize',12);
saveas(gcf,'ERM_partB_scatter.png'); % <-- saved for LaTeX

% ----------------------- Export results for LaTeX ------------------------
fid = fopen('partB_results.tex','w');
fprintf(fid,'\\newcommand{\\AvgRisk}{%.4f}\n', avg_risk);
fprintf(fid,'\\newcommand{\\ERMErr}{%.4f}\n', overall_err);
fprintf(fid,'\\newcommand{\\PartBConfusionTable}{%%\n');
fprintf(fid,'\\begin{tabular}{lcccc}\\toprule\n');
fprintf(fid,' & $L{=}1$ & $L{=}2$ & $L{=}3$ & $L{=}4$\\\\\\midrule\n');
for i = 1:K
    fprintf(fid,'$D{=}%d$ & %.3f & %.3f & %.3f & %.3f\\\\\n', ...
        i, C_prob(i,1), C_prob(i,2), C_prob(i,3), C_prob(i,4));
end
fprintf(fid,'\\bottomrule\\end{tabular}}\n');
fclose(fid);

% ============================ helpers ====================================
function ll = log_gauss_pdf(X, mu, Sigma)
[~, d] = size(X);
L = chol(Sigma,'lower');
y = (X - mu) / L';
q = sum(y.^2,2);
logdet = 2*sum(log(diag(L)));
ll = -0.5*(d*log(2*pi) + logdet + q);
end

function C = confusion_counts(L, D, K)
C = zeros(K);
for j = 1:K
    idx = (L==j);
    for i = 1:K
        C(i,j) = sum(D(idx)==i);
    end
end
end
