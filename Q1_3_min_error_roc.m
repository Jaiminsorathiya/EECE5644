%% Q1(3): Minimum-Error Threshold, ROC, and Comparison to Theory
clear; clc; close all; rng(42);

% ---- Priors and Gaussian parameters ----
p0 = 0.65; p1 = 0.35;
m0 = [-0.5; -0.5; -0.5];
C0 = [1 -0.5 0.3; -0.5 1 -0.3; 0.3 -0.3 1];
m1 = [1;1;1];
C1 = [1 0.3 -0.2; 0.3 1 0.3; -0.2 0.3 1];
N = 10000;

% ---- Generate samples ----
u = rand(1, N);
idx0 = find(u <= p0); idx1 = find(u > p0);
N0 = numel(idx0); N1 = numel(idx1);
x = zeros(3, N);
x(:, idx0) = mvnrnd(m0', C0, N0)';
x(:, idx1) = mvnrnd(m1', C1, N1)';
L = zeros(1, N); L(idx1) = 1;

% ---- Log-likelihood ratio ----
logp0 = logmvnpdf_cols(x, m0, C0);
logp1 = logmvnpdf_cols(x, m1, C1);
llr = logp1 - logp0;

% ---- Sweep thresholds ----
gammas = logspace(-4, 4, 2001);
lgammas = log(gammas);

Pfp = zeros(size(gammas)); Ptp = zeros(size(gammas)); Perror = zeros(size(gammas));
is0 = (L==0); is1 = (L==1);
n0 = sum(is0); n1 = sum(is1);

for i = 1:numel(gammas)
    decisions = llr > lgammas(i);
    Pfp(i) = sum(decisions & is0)/n0;
    Ptp(i) = sum(decisions & is1)/n1;
    Perror(i) = Pfp(i)*p0 + (1 - Ptp(i))*p1;
end

% ---- Minimum empirical P(error) ----
[minErr, idxMin] = min(Perror);
gamma_emp = gammas(idxMin);

% ---- Theoretical threshold ----
gamma_theory = p0 / p1;
[~, idxTh] = min(abs(gammas - gamma_theory));

% ---- Plot ROC ----
figure('Color','w');
plot(Pfp, Ptp, 'LineWidth', 1.5); hold on; grid on;
plot(Pfp(idxMin), Ptp(idxMin), 'ro', 'MarkerFaceColor','r');
plot(Pfp(idxTh), Ptp(idxTh), 'ys', 'MarkerFaceColor','y');
xlabel('False Positive Rate P(D=1 | L=0)');
ylabel('True Positive Rate P(D=1 | L=1)');
title('ROC with Minimum-Error and Theoretical Operating Points');
legend('ROC', 'Min P(error)', 'Theoretical \gamma', 'Location', 'SouthEast');

% ---- Helper function ----
function logp = logmvnpdf_cols(X, m, C)
    [d, N] = size(X);
    R = chol(C);
    logdetC = 2*sum(log(diag(R)));
    CiX = C \ (X - m);
    qf = sum((X - m) .* CiX, 1);
    logp = -0.5*(qf + d*log(2*pi) + logdetC);
end
