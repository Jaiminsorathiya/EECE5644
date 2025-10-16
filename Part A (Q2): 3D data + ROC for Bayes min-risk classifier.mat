%% Part A (Q2): 3D data + ROC for Bayes min-risk classifier
clear; clc; close all; rng(7); % reproducible

% --- Parameters ---
d = 3; N = 10000;
p0 = 0.65; p1 = 0.35;

m0 = [-1/2; -1/2; -1/2];
C0 = [ 1 -0.5 0.3;
      -0.5 1 -0.5;
       0.3 -0.5 1 ];

m1 = [1; 1; 1];
C1 = [ 1 0.3 -0.2;
       0.3 1 0.3;
      -0.2 0.3 1 ];

% --- Labels and samples ---
u = rand(N,1);
L = zeros(N,1); % 0/1 labels
L(u >= p0) = 1; % P(L=1) = p1

N0 = sum(L==0); N1 = sum(L==1);
X0 = mvnrnd(m0', C0, N0);
X1 = mvnrnd(m1', C1, N1);

X = zeros(N,d);
X(L==0,:) = X0;
X(L==1,:) = X1;

% --- Log-likelihoods and log-likelihood ratio ---
invC0 = inv(C0); invC1 = inv(C1);
logdetC0 = log(det(C0)); logdetC1 = log(det(C1));
cst = d*log(2*pi);

XM0 = X - m0'; quad0 = sum((XM0*invC0).*XM0,2);
XM1 = X - m1'; quad1 = sum((XM1*invC1).*XM1,2);

logp0 = -0.5*(quad0 + logdetC0 + cst);
logp1 = -0.5*(quad1 + logdetC1 + cst);
logLR = logp1 - logp0;

% --- Save dataset (.mat) ---
save('gauss3d_10k_samples.mat','X','L','logLR','m0','C0','m1','C1','p0','p1');

% --- Threshold sweep in log-domain ---
logGammas = [-Inf, linspace(min(logLR)-5, max(logLR)+5, 400), Inf];
is1 = (L==1); is0 = ~is1;

TPR = zeros(size(logGammas));
FPR = zeros(size(logGammas));
Pmis = zeros(size(logGammas));
Pfa = zeros(size(logGammas));

for i = 1:numel(logGammas)
    D = (logLR > logGammas(i)); % decide class 1 if logLR > log(gamma)
    TPR(i) = mean(D(is1)); % P(D=1 | L=1; gamma)
    FPR(i) = mean(D(is0)); % P(D=1 | L=0; gamma)
    Pmis(i) = 1 - TPR(i); % P(D=0 | L=1; gamma)
    Pfa(i) = FPR(i);      % P(D=1 | L=0; gamma)
end
gamma = exp(logGammas);

% --- Bayes 0--1 loss operating point ---
gamma_star = p0/p1; % ~1.8571
[~, idx_star] = min(abs(gamma - gamma_star));

% --- Save ROC points ---
T = table(gamma(:), TPR(:), FPR(:), Pmis(:), Pfa(:), 'VariableNames', {'gamma','TPR','FPR','P_miss','P_fa'});
writetable(T,'roc_points.csv');

% --- ROC figure ---
figure('Color','w');
plot(FPR, TPR, 'LineWidth', 1.8); grid on; axis([0 1 0 1]); axis square;
xlabel('False Positive Rate P(D=1 | L=0; \gamma)');
ylabel('True Positive Rate P(D=1 | L=1; \gamma)');
title('ROC Minimum Expected Risk Classifier (Gaussian classes)');
hold on;
plot(FPR(1), TPR(1), 'o', 'MarkerSize', 6, 'DisplayName','\gamma=0 (always 1)');
plot(FPR(end), TPR(end), 'o', 'MarkerSize', 6, 'DisplayName','\gamma=\infty (always 0)');
plot(FPR(idx_star), TPR(idx_star), 'p', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', sprintf('\\gamma^*\\approx%.3f', gamma_star));
legend('Location','SouthEast');
saveas(gcf, 'roc_minrisk.png');

fprintf('Saved: gauss3d_10k_samples.mat, roc_points.csv, roc_minrisk.png\n');
fprintf('Bayes point ~ FPR=%.4f, TPR=%.4f at gamma*=%.4f\n', FPR(idx_star), TPR(idx_star), gamma_star);

% --- OPTIONAL: 3D dotted scatter visualization ---
try
    idx0 = (L==0); idx1 = ~idx0;
    max_plot = min(N,10000);
    I = randperm(N, max_plot);
    Xp = X(I,:); j0 = idx0(I); j1 = idx1(I);

    figure('Color','w'); set(gcf,'Renderer','opengl');
    scatter3(Xp(j0,1), Xp(j0,2), Xp(j0,3), 6, [0 0.45 1], '.'); hold on;
    scatter3(Xp(j1,1), Xp(j1,2), Xp(j1,3), 6, [1 0 0], '.');
    grid on; axis equal vis3d;
    xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
    title('3D dotted scatter of generated samples');
    legend({'Class 0','Class 1'},'Location','best');
    view(135,28);
    saveas(gcf, 'scatter3_10k.png');
catch
    warning('3D scatter failed to render; continue without it.');
end
