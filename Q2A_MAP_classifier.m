% =========================================================================
% Question 2 - Part A: Minimum Probability-of-Error (MAP) Classifier
% =========================================================================
clear; clc; close all; rng(42);

K = 4; d = 2; N = 10000;
priors = 0.25 * ones(1,K);

% Means and covariance matrices for four Gaussian classes
mu = [ -2  2;   % class 1
       -3 -1;   % class 2
        2 -2;   % class 3
        3  2];  % class 4

Sigma(:,:,1) = [1.0  0.3;  0.3  1.0];
Sigma(:,:,2) = [0.8 -0.4; -0.4  1.2];
Sigma(:,:,3) = [1.2  0.2;  0.2  0.8];
Sigma(:,:,4) = [0.6  0.0;  0.0  0.7];

% Step 1: Generate labels according to priors
edges = [0 cumsum(priors)];
u = rand(N,1);
L = zeros(N,1);
for i = 1:K
    L(u>edges(i) & u<=edges(i+1)) = i;
end
n_per = arrayfun(@(i) sum(L==i), 1:K);

% Step 2: Generate samples
X = zeros(N,d);
idx = 1;
for i = 1:K
    Xi = mvnrnd(mu(i,:), Sigma(:,:,i), n_per(i));
    X(idx:idx+n_per(i)-1,:) = Xi;
    idx = idx + n_per(i);
end

% Step 3: MAP classification
invS = zeros(d,d,K); logdetS = zeros(1,K);
for i = 1:K
    invS(:,:,i) = inv(Sigma(:,:,i));
    logdetS(i) = log(det(Sigma(:,:,i)));
end
G = zeros(N,K);
for i = 1:K
    diff = X - mu(i,:);
    qf = sum((diff * invS(:,:,i)) .* diff, 2);
    G(:,i) = -0.5*qf - 0.5*logdetS(i) + log(priors(i));
end
[~, D] = max(G, [], 2); % decisions

% Step 4: Confusion matrix and empirical error
C = zeros(K,K);
for j = 1:K
    for i = 1:K
        C(i,j) = sum(D(L==j)==i);
    end
end
P_hat = C ./ n_per;
P_error = mean(D ~= L);

fprintf('\n=== MAP Classifier Results ===\n');
disp('Confusion matrix C(i,j) = # decided i given true j:');
disp(C);
disp('Estimated conditional probabilities P(D=i | L=j):');
disp(P_hat);
fprintf('Empirical Probability of Error = %.4f\n', P_error);

% Step 5: Visualization (filled markers)
markerShapes = {'.','o','^','s'}; % dot, circle, triangle, square
correct = (D == L);

figure; hold on; grid on; box on;
for j = 1:K
    idxj = (L == j);
    % Correct (green)
    plot(X(idxj & correct,1), X(idxj & correct,2), markerShapes{j}, ...
        'MarkerFaceColor',[0 0.7 0], 'MarkerEdgeColor','k', ...
        'MarkerSize',6, 'LineWidth',0.8);
    % Incorrect (red)
    plot(X(idxj & ~correct,1), X(idxj & ~correct,2), markerShapes{j}, ...
        'MarkerFaceColor',[0.9 0 0], 'MarkerEdgeColor','k', ...
        'MarkerSize',6, 'LineWidth',0.8);
end
xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('MAP Classification (Green = Correct, Red = Incorrect)','Interpreter','latex');
legend({'C1 correct','C1 wrong','C2 correct','C2 wrong',...
        'C3 correct','C3 wrong','C4 correct','C4 wrong'},...
        'Location','bestoutside');
axis equal;
set(gca,'FontSize',12);
saveas(gcf,'Q2A_MAP_Scatter.png');
% ------------------------------------------------------------------------
% EXTRA PLOT: Visualization of the 10,000 generated samples by true class
% ------------------------------------------------------------------------
figure; hold on; grid on; box on;
colors = lines(K);
markerShapes = {'.','o','^','s'};

for j = 1:K
    idx = (L == j);
    scatter(X(idx,1), X(idx,2), 20, colors(j,:), markerShapes{j}, 'filled', ...
            'DisplayName', sprintf('Class %d', j));
end

xlabel('$x_1$','Interpreter','latex');
ylabel('$x_2$','Interpreter','latex');
title('Generated 2D Gaussian Samples by Class','Interpreter','latex');
legend('Location','bestoutside');
axis equal;
set(gca,'FontSize',12);
saveas(gcf,'Q2A_GeneratedSamples.png');
