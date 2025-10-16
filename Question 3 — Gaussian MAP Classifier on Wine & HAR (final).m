%% Question 3 — Gaussian MAP Classifier on Wine & HAR (final)
% End-to-end: load -> estimate -> classify -> confusion matrices -> PCA plots
clear; clc; close all; rng(123);

% -------------------------------------------------------------------------
% Regularization: C_reg = C + λ I, with λ = alpha * trace(C)/rank(C)
alpha = 0.01;              % try 0.01–0.05 if ill-conditioned
STANDARDIZE = true;        % z-score features globally before modeling
% -------------------------------------------------------------------------

fprintf('\n==================== WINE QUALITY (White) ====================\n');
[wineX, wineY] = load_wine_white();
run_experiment(wineX, wineY, alpha, STANDARDIZE, 'Wine (White)');

fprintf('\n==================== HUMAN ACTIVITY (HAR) ====================\n');
[harX, harY, harNames] = load_har_all();  % class names for pretty legends
run_experiment(harX, harY, alpha, STANDARDIZE, 'HAR (Train+Test Combined)', harNames);

%% ---------------------------- FUNCTIONS ---------------------------- %%
function run_experiment(X, y, alpha, STANDARDIZE, datasetName, classNames)
    if nargin < 6, classNames = []; end
    [N, D] = size(X);

    % (optional) global standardization
    if STANDARDIZE
        mu_all = mean(X,1);
        sig_all = std(X,0,1); sig_all(sig_all==0) = 1;
        X = (X - mu_all) ./ sig_all;
    end

    classes = unique(y(:)');
    K = numel(classes);

    % ----- Estimate priors, means, covariances (with regularization) -----
    priors = zeros(1,K);
    Mu = zeros(K, D);
    SigReg = zeros(D, D, K);
    for k = 1:K
        idx = (y == classes(k));
        priors(k) = sum(idx) / N;
        Xk = X(idx, :);
        Mu(k,:) = mean(Xk, 1);
        Ck = cov(Xk, 1);                 % ML covariance (1/N)
        rk = max(rank(Ck),1);
        lam = alpha * (trace(Ck) / rk);  % per assignment hint
        SigReg(:,:,k) = Ck + lam * eye(D);
    end

    % ----- MAP classification (QDA form) -----
    logPost = zeros(N, K);
    const = -0.5 * D * log(2*pi);
    for k = 1:K
        mu = Mu(k,:);
        S  = SigReg(:,:,k);
        [R,p] = chol(S);
        if p>0
            % escalate jitter until PD
            jitter = 1e-8 * max(trace(S)/size(S,1), 1);
            while p>0
                S = S + jitter*eye(D);
                [R,p] = chol(S);
                jitter = jitter * 10;
                if jitter > 1e6, error('Covariance regularization failed.'); end
            end
        end
        logDet = 2*sum(log(diag(R)));
        iS = R \ (R' \ eye(D));          % inv(S) via Cholesky

        XC = X - mu;                     % [N x D]
        Q = sum((XC*iS).*XC, 2);         % Mahalanobis terms
        logLik = const - 0.5*(logDet + Q);
        logPost(:,k) = log(priors(k)) + logLik;
    end
    [~, idxHat] = max(logPost, [], 2);
    yhat = classes(idxHat);

    % ----- Confusion matrix & error -----
    [C, order] = confusionmat(y, yhat, 'Order', classes);
    errRate = 1 - sum(diag(C)) / N;

    fprintf('%s — Classes: %s\n', datasetName, mat2str(classes));
    fprintf('Samples: %d, Features: %d\n', N, D);
    fprintf('Training (in-sample) error: %.4f (%.2f%%)\n', errRate, 100*errRate);

    if isempty(classNames)
        rowLabels = arrayfun(@(c) sprintf('True%d', c), order, 'UniformOutput', false);
        colLabels = arrayfun(@(c) sprintf('Pred%d', c), order, 'UniformOutput', false);
    else
        rowLabels = strcat("True_", string(classNames(order)));
        colLabels = strcat("Pred_", string(classNames(order)));
    end
    disp('Confusion matrix (rows=true, cols=predicted):');
    disp(array2table(C, 'VariableNames', matlab.lang.makeValidName(colLabels), ...
                        'RowNames',    matlab.lang.makeValidName(rowLabels)));

    % ----- PCA visualizations -----
    plot_pca_scatter(X, y, order, datasetName, classNames);
end

function plot_pca_scatter(X, y, classOrder, datasetName, classNames)
    fprintf('Generating PCA visualizations for %s ...\n', datasetName);

    % PCA via SVD
    Xc = X - mean(X,1);
    [U,S,~] = svd(Xc, 'econ');
    score = U*S;
    PC2 = score(:,1:2);
    PC3 = score(:,1:3);

    classes = classOrder(:)';  K = numel(classes);
    cmap = lines(max(K,7));
    mks = {'o','s','^','d','v','>','<','p','h','x','+'};

    if isempty(classNames)
        legText = arrayfun(@(c) sprintf('Class %d', c), classes, 'UniformOutput', false);
    else
        legText = cellstr(classNames(classes));
    end

    % 2D
    figure('Name',[datasetName ' — PCA 2D'],'Color','w');
    hold on; grid on; box on;
    for k = 1:K
        idx = (y == classes(k));
        scatter(PC2(idx,1), PC2(idx,2), 18, cmap(k,:), mks{1+mod(k-1,numel(mks))}, ...
                'filled', 'DisplayName', legText{k});
    end
    xlabel('PC1'); ylabel('PC2'); title([datasetName ' — PCA (2D)']);
    legend('Location','bestoutside'); axis tight;

    % 3D
    figure('Name',[datasetName ' — PCA 3D'],'Color','w');
    hold on; grid on; box on;
    for k = 1:K
        idx = (y == classes(k));
        scatter3(PC3(idx,1), PC3(idx,2), PC3(idx,3), 16, cmap(k,:), ...
                 mks{1+mod(k-1,numel(mks))}, 'filled', 'DisplayName', legText{k});
    end
    xlabel('PC1'); ylabel('PC2'); zlabel('PC3'); title([datasetName ' — PCA (3D)']);
    legend('Location','bestoutside'); axis tight; view(45,30);
end

%% ------------------------- DATA LOADERS ------------------------------ %%
function [X, y] = load_wine_white()
    % White wine quality (4898 x 12 with 'quality' as label)
    file = 'winequality-white.csv';
    if ~isfile(file)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv';
        fprintf('Downloading Wine dataset ...\n');
        try
            websave(file, url);
        catch
            error('Failed to download winequality-white.csv. Please place it in the working folder.');
        end
    end
    % Preserve original headers; the last column is 'quality'
    T = readtable(file, 'Delimiter',';', 'VariableNamingRule','preserve');
    y = T{:,'quality'};        % robust even if headers preserved
    T.('quality') = [];        % drop label
    X = table2array(T);
    bad = any(~isfinite(X),2) | ~isfinite(y);
    X(bad,:) = []; y(bad) = [];
end

function [X, y, classNames] = load_har_all()
    % UCI HAR train+test combined
    rootDir = 'UCI HAR Dataset';
    if ~isfolder(rootDir)
        zipFile = 'UCI_HAR_Dataset.zip';
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip';
        fprintf('Downloading HAR dataset (≈ 60MB) ...\n');
        try
            websave(zipFile, url); unzip(zipFile);
        catch
            error('Failed to download/unzip HAR dataset. Place folder "UCI HAR Dataset" here.');
        end
    end

    % Robust parsing of activity_labels.txt (ignore blanks, no regex)
    fid = fopen(fullfile(rootDir,'activity_labels.txt'),'r');
    if fid==-1, error('Cannot open activity_labels.txt'); end
    C = textscan(fid, '%d %s', 'Delimiter', {' ','\t'}, ...
                 'MultipleDelimsAsOne', true, 'CollectOutput', false);
    fclose(fid);
    ids = C{1}; names = string(C{2});
    keep = isfinite(ids) & strlength(names)>0;
    ids = ids(keep); names = names(keep);
    [ids, ord] = sort(ids); %#ok<ASGLU>
    classNames = names(ord); % 1..6: WALKING, WALKING_UPSTAIRS, ...

    % Load features and labels
    Xtr = readmatrix(fullfile(rootDir,'train','X_train.txt'));
    ytr = readmatrix(fullfile(rootDir,'train','y_train.txt'));
    Xte = readmatrix(fullfile(rootDir,'test','X_test.txt'));
    yte = readmatrix(fullfile(rootDir,'test','y_test.txt'));

    X = [Xtr; Xte];   y = [ytr; yte];
    bad = any(~isfinite(X),2) | ~isfinite(y);
    X(bad,:) = []; y(bad) = [];
end
