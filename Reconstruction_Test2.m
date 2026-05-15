% initial testing to see see how model performs on reconstruction of other
%   images
clc; clear;

addpath("HelperFunctions/", "TrainedModels/");
N = 64;
all_models = {'forblid', 'camera', 'shepp'};
all_images = {'forblid', 'camera', 'shepp'};
max_iters = 500;

summary = struct();
for m_idx = 1:length(all_models)
    training_model = all_models{m_idx};
    fprintf('Using %s model\n',training_model);
    figure(m_idx); tiledlayout(1,2); ctr = 1;

    for im_idx = 1:length(all_images)
        test_im = all_images{im_idx};

        if ~strcmp(test_im,training_model)
            fprintf('\t Testing on %s Img recon\n',test_im);
            % set image to nonmodel
            TargetImg = SelectImg(test_im,N);

            run("LoadSettings.m");
            max_iters = 500;
            ProblemSetup.Iterations = 500; 
            lambda_counts = zeros(1, ProblemSetup.Iterations);
            lambda_cumulate = zeros(1,ProblemSetup.Iterations);
            lambda_cumulate_sq = zeros(1,ProblemSetup.Iterations);
            rand_errors = [];
            rand_counts = [];
            for kk = 1:90
                ProblemSetup.init = rand(N*N,1);
            
                [all_lambda_ml, recon_ml, recon_const, recon_rand, count] = Reconstruction_Testing_(AlgorithmSettings,ProblemSetup, training_mode, 1);
            
                rand_errors = [rand_errors; [norm(recon_ml-img),norm(recon_const-img),norm(recon_rand-img)]];
                rand_counts = [rand_counts; [count]];

                for ii = 1:length(all_lambda_ml)
                    lambda_counts(ii) = lambda_counts(ii)+1;
                    lambda_cumulate(ii) = lambda_cumulate(ii) + all_lambda_ml(ii);
                    lambda_cumulate_sq(ii) = lambda_cumulate_sq(ii) + all_lambda_ml(ii)^2;
                end
            end
            
            ones_errors = [];
            ones_counts = [];
            for kk = 1:5
                ProblemSetup.init = ones(N*N,1);
            
                [all_lambda_ml, recon_ml, recon_const, recon_rand, count] = Reconstruction_Testing_(AlgorithmSettings,ProblemSetup, training_mode, 1);
            
                ones_errors = [ones_errors; [norm(recon_ml-img),norm(recon_const-img),norm(recon_rand-img)]];
                ones_counts = [ones_counts; [count]];

                for ii = 1:length(all_lambda_ml)
                    lambda_counts(ii) = lambda_counts(ii)+1;
                    lambda_cumulate(ii) = lambda_cumulate(ii) + all_lambda_ml(ii);
                    lambda_cumulate_sq(ii) = lambda_cumulate_sq(ii) + all_lambda_ml(ii)^2;
                end
            end
    
            zero_errors = [];
            zero_counts = [];
            for kk = 1:5
                ProblemSetup.init = zeros(N*N,1);
            
                [all_lambda_ml, recon_ml, recon_const, recon_rand, count] = Reconstruction_Testing_(AlgorithmSettings,ProblemSetup, training_mode,1);
            
                zero_errors = [zero_errors; [norm(recon_ml-img),norm(recon_const-img),norm(recon_rand-img)]];
                zero_counts = [zero_counts; [count]];

                for ii = 1:length(all_lambda_ml)
                    lambda_counts(ii) = lambda_counts(ii)+1;
                    lambda_cumulate(ii) = lambda_cumulate(ii) + all_lambda_ml(ii);
                    lambda_cumulate_sq(ii) = lambda_cumulate_sq(ii) + all_lambda_ml(ii)^2;
                end
            end
            
            all_errors = [rand_errors; ones_errors; zero_errors];
            all_counts = [rand_counts; ones_counts; zero_counts];

            all_mean_errors  = mean(all_errors,1);
            all_mean_counts  = mean(all_counts,1);
            all_std_errors   = std(all_errors,0,1);
            all_std_counts   = std(all_counts,0,1);
            stats = struct( 'mean_error', all_mean_errors, ...
                            'mean_counts', all_mean_counts, ...
                            'std_error', all_std_errors, ...
                            'std_counts', all_std_counts);
        
            summary.(strcat(training_model,'_',test_im)) = stats;

            % compute mean + std
            l_mean = zeros(max_iters,1);
            l_std  = zeros(max_iters,1);
            
            for jj = 1:max_iters
                l_mean(jj) = lambda_cumulate(jj) / lambda_counts(jj);
            
                var_jj = (lambda_cumulate_sq(jj) / lambda_counts(jj)) - l_mean(jj)^2;
                var_jj = max(var_jj, 0); % numerical safety
            
                l_std(jj) = sqrt(var_jj);
            end

    x = 1:max_iters;
    nexttile(ctr); ctr = ctr + 1;
    
    yyaxis left
    hold on;
    plot(x, l_mean, 'b', 'LineWidth', 2);
    plot(x, l_mean + 2*l_std, 'r--', 'LineWidth', 1.5);
    plot(x, l_mean - 2*l_std, 'r--', 'LineWidth', 1.5);
    
    fill([x fliplr(x)], ...
         [l_mean+2*l_std; flipud(l_mean-2*l_std)], ...
         'r', 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    
    ylabel('Lambda Value');
    
    yyaxis right
    bar(x, lambda_counts, 0.4, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    ylabel('Sample Count');
    
    xlabel('Iteration');
    title(sprintf('Mean, Variability, and Sample Count: %s with %s Model', test_im, training_model));
    
    legend('Mean', '±2 Std Dev', 'Location', 'best');
    
    [minv, mini] = min(lambda_counts);
    if minv==0
        xlim([0,mini])
    end

        end
    end

    ctr = 1;
end

function [all_lambda_ml, recon_ml, recon_const, recon_rand, counts] = Reconstruction_Testing_(AlgorithmSettings,ProblemSetup,model, fignum)

    % model = 'shepp2'; %camera2, shepp2
    AlgorithmSettings.visualize=false;
    [recon_ml, all_lambda_ml, recon_count_ml]   = SART_ML(AlgorithmSettings, ProblemSetup, model, fignum);
    fprintf('ML used %i Iterations\n', recon_count_ml);
    AlgorithmSettings.lambda = 1 + (1.5 - 1) * rand;
    [recon_const, recon_count_const]            = SART_Constant(AlgorithmSettings, ProblemSetup, fignum+1);
    fprintf('Constant (%0.5f) used %i iterations\n', AlgorithmSettings.lambda, recon_count_const)
    [recon_rand, recon_count_rand]            = SART_Random(AlgorithmSettings, ProblemSetup, fignum+2);
    fprintf('Random used %i iterations\n', recon_count_rand)

    counts = [recon_count_ml, recon_count_const, recon_count_rand];
end

function [recon, all_lambda, recon_count] = SART_ML(pm_SART, ProblemSetup, model, fignum)
if ismac
    pyenv("Version", "/opt/anaconda3/envs/matlab_torch/bin/python");

elseif ispc
    pyenv("Version", "C:\Users\clin4\.conda\envs\matlab_python\python.exe"); %matlab_torch

else
    error("Unsupported operating system");
end
    
    np = py.importlib.import_module('numpy');
    
    lambda_model = py.torch.jit.load(sprintf("TrainedModels/lambda_%s_test1.pt", model));
    % General Variables
    A                 = ProblemSetup.A;
    projections       = ProblemSetup.projections;
    N                 = ProblemSetup.N;
    bins              = ProblemSetup.bins;
    num_angles        = length(ProblemSetup.angles);
    iterations        = ProblemSetup.Iterations;
    img               = ProblemSetup.img;
    % SART variables
    visualize         = pm_SART.visualize;
    exit_criteria     = pm_SART.exit_criteria;

    % calculate weighting and preconditioning matrices for blocks
    subA = cell(1,num_angles);
    subP = cell(1,num_angles);
    subV = cell(1,num_angles);
    subW = cell(1,num_angles);

    for ang = 1:num_angles
        tmpA = A((ang-1)*bins+1:ang*bins,:);
        subA{ang} = tmpA;
        subP{ang} = projections((ang-1)*bins+1:ang*bins);
        tmpV = sum(tmpA,1); tmpV = 1./tmpV; tmpV(isinf(tmpV)) = 0; tmpV = sparse(1:length(tmpV), 1:length(tmpV), tmpV, length(tmpV), length(tmpV));
        subV{ang} = tmpV;
        tmpW = sum(tmpA,2); tmpW = 1./tmpW; tmpW(isinf(tmpW)) = 0; tmpW = sparse(1:length(tmpW), 1:length(tmpW), tmpW, length(tmpW), length(tmpW)); 
        subW{ang} = tmpW;
    end
    
    % set initial ordering
    AngleOrder        = randperm(num_angles);
    
    if visualize
        figure(fignum); set(gcf, 'units', 'normalized', 'outerposition', [.25 .25 .5 .5]); 
    end
    
    % initialize
    all_lambda      = zeros(1,iterations);
    
    recon = ProblemSetup.init;

    for iter = 1:iterations
        prev_recon = recon;
        % predict lambda given current guess
        x = single(prev_recon);
        x = reshape(x, [1,1,N,N]);
        x_torch = py.torch.from_numpy(np.array(x));

        l = single(1);
        l = reshape(l, [1,1,1,1]);   % must already be 4D
        
        l_np = np.array(l, pyargs('dtype', np.float32));
        l_torch = py.torch.from_numpy(l_np.reshape(int32([1,1,1,1])));

        lambda = lambda_model(x_torch, x_torch, l_torch);
        lambda = lambda{2}.detach(); lambda = double(np.array(lambda));

        all_lambda(iter) = lambda; 
        for angle_idx = 1:num_angles
            subset = AngleOrder(angle_idx);
            A_sub = subA{subset};
            b_sub = subP{subset};
            W_sub = subW{subset};
            V_sub = subV{subset};
            
            residual = b_sub - A_sub * recon;
            update = (V_sub * A_sub' * W_sub * (residual));

            recon = recon + lambda .* (update);

        end

        % shuffle angle order
        AngleOrder        = randperm(num_angles);

        if visualize
            figure(fignum); 
            imshow(reshape(recon,N,N), [], 'InitialMagnification', 'fit'); 
            title(sprintf("SART Reconstruction Inner Iteration %i",iter)); pause(.000001);
        end
        change = norm(prev_recon - recon)/sum(prev_recon);
% change = norm(recon-img);
        if change < exit_criteria
            % fprintf('exit criteria met in %i iterations\n', iter);
            all_lambda = all_lambda(1:iter);
            recon_count = iter;
            return;
        end
    end
    recon_count = iterations;
    
end

function [recon, recon_count] = SART_Constant(pm_SART, ProblemSetup, fignum)
    % General Variables
    A                 = ProblemSetup.A;
    projections       = ProblemSetup.projections;
    N                 = ProblemSetup.N;
    bins              = ProblemSetup.bins;
    num_angles        = length(ProblemSetup.angles);
    iterations        = ProblemSetup.Iterations;
    img               = ProblemSetup.img;

    % SART variables
    visualize         = pm_SART.visualize;
    exit_criteria     = pm_SART.exit_criteria;
    lambda            = pm_SART.lambda;

    % calculate weighting and preconditioning matrices for blocks
    subA = cell(1,num_angles);
    subP = cell(1,num_angles);
    subV = cell(1,num_angles);
    subW = cell(1,num_angles);

    for ang = 1:num_angles
        tmpA = A((ang-1)*bins+1:ang*bins,:);
        subA{ang} = tmpA;
        subP{ang} = projections((ang-1)*bins+1:ang*bins);
        tmpV = sum(tmpA,1); tmpV = 1./tmpV; tmpV(isinf(tmpV)) = 0; tmpV = sparse(1:length(tmpV), 1:length(tmpV), tmpV, length(tmpV), length(tmpV));
        subV{ang} = tmpV;
        tmpW = sum(tmpA,2); tmpW = 1./tmpW; tmpW(isinf(tmpW)) = 0; tmpW = sparse(1:length(tmpW), 1:length(tmpW), tmpW, length(tmpW), length(tmpW)); 
        subW{ang} = tmpW;
    end
    
    % set initial ordering
    AngleOrder        = randperm(num_angles);
    
    if visualize
        figure(fignum); set(gcf, 'units', 'normalized', 'outerposition', [.25 .25 .5 .5]); 
    end
    
    % initialize   
    recon = ProblemSetup.init;

    for iter = 1:iterations
        prev_recon = recon;

        for angle_idx = 1:num_angles
            subset = AngleOrder(angle_idx);
            A_sub = subA{subset};
            b_sub = subP{subset};
            W_sub = subW{subset};
            V_sub = subV{subset};
            
            residual = b_sub - A_sub * recon;
            update = (V_sub * A_sub' * W_sub * (residual));

            recon = recon + lambda .* (update);

        end

        % shuffle angle order
        AngleOrder        = randperm(num_angles);

        if visualize
            figure(fignum); 
            imshow(reshape(recon,N,N), [], 'InitialMagnification', 'fit'); 
            title(sprintf("SART Reconstruction Inner Iteration %i",iter)); pause(.000001);
        end
        change = norm(prev_recon - recon)/sum(prev_recon);
% change = norm(recon-img);

        if change < exit_criteria
            % fprintf('exit criteria met in %i iterations\n', iter);
            recon_count = iter;
            return;
        end
    end
    recon_count = iterations;


    
end

function [recon, recon_count] = SART_Random(pm_SART, ProblemSetup, fignum)
    % General Variables
    A                 = ProblemSetup.A;
    projections       = ProblemSetup.projections;
    N                 = ProblemSetup.N;
    bins              = ProblemSetup.bins;
    num_angles        = length(ProblemSetup.angles);
    iterations        = ProblemSetup.Iterations;
    img               = ProblemSetup.img;
    % SART variables
    visualize         = pm_SART.visualize;
    exit_criteria     = pm_SART.exit_criteria;

    % calculate weighting and preconditioning matrices for blocks
    subA = cell(1,num_angles);
    subP = cell(1,num_angles);
    subV = cell(1,num_angles);
    subW = cell(1,num_angles);

    for ang = 1:num_angles
        tmpA = A((ang-1)*bins+1:ang*bins,:);
        subA{ang} = tmpA;
        subP{ang} = projections((ang-1)*bins+1:ang*bins);
        tmpV = sum(tmpA,1); tmpV = 1./tmpV; tmpV(isinf(tmpV)) = 0; tmpV = sparse(1:length(tmpV), 1:length(tmpV), tmpV, length(tmpV), length(tmpV));
        subV{ang} = tmpV;
        tmpW = sum(tmpA,2); tmpW = 1./tmpW; tmpW(isinf(tmpW)) = 0; tmpW = sparse(1:length(tmpW), 1:length(tmpW), tmpW, length(tmpW), length(tmpW)); 
        subW{ang} = tmpW;
    end
    
    % set initial ordering
    AngleOrder        = randperm(num_angles);
    
    if visualize
        figure(fignum); set(gcf, 'units', 'normalized', 'outerposition', [.25 .25 .5 .5]); 
    end
    
    recon = ProblemSetup.init;

    for iter = 1:iterations
        lambda = 0.75 + (2 - 0.75) * rand;
        prev_recon = recon;

        for angle_idx = 1:num_angles
            subset = AngleOrder(angle_idx);
            A_sub = subA{subset};
            b_sub = subP{subset};
            W_sub = subW{subset};
            V_sub = subV{subset};
            
            residual = b_sub - A_sub * recon;
            update = (V_sub * A_sub' * W_sub * (residual));

            recon = recon + lambda .* (update);

        end

        % shuffle angle order
        AngleOrder        = randperm(num_angles);

        if visualize
            figure(fignum); 
            imshow(reshape(recon,N,N), [], 'InitialMagnification', 'fit'); 
            title(sprintf("SART Reconstruction Inner Iteration %i",iter)); pause(.000001);
        end
        change = norm(prev_recon - recon)/sum(prev_recon);
% change = norm(recon-img);

        if change < exit_criteria
            % fprintf('exit criteria met in %i iterations\n', iter);
            recon_count = iter;
            return;
        end
    end
    recon_count = iterations;


    
end

function TargetImg = SelectImg(test_im,N)
    if strcmp(test_im, 'camera')
        TargetImg = imresize(im2double(imread('cameraman.tif')), [N, N]);
        [N, M] = size(TargetImg);
        cx = (M+1)/2;
        cy = (N+1)/2;
        [x, y] = meshgrid(1:M, 1:N);
        r = 28;
        mask = (x - cx).^2 + (y - cy).^2 <= r^2;
        TargetImg = TargetImg .* mask;
    elseif strcmp(test_im,'shepp')
        TargetImg = phantom(N);
    elseif strcmp(test_im, 'forblid')
        TargetImg = analytical_phantom(N);
    end
end