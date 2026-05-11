% file for all general projection reconstruction methods
% inputs are: A, b, method, iterations (N)
% outputs are N arrays of the solution size, one for each iteration step

addpath("HelperFunctions/", "TrainedModels/");
N = 64;
% TargetImg = imresize(im2double(imread('cameraman.tif')), [N, N]);
TargetImg = phantom(N);

run("LoadSettings.m");

ProblemSetup.Iterations = 500;
    ProblemSetup.init = zeros(N*N,1);
    % ProblemSetup.init = ones(N*N,1);
errors = [];
counts = [];
for kk = 1:100
        ProblemSetup.init = rand(N*N,1);

    [all_lambda_ml, recon_ml, recon_const, recon_rand, count] = Reconstruction_Testing_(AlgorithmSettings,ProblemSetup, 1);

    errors = [errors; [norm(recon_ml-img),norm(recon_const-img),norm(recon_rand-img)]];
    counts = [counts; [count]];
end
figure(4); plot(all_lambda_ml);title('Model Predicted Lambda'); xlabel('Iteration'); ylabel('Lambda')
mean_errors = mean(errors, 1);
mean_counts = mean(counts, 1);

% Standard deviations
std_errors = std(errors, 0, 1);
std_counts = std(counts, 0, 1);

function [all_lambda_ml, recon_ml, recon_const, recon_rand, counts] = Reconstruction_Testing_(AlgorithmSettings,ProblemSetup, fignum)

    model = 'shepp2'; %camera2, shepp2
    AlgorithmSettings.visualize=true;
    [recon_ml, all_lambda_ml, recon_count_ml]   = SART_ML(AlgorithmSettings, ProblemSetup, model, fignum);
    fprintf('ML used %i Iterations\n', recon_count_ml);
    AlgorithmSettings.lambda = 1 + (1.5 - 1) * rand;
    [recon_const, recon_count_const]            = SART_Constant(AlgorithmSettings, ProblemSetup, fignum+1);
    fprintf('Constant (%0.5f) used %i iterations', AlgorithmSettings.lambda, recon_count_const)
    [recon_rand, recon_count_rand]            = SART_Random(AlgorithmSettings, ProblemSetup, fignum+2);
    fprintf('Random used %i iterations', recon_count_rand)

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
    
    lambda_model = py.torch.jit.load(sprintf("TrainedModels/lambda_%s.pt", model));
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
            fprintf('exit criteria met in %i iterations\n', iter);
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
            fprintf('exit criteria met in %i iterations\n', iter);
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
            fprintf('exit criteria met in %i iterations\n', iter);
            recon_count = iter;
            return;
        end
    end
    recon_count = iterations;


    
end

