% Script to generate all datasets:
%   Data_Forblid_Test1
%   Data_Cameraman_Test1 (reduced)
%   Data_Shepplogan_Test1
%   
%   For each dataset, produce:
%       100 reconstructions with random initialization
%         5 reconstructions with zero initialization
%         5 reconstructions with ones initialization
%   Data points are:
%       Current reconstruction x_i
%       Optimal lambda l
%       Reconstruction using optimal lambda

addpath("HelperFunctions/");
all_modes = {'forb','camera', 'shepp'};

reduced = true;
N       = 64;
lambdas = .05:.05:2;
for m_idx = 1:length(all_modes)
    training_mode = all_modes{m_idx};

    if strcmp(training_mode, 'camera')
        out_name = 'Data_Cameraman_Test1/data.mat';
        TargetImg = imresize(im2double(imread('cameraman.tif')), [N, N]);
        if reduced
            out_name = 'Data_Cameraman_Test1/data.mat';
            % check to see if directory exists
            if ~isfolder('Data_Cameraman_Test1/')
                mkdir('Data_Cameraman_Test1/');
            end
            % assume img is 64x64
            [N, M] = size(TargetImg);
            
            % center (use half-pixel convention for even size)
            cx = (M+1)/2;
            cy = (N+1)/2;
            
            % create coordinate grid
            [x, y] = meshgrid(1:M, 1:N);
            
            % radius
            r = 28;
            
            % circular mask
            mask = (x - cx).^2 + (y - cy).^2 <= r^2;
            
            % apply mask
            TargetImg = TargetImg .* mask;
        end
    elseif strcmp(training_mode,'shepp')
    
        if ~isfolder('Data_Shepplogan_Test1/')
            mkdir('Data_Shepplogan_Test1/');
        end
    
        out_name = 'Data_Shepplogan_Test1/data.mat';
        TargetImg = phantom(N);
    
    elseif strcmp(training_mode, 'forb')
        
        if ~isfolder('Data_Forblid_Test1/')
            mkdir('Data_Forblid_Test1/');
        end
    
        out_name = 'Data_Forblid_Test1/data.mat';
        TargetImg = analytical_phantom(N);
    
    end
        
    run("LoadSettings.m");
    total_recon_curr = [];
    total_recon_next = [];
    total_lambda = [];
    
    % random initialization
    all_lambda_check= {};
    for ctr = 1:100
        ProblemSetup.init = rand(size(A,2),1);
        [all_recon_curr,all_recon_next, all_lambda] = Reconstruction_v3(AlgorithmSettings,ProblemSetup, lambdas, 1);
    
        final_it_count = size(all_recon_next,2);
    disp(final_it_count)
        all_recon_curr = reshape(all_recon_curr,N,N,final_it_count);
        all_recon_next = reshape(all_recon_next,N,N,final_it_count);
        total_recon_curr = cat(3, total_recon_curr, all_recon_curr);
        total_recon_next = cat(3, total_recon_next, all_recon_next);
        total_lambda = [ total_lambda , all_lambda];
        all_lambda_check{ctr} = all_lambda;
    end
    
    % zero initialization
    ProblemSetup.init = zeros(size(A,2),1);
    for ctr = 1:5
        [all_recon_curr,all_recon_next, all_lambda] = Reconstruction_v3(AlgorithmSettings,ProblemSetup, lambdas, 1);
        
        final_it_count = size(all_recon_next,2);
        all_recon_curr = reshape(all_recon_curr,N,N,final_it_count);
        all_recon_next = reshape(all_recon_next,N,N,final_it_count);
        total_recon_curr = cat(3, total_recon_curr, all_recon_curr);
        total_recon_next = cat(3, total_recon_next, all_recon_next);
        total_lambda = [ total_lambda , all_lambda];
        
    end
    
    % one initialization
    ProblemSetup.init = ones(size(A,2),1);
    for ctr = 1:5
        
        [all_recon_curr,all_recon_next, all_lambda] = Reconstruction_v3(AlgorithmSettings,ProblemSetup, lambdas, 1);
        final_it_count = size(all_recon_next,2);
        all_recon_curr = reshape(all_recon_curr,N,N,final_it_count);
        all_recon_next = reshape(all_recon_next,N,N,final_it_count);
        total_recon_curr = cat(3, total_recon_curr, all_recon_curr);
        total_recon_next = cat(3, total_recon_next, all_recon_next);
        total_lambda = [ total_lambda , all_lambda];
    end
    
    disp('verifying equal amounts of each')
    disp(size(total_recon_curr))
    disp(size(total_recon_next))
    disp(size(total_lambda))
    
    img = reshape(img, [N,N]);
    
    % save(out_name, 'total_recon_curr', 'total_recon_next','total_lambda','img','N','-v7.3');
    
    %%
    % visualize average lambda for each iteration
    it_count = zeros(max_iters,1);
    l_total  = zeros(max_iters,1);
    l_sqtotal = zeros(max_iters,1);   % NEW: for second moment

    for ii = 1:length(all_lambda_check)
        tmp = all_lambda_check{ii};
    
        for jj = 1:length(tmp)
            it_count(jj) = it_count(jj) + 1;
            l_total(jj)  = l_total(jj) + tmp(jj);
            l_sqtotal(jj)= l_sqtotal(jj) + tmp(jj)^2;
        end
    end
    
    % compute mean + std
    l_mean = zeros(max_iters,1);
    l_std  = zeros(max_iters,1);
    
    for jj = 1:max_iters
        l_mean(jj) = l_total(jj) / it_count(jj);
    
        var_jj = (l_sqtotal(jj) / it_count(jj)) - l_mean(jj)^2;
        var_jj = max(var_jj, 0); % numerical safety
    
        l_std(jj) = sqrt(var_jj);
    end
    
    x = 1:max_iters;
    
    figure(m_idx + 10);
    
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
    bar(x, it_count, 0.4, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    ylabel('Sample Count');
    
    xlabel('Iteration');
    title(sprintf('Mean, Variability, and Sample Count: %s', training_mode));
    
    legend('Mean', '±2 Std Dev', 'Location', 'best');
    
    [minv, mini] = min(it_count);
    if minv==0
        xlim([0,mini])
    end
end
