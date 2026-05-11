% generate dataset
%   x_i, r_i
%       r_i = b_i - A_i x_{i-1}
addpath("HelperFunctions/");
training_mode = 'camera';
reduced=true;
N                           = 64;

if strcmp(training_mode, 'camera')
    out_name = 'Data_Cameraman2/data.mat';
    TargetImg = imresize(im2double(imread('cameraman.tif')), [N, N]);
    if reduced
        out_name = 'Data_Cameraman_red/data.mat';
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
    out_name = 'Data_SheppLogan2/data.mat';
    TargetImg = phantom(N);
end

% TargetImg = phantom(N);

run("LoadSettings.m");
total_recon_curr = [];
total_recon_next = [];
total_lambda = [];

lambdas = .05:.05:2;

% random initialization
all_lambda_check= {};
for ctr = 1:100
    ProblemSetup.init = rand(size(A,2),1);
    [all_recon_curr,all_recon_next, all_lambda] = Reconstruction_v3(AlgorithmSettings,ProblemSetup, lambdas, 1);

    final_it_count = size(all_recon_next,2);
disp(final_it_count)
    all_recon_curr = reshape(all_recon_curr,N,N,final_it_count);
    all_recon_next = reshape(all_recon_next,N,N,final_it_count);
    %{
residual_freq_power = zeros(size(all_residual));
for ii = 1:size(all_residual,2)
    res = reshape(all_residual(:,ii), [N,N]);
    res_ft_power = fftshift(abs(fft2(res)).^2);
    residual_freq_power(:,ii) = res_ft_power(:);
end

    %}
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

save(out_name, 'total_recon_curr', 'total_recon_next','total_lambda','img','N','-v7.3');

%%
% visualize average lambda for each iteration
it_count = zeros(75,1);
l_total = zeros(75,1);
for ii = 1:length(all_lambda_check)
    for jj = 1:length(all_lambda_check{ii})
        tmp = all_lambda_check{ii};
        it_count(jj) = it_count(jj)+1;
        l_total(jj) = l_total(jj) + tmp(jj);
    end
end
for jj = 1:75
    l_total(jj) = l_total(jj) / it_count(jj);

end
figure; plot(l_total);

