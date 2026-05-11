% generate dataset
%   x_i, r_i
%       r_i = b_i - A_i x_{i-1}
addpath("HelperFunctions/");
run("LoadSettings.m");
total_recon_curr = [];
total_recon_next = [];
total_lambda = [];

lambdas = .05:.05:2;

% random initialization
for ctr = 1:30
    ProblemSetup.init = rand(size(A,2),1);
    [all_recon_curr,all_recon_next, all_lambda] = Reconstruction_v2(AlgorithmSettings,ProblemSetup, lambdas, 1);

    final_it_count = size(all_recon_next,2);
disp(final_it_count)
    all_recon_curr = reshape(all_recon_curr,N,N,final_it_count);
    all_recon_next = reshape(all_recon_next,N,N,final_it_count);

    total_recon_curr = cat(3, total_recon_curr, all_recon_curr);
    total_recon_next = cat(3, total_recon_next, all_recon_next);
    total_lambda = [ total_lambda , all_lambda];

end
% zero initialization
ProblemSetup.init = zeros(size(A,2),1);
[all_recon_curr,all_recon_next, all_lambda] = Reconstruction_v2(AlgorithmSettings,ProblemSetup, lambdas, 1);

final_it_count = size(all_recon_next,2);
all_recon_curr = reshape(all_recon_curr,N,N,final_it_count);
all_recon_next = reshape(all_recon_next,N,N,final_it_count);
total_recon_curr = cat(3, total_recon_curr, all_recon_curr);
total_recon_next = cat(3, total_recon_next, all_recon_next);
total_lambda = [ total_lambda , all_lambda];
disp('verifying equal amounts of each')
disp(size(total_recon_curr))
disp(size(total_recon_next))
disp(size(total_lambda))

img = reshape(img, [N,N]);

save('Data3/data.mat', 'total_recon_curr', 'total_recon_next','total_lambda','img','N','-v7.3');

