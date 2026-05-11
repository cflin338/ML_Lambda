% generate dataset
%   x_i, r_i
%       r_i = b_i - A_i x_{i-1}
addpath("HelperFunctions/");
run("LoadSettings.m");
total_recon = [];
total_residual = [];
total_lambda = [];
total_freq_power = [];

for ctr = 1:30
    [all_recon,all_residual, all_lambda] = Reconstruction(AlgorithmSettings,ProblemSetup, 1);
    final_it_count = size(all_residual,2);
    
    % residual freq-magnitude
    % in theory, later should have less low-freq 
    residual_freq_power = zeros(size(all_residual));
    for ii = 1:size(all_residual,2)
        res = reshape(all_residual(:,ii), [N,N]);
        res_ft_power = fftshift(abs(fft2(res)).^2);
        residual_freq_power(:,ii) = res_ft_power(:);
    end
    
    all_recon = reshape(all_recon,N,N,final_it_count);
    all_residual = reshape(all_residual,N,N,final_it_count);
    residual_freq_power = reshape(residual_freq_power,N,N,final_it_count);
    
    total_recon = cat(3, total_recon, all_recon);
    total_residual = cat(3, total_residual, all_residual);
    total_lambda = [ total_lambda , all_lambda];
    total_freq_power = cat(3, total_freq_power, residual_freq_power);

    return;
end
disp('verifying equal amounts of each')
disp(size(total_recon))
disp(size(total_residual))
disp(size(total_lambda))
disp(size(total_freq_power))
% visualize for last one
% demonstrate that for later iterations, the magnitude of freq decreases,
%   so updates are only high freq
figure(4); tiledlayout(2,2);
nexttile(1);
res = all_residual(:,:,round(final_it_count/4));
res_ft1 = abs(fft2(res)).^2; tmp = log(fftshift(res_ft1));
imagesc(log(fftshift(res_ft1))); title(round(final_it_count/4)); caxis([min(tmp(:)),max(tmp(:))]);
nexttile(2);
res = all_residual(:,:,round(final_it_count/2));
res_ft2 = abs(fft2(res)).^2; 
imagesc(log(fftshift(res_ft2))); title(round(final_it_count/2)); caxis([min(tmp(:)),max(tmp(:))]);
nexttile(3);
res = all_residual(:,:,round(final_it_count*3/4));
res_ft3 = abs(fft2(res)).^2; 
imagesc(log(fftshift(res_ft3))); title(round(final_it_count*3/4)); caxis([min(tmp(:)),max(tmp(:))]);
nexttile(4);
res = all_residual(:,:,end);
res_ft4 = abs(fft2(res)).^2; 
imagesc(log(fftshift(res_ft4))); title('last'); caxis([min(tmp(:)),max(tmp(:))]);

% visualize these residuals
figure(5); tiledlayout(2,2);
nexttile(1); 
res = all_residual(:,:,round(final_it_count/4)); imshow(res./max(res(:)), []);
nexttile(2); 
res = all_residual(:,:,round(final_it_count*2/4)); imshow(res./max(res(:)), []);
nexttile(3); 
res = all_residual(:,:,round(final_it_count*3/4)); imshow(res./max(res(:)), []);
nexttile(4); 
res = all_residual(:,:,end); imshow(res./max(res(:)), []);
% now, save variables
%   all_recon
%   all_residual
%   residual_freq
img = reshape(img, [N,N]);

save('Data/data.mat', 'total_recon', 'total_residual','total_lambda','total_freq_power', 'img','N','-v7.3');

% residuals for SART are equivalent to data updates from radiographs

% recon_diff = zeros(1,size(all_recon,2)-1);
% residual_diff = zeros(1,size(all_residual,2)-1);
% 
% for ii = 1:size(all_recon,2)-1
%     recon_diff(ii) = (norm(all_recon(:,ii+1) - all_recon(:,ii))) / sum(all_recon(:,ii));
%     residual_diff(ii) = (norm(all_residual(:,ii+1) - all_residual(:,ii))) / sum(all_recon(:,ii));
% end
% 
% figure(2); tiledlayout(1,2);
% nexttile(1); plot(log(recon_diff));
% nexttile(2); plot(log(residual_diff));