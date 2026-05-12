% file for all general projection reconstruction methods
% inputs are: A, b, method, iterations (N)
% outputs are N arrays of the solution size, one for each iteration step
function [all_recon,all_residual, all_lambda] = Reconstruction(AlgorithmSettings,ProblemSetup, fignum)
    AlgorithmSettings.visualize=true;
    [all_recon, all_residual, all_lambda] = SART(AlgorithmSettings, ProblemSetup, fignum);
    
end


function [all_recon, all_residual, all_lambda] = SART(pm_SART, ProblemSetup, fignum)
    % General Variables
    A                 = ProblemSetup.A;
    projections       = ProblemSetup.projections;
    N                 = ProblemSetup.N;
    bins              = ProblemSetup.bins;
    num_angles        = length(ProblemSetup.angles);
    iterations        = ProblemSetup.Iterations;

    % SART variables
    lambda            = pm_SART.SART_lambda;
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
    all_recon = zeros(size(A,2), iterations*num_angles);
    all_residual = zeros(size(A,2), iterations*num_angles);
    all_lambda = zeros(1,iterations*num_angles);

    counter = 1;
    recon = zeros(size(A,2),1);
    for iter = 1:iterations
        prev_recon = recon;
        for angle_idx = 1:num_angles
            prev_sub = recon;
            subset = AngleOrder(angle_idx);
            A_sub = subA{subset};
            b_sub = subP{subset};
            W_sub = subW{subset};
            V_sub = subV{subset};
            
            residual = b_sub - A_sub * recon;
            update = (V_sub * A_sub' * W_sub * (residual));
lambda = 0.0001 + (2 - 0.0001) * rand;
            recon = recon + lambda .* (update);
            recon(recon<0) = 0;
            
            % store recons and residuals
            all_recon(:,counter) = prev_sub;
            all_residual(:,counter) = update;
            all_lambda(counter) = lambda;
            counter = counter + 1;
        end
        
        % shuffle angle order
        AngleOrder        = randperm(num_angles);

        if visualize
            figure(fignum); 
            imshow(reshape(recon,N,N), [], 'InitialMagnification', 'fit'); 
            title(sprintf("SART Reconstruction Inner Iteration %i",iter)); pause(.000001);
        end
        change = norm(prev_recon - recon)./sum(prev_recon);
disp(change)
        if change < exit_criteria
            fprintf('exit criteria met in %i iterations\n', iter);
            all_recon = all_recon(:,1:counter-1);
            all_residual = all_residual(:,1:counter-1);
            all_lambda = all_lambda(1:counter-1);
            return;
        end
    end

    
end