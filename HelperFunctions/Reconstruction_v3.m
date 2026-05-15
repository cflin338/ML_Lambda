% file for all general projection reconstruction methods
% inputs are: A, b, method, iterations (N)
% outputs are N arrays of the solution size, one for each iteration step
function [all_recon,all_residual, all_lambda] = Reconstruction_v3(AlgorithmSettings,ProblemSetup, lambdas, fignum)
    AlgorithmSettings.visualize=false;
    [all_recon, all_residual, all_lambda] = SART(AlgorithmSettings, ProblemSetup, lambdas, fignum);
    
end


function [all_recon_curr, all_recon_next, all_optimal_lambda] = SART(pm_SART, ProblemSetup, lambdas, fignum)
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
    all_recon_curr      = zeros(size(A,2), iterations);
    all_recon_next      = zeros(size(A,2), iterations);
    all_optimal_lambda  = zeros(1,iterations);

    counter = 1;

    recon = ProblemSetup.init;
    for iter = 1:iterations
        prev_recon = recon;
        best_recon = prev_recon;
        best_lambda = 5;
        best_lambda_error = inf;
        for lambda = lambdas
            % for this current reconstruction, do reconstruction with each lambda
            % revert back to original recon
            recon = prev_recon;
            
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
        
            lambda_error = norm(img - recon);
            if lambda_error < best_lambda_error
                % update to select lambda that produces lowest error
                best_lambda_error = lambda_error;
                best_lambda = lambda;
                best_recon = recon;
            end
        
        end
        fprintf('Best lambda for iteration %i: %0.4f\n',iter, best_lambda);
        % update recon to be that with the best update
        recon = best_recon;
        
        all_recon_curr(:,counter) = prev_recon;
        all_recon_next(:,counter) = recon;
        all_optimal_lambda(counter) = best_lambda;
        counter = counter + 1;

        % shuffle angle order
        AngleOrder        = randperm(num_angles);

        if visualize
            figure(fignum); 
            imshow(reshape(recon,N,N), [], 'InitialMagnification', 'fit'); 
            title(sprintf("SART Reconstruction Inner Iteration %i",iter)); pause(.000001);
        end
        change = norm(prev_recon - recon)/sum(prev_recon);
% disp(change)
        if change < exit_criteria
            fprintf('exit criteria met in %i iterations\n', iter);
            all_recon_curr = all_recon_curr(:,1:counter-1);
            all_recon_next = all_recon_next(:,1:counter-1);
            all_optimal_lambda = all_optimal_lambda(1:counter-1);
            return;
        end
    end

    
end