% gpuDevice(1);

% Problem Reconstruction Setup
% ordering parameters
seed = 1; rng(seed);                % randomization for determining order
prime = 997;                        % prime for cycling through projections

% image and projection parameters
max_iters                   = 75; 
max_angle                   = 180;
Max_angle                   = max_angle; 
Min_angle                   = 0;
num_projections             = max_angle/2;
Angle_increment             = 1;
Angle_Range                 = Max_angle - Min_angle;
Initial_angle               = Angle_Range/num_projections;
angles                      = Initial_angle:Angle_increment:Max_angle;
exit_criteria               = 1e-5;

% setting up problem parameters
ProblemSetup.N              = N;
ProblemSetup.bins           = N*2;
ProblemSetup.width          = round(sqrt(2)*N)-1;
ProblemSetup.Nonnegative    = true;
ProblemSetup.SystemMatrix   = "ChordLength";
ProblemSetup.prime          = prime;

ProblemSetup.angles         = angles;

ProblemSetup.Img = TargetImg;

[A, projections, img]       = PRtomo(ProblemSetup);

ProblemSetup.A              = A;
ProblemSetup.projections    = projections;
ProblemSetup.img            = img;

ProblemSetup.Iterations     = max_iters;

% reconstruction-specific settings
% BOTH
AlgorithmSettings.visualize      = false;
% SART
AlgorithmSettings.lambda    = 1.5; %fixed: .003
AlgorithmSettings.exit_criteria = exit_criteria;