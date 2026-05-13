function img = analytical_phantom(N)
    img = forbild_head_phantom(512, .05);
    img = imresize(img, [N,N],'lanczos2'); 
end
function phantom = forbild_head_phantom(N, voxel_size)
% FORBILD_HEAD_PHANTOM  Generate the FORBILD head phantom (z=0 axial slice).
%
%   phantom = forbild_head_phantom(N, voxel_size)
%
%   Inputs
%   ------
%   N          : image size in pixels (N x N).  Default: 512
%   voxel_size : pixel side length in centimetres.  Default: 0.05 cm
%
%                The phantom extends to ~13 cm from isocentre.
%                Required: (N-1)/2 * voxel_size >= 13 cm
%                  N=512 → voxel_size >= 0.051   (default 0.05 is fine)
%                  N=256 → voxel_size >= 0.102
%                  N=128 → voxel_size >= 0.205
%
%   Output
%   ------
%   phantom : N x N single array of relative densities (rho).
%             Display convention (matches reference FORBILD image):
%               row 1  = top    = +y = anterior  (forehead / eyes)
%               row N  = bottom = -y = posterior  (back of head / chin)
%               col 1  = left   = +x = patient left
%               col N  = right  = -x = patient right
%
%   Usage
%   -----
%   ph = forbild_head_phantom(512, 0.05);
%   figure; imagesc(ph, [0.9 1.1]); axis image; colormap gray; colorbar;
%
%   IMPORTANT – parameter convention
%   ---------------------------------
%   In the FORBILD spec, dx/dy/dz are SEMI-AXES (half-widths), NOT full
%   diameters.  This is the most common source of implementation errors.
%
%   Reference: Lauritsch & Haerer (1998), SPIE Med. Imaging 3336:561-573.

if nargin < 1 || isempty(N),          N          = 512;  end
if nargin < 2 || isempty(voxel_size), voxel_size = 0.05; end
N = N(1);

% ── FOV guard ─────────────────────────────────────────────────────────────
half_fov = (N-1)/2 * voxel_size;
if half_fov < 13
    warning('forbild_head_phantom:smallFOV', ...
        'FOV half-width %.2f cm < 13 cm. Use voxel_size >= %.4f for N=%d.', ...
        half_fov, 13/((N-1)/2), N);
end

% ── coordinate grids ──────────────────────────────────────────────────────
% ax: columns, +x = patient left (radiological convention)
% ay: rows,    +y = top of image (anterior / forehead)
ax = linspace( half_fov, -half_fov, N);   % col 1 = +x (left)
ay = linspace( half_fov, -half_fov, N);   % row 1 = +y (top/anterior)

[XX, YY] = meshgrid(ax, ay);
ZZ = zeros(N, N, 'double');
phantom = zeros(N, N, 'double');

% =========================================================================
%  PRIMITIVE HELPERS
%  dx/dy/dz from the spec are used DIRECTLY as semi-axes (no /2 needed).
% =========================================================================

    % Axis-aligned ellipsoid. a,b,c are semi-axes.
    function mask = E(x0,y0,z0, a,b,c, clip)
        mask = ((XX-x0)/a).^2 + ((YY-y0)/b).^2 + ((ZZ-z0)/c).^2 <= 1;
        if nargin == 7, mask = mask & clip; end
    end

    % Arbitrarily oriented ellipsoid.
    % a_vec, b_vec : orthogonal unit vectors; c-axis = cross(a_vec,b_vec).
    % sa, sb, sc   : semi-axes along a_vec, b_vec, c_vec.
    function mask = EF(x0,y0,z0, sa,sb,sc, a_vec, b_vec)
        c_vec = cross(a_vec, b_vec);
        dX = XX-x0; dY = YY-y0; dZ = ZZ-z0;
        pa = dX*a_vec(1) + dY*a_vec(2) + dZ*a_vec(3);
        pb = dX*b_vec(1) + dY*b_vec(2) + dZ*b_vec(3);
        pc = dX*c_vec(1) + dY*c_vec(2) + dZ*c_vec(3);
        mask = (pa/sa).^2 + (pb/sb).^2 + (pc/sc).^2 <= 1;
    end

    % Elliptic cylinder.
    % axis_v : unit vector along the long axis.
    % ax_v   : unit vector along the sa semi-axis of the cross-section.
    %          sb-axis = cross(axis_v, ax_v).
    % sa, sb : cross-section semi-axes.
    % hl     : half-length along axis_v.
    function mask = EC(x0,y0,z0, sa,sb, hl, axis_v, ax_v)
        sb_v = cross(axis_v, ax_v);
        dX = XX(:)-x0; dY = YY(:)-y0; dZ = ZZ(:)-z0;
        pa = dX*axis_v(1) + dY*axis_v(2) + dZ*axis_v(3);
        px = dX*ax_v(1)   + dY*ax_v(2)   + dZ*ax_v(3);
        py = dX*sb_v(1)   + dY*sb_v(2)   + dZ*sb_v(3);
        mask = reshape((px/sa).^2 + (py/sb).^2 <= 1 & abs(pa) <= hl, N, N);
    end

    % Cone along +y starting at y=y0, axis offset in z by z0.
    % r1 = radius at y0, r2 = radius at y0+l, l = full length.
    function mask = CY(y0, z0, r1, r2, l)
        ry   = r1 + (r2-r1) .* (YY - y0) / l;
        dist = sqrt(XX.^2 + (ZZ-z0).^2);
        mask = dist <= ry & YY >= y0 & YY <= (y0+l);
    end

% =========================================================================
%  SHAPE PAINTING  (painter's order; later shapes overwrite earlier ones)
%
%  dx/dy/dz values from spec are used as-is (they are already semi-axes).
% =========================================================================

% Shape 5 – outer skull (rho=1.800)
phantom( E(0,0,0, 9.6,12.0,12.5) ) = 1.800;

% Shape 6 – brain tissue (rho=1.050)
phantom( E(0,0,0, 9.0,11.4,11.9) ) = 1.050;

% Unnamed – right-hemisphere extension (x < 9.11, rho=1.050)
phantom( E(9.1,0,0, 4.2,1.8,1.8, XX<9.11) ) = 1.050;

% Low-density cones at jaw/chin  (rho=0.3, union group)
phantom( CY(-11.15,-0.2, 0.5,0.2,1.5) ) = 0.3;
phantom( CY(-11.15, 0.2, 0.5,0.2,1.5) ) = 0.3;

% Shape 12 – cerebellum  (rho=1.045)
phantom( E(0,-3.6,0, 1.8,3.6,3.6) ) = 1.045;

% Shape 7 – CSF / ventricle void  (rho=0)
phantom( E(0,8.4,0, 1.8,3.0,3.0) ) = 0;

% Shape 1 – left eye  (rho=1.060)
phantom( E(-4.7,4.3,0.872, 2.0,2.0,2.0) ) = 1.060;

% Shape 2 – right eye  (rho=1.060)
phantom( E( 4.7,4.3,0.872, 2.0,2.0,2.0) ) = 1.060;

% Shape 3 – left ear detail  (rho=1.0525)
phantom( E(-1.08,-9,0, 0.4,0.4,0.4) ) = 1.0525;

% Shape 4 – right ear detail  (rho=1.0475)
phantom( E( 1.08,-9,0, 0.4,0.4,0.4) ) = 1.0475;

% Shape 14 – left optic nerve, elliptic cylinder  (rho=1.800)
%   axis(0,-0.866025,+0.500000)  a_y(0,+0.500000,+0.866025)  l=0.482963 (half-length)
axis14 = [0, -0.866025, 0.500000];
ay14   = [0,  0.500000, 0.866025];
axs14  = cross(ay14, axis14);  axs14 = axs14/norm(axs14);
phantom( EC(0,3.6,0, 1.2,4.0, 0.482963, axis14,axs14) ) = 1.800;

% Shape 15 – optic chiasm, elliptic cylinder  (rho=1.800)
%   axis(1,0,0)  a_y(0,-0.500000,+0.866025)  l=0.4 (half-length)
axis15 = [1, 0, 0];
ay15   = [0, -0.500000, 0.866025];
axs15  = cross(ay15, axis15);  axs15 = axs15/norm(axs15);
phantom( EC(0,9.6,0, 0.525561,2.0, 0.4, axis15,axs15) ) = 1.800;

% Shape 8 – left internal capsule, free ellipsoid  (rho=1.800)
%   a_z(+0.258819,0,+0.965926)  a_x(-0.482963,+0.866025,+0.129410)
az8 = [+0.258819, 0, +0.965926];
ax8 = [-0.482963, +0.866025, +0.129410];
phantom( EF(-1.9,5.4,0, 1.165373,0.405956,3.0, ax8,az8) ) = 1.800;

% Shape 9 – right internal capsule, free ellipsoid  (rho=1.800)
az9 = [-0.258819, 0, +0.965926];
ax9 = [+0.482963, +0.866025, +0.129410];
phantom( EF(1.9,5.4,0, 1.159111,0.405689,3.0, ax9,az9) ) = 1.800;

% Shape 10 – left sylvian fissure, elliptic cylinder  (rho=1.800)
%   axis(0,0,1)  a_x(-0.866025,+0.500000,0)  l=4.0 → half=2.0
axis10 = [0,0,1];
axs10  = [-0.866025, +0.500000, 0];
phantom( EC(-4.3,6.8,-1.0, 1.8,0.24, 2.0, axis10,axs10) ) = 1.800;

% Shape 11 – right sylvian fissure, elliptic cylinder  (rho=1.800)
axis11 = [0,0,1];
axs11  = [+0.866025, +0.500000, 0];
phantom( EC(4.3,6.8,-1.0, 1.8,0.24, 2.0, axis11,axs11) ) = 1.800;

% Shape 13 – petrous bone / MTF test object, free ellipsoid  (rho=1.055)
ax13 = [+0.528438, +0.848972, 0];
az13 = [0, 0, 1];
phantom( EF(6.393945,-6.393945,0, 1.2,0.42,1.2, ax13,az13) ) = 1.055;

% Shapes 16+17 – dense bone cones  (rho=1.800, overwrite low-density ones)
phantom( CY(-11.15,-0.2, 0.5,0.2,1.5) ) = 1.800;
phantom( CY(-11.15, 0.2, 0.5,0.2,1.5) ) = 1.800;

% Unnamed – right hemisphere bone cap  (x < 9.1, rho=1.800)
phantom( E(9.1,0,0, 4.2,1.8,1.8, XX<9.1) ) = 1.800;

% =========================================================================
%  RESOLUTION TEST STRUCTURES
% =========================================================================

% Zero-density test spheres (left side of image, x = +5.6 to +8.8 cm)
r = 0.15;
pts = test_sphere_grid();
for k = 1:size(pts,1)
    phantom( E(pts(k,1),pts(k,2),pts(k,3), r,r,r) ) = 0;
end

% Line-pair resolution gauges (at x = -7, -6.92, -6.84, -6.76 cm)
gd = gauge_data();
for k = 1:size(gd,1)
    phantom( E(gd(k,1),gd(k,2),gd(k,3), gd(k,4),gd(k,5),gd(k,6)) ) = 1.8;
end

end % ═══════════════════════════════════════════════════════════════════════


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function pts = test_sphere_grid()
% Hexagonal grid of test sphere centres at z=0.
% All x-values are POSITIVE (patient left = image left).

rows = { ...
%    y          x-range
      0,         5.6:0.4:8.8; ...
  0.346410,      5.8:0.4:8.6; ...
  0.692820,      6.0:0.4:8.8; ...
  1.039231,      6.6:0.4:8.6; ...
 -0.346410,      5.8:0.4:8.6; ...
 -0.692820,      6.0:0.4:8.8; ...
 -1.039231,      6.6:0.4:8.6; ...
};

pts = zeros(0,3);
for k = 1:size(rows,1)
    y  = rows{k,1};
    xv = rows{k,2};
    pts = [pts; xv(:), repmat(y, numel(xv),1), zeros(numel(xv),1)]; %#ok<AGROW>
end
end

% -------------------------------------------------------------------------
function gd = gauge_data()
% Returns Mx6: [x0,y0,z0, sa,sb,sc] semi-axes for resolution gauge bars.
% Four x-planes; each has 4 bar-width groups; each group has
% 5 y-positions at z=0 and 4 z-positions at y=y_start.

x_planes = [-7.0,     -6.92,    -6.84,    -6.76   ];
dxy      = [ 0.017850, 0.015600, 0.013900, 0.012500];  % x,y semi-axes
dy_step  = [ 0.07140,  0.06240,  0.05560,  0.05000 ];  % y step between bars

% 4 bar-width groups per plane
dz_semi  = [0.200,  0.100,  0.050,  0.025];
y_start  = [-1.0,  -0.520, -0.040,  0.440];
z_step   = [ 0.800,  0.400,  0.200,  0.100];

gd = zeros(0,6);
for ip = 1:4
    for ig = 1:4
        y0 = y_start(ig);
        for j = 0:4                   % 5 bars along y (z=0)
            gd(end+1,:) = [x_planes(ip), y0+j*dy_step(ip), 0, ...
                           dxy(ip), dxy(ip), dz_semi(ig)]; %#ok<AGROW>
        end
        for j = 1:4                   % 4 bars along z (y=y0)
            gd(end+1,:) = [x_planes(ip), y0, j*z_step(ig), ...
                           dxy(ip), dxy(ip), dz_semi(ig)]; %#ok<AGROW>
        end
    end
end
end