function [A, projections, img] = PRtomo(options) 
    angles  = options.angles; 
    bins    = options.bins; 
    width   = options.width; 
    N       = options.N; 
    
    if isfield(options, "SystemMatrix")
        matrixMode = options.SystemMatrix;
    else
        matrixMode = "ChordLength";
    end
    if isfield(options, "Img")
        img = options.Img;
    else
        img = phantom(N);
    end
    img = img(:);
    A = GenSystemMatrix(N,angles,bins,width);

    if strcmp(matrixMode, "Intercept")
        A = double(A>0);
    end
    
    if isfield(options,"Nonnegative")
        if options.Nonnegative
            A(A<0) = 0;
        end
    end
    projections = A*img;
    
end

function A = GenSystemMatrix(N,theta,bins,width) 
    num_projections = length(theta);

    % The starting values both the x and the y coordinates.
    x0 = linspace(-width/2,width/2,bins)';
    y0 = zeros(bins,1);

    % The intersection lines.
    x = (-N/2:N/2)';
    y = x;

    rows = zeros(2*N*num_projections*bins,1);
    cols = rows;
    vals = rows;
    idxend = 0;

    Proj_Indices = 1:num_projections;
    Bin_Indices = 1:bins;


    % Loop over the chosen angles.
    for Proj_Num = Proj_Indices
        % All the starting points for the current angle.
        x0theta = cosd(theta(Proj_Num))*x0-sind(theta(Proj_Num))*y0;
        y0theta = sind(theta(Proj_Num))*x0+cosd(theta(Proj_Num))*y0;

        % The direction vector for all rays corresponding to the current angle.
        a = -sind(theta(Proj_Num));
        b = cosd(theta(Proj_Num));

        % Loop over the rays.
        for Bin_Num = Bin_Indices

            % Use the parametrisation of line to get the y-coordinates of
            % intersections with x = constant.
            tx = (x - x0theta(Bin_Num))/a;
            yx = b*tx + y0theta(Bin_Num);

            % Use the parametrisation of line to get the x-coordinates of
            % intersections with y = constant.
            ty = (y - y0theta(Bin_Num))/b;
            xy = a*ty + x0theta(Bin_Num);

            % Collect the intersection times and coordinates.
            t = [tx; ty];
            xxy = [x; xy];
            yxy = [yx; y];

            % Sort the coordinates according to intersection time.
            [~,I] = sort(t);
            xxy = xxy(I);
            yxy = yxy(I);

            % Skip the points outside the box.
            I = (xxy >= -N/2 & xxy <= N/2 & yxy >= -N/2 & yxy <= N/2);
            xxy = xxy(I);
            yxy = yxy(I);

            % Skip double points.
            I = (abs(diff(xxy)) <= 1e-10 & abs(diff(yxy)) <= 1e-10);
            xxy(I) = [];
            yxy(I) = [];

            % Calculate the length within cell and determines the number of
            % cells which is hit.
            aval = sqrt(diff(xxy).^2 + diff(yxy).^2);
            col = [];

            % Store the values inside the box.
            if numel(aval) > 0

                % If the ray is on the boundary of the box in the top or to the
                % right the ray does not by definition lie with in a valid cell.
                if ~((b == 0 && abs(y0theta(Bin_Num) - N/2) < 1e-15) || ...
                        (a == 0 && abs(x0theta(Bin_Num) - N/2) < 1e-15)       )

                    % Calculates the midpoints of the line within the cells.
                    xm = 0.5*(xxy(1:end-1)+xxy(2:end)) + N/2;
                    ym = 0.5*(yxy(1:end-1)+yxy(2:end)) + N/2;

                    % Translate the midpoint coordinates to index.
                    col = floor(xm)*N + (N - floor(ym));

                end
            end
            if ~isempty(col)
                
                idxstart = idxend + 1;
                idxend = idxstart + numel(col) - 1;
                idx = idxstart:idxend;

                % Store row numbers, column numbers and values.
                rows(idx) = (Proj_Num-1)*bins + Bin_Num;
                cols(idx) = col;
                vals(idx) = aval;

            end
        end % end j
    end % end i

    rows = rows(1:idxend);
    cols = cols(1:idxend);
    vals = vals(1:idxend);

    % Create sparse matrix A from the stored values.
    A = sparse(rows,cols,vals,bins*num_projections,N^2);
    
end

