% This programm calls "adaptive_WKB_marching_method.m" in
% order to demonstrate the adaptive WKB-marching method applied to a
% simple example (Airy Equation).
%
% Comments in this script may refer to J. Körner, A. Arnold, K. Döpfner,
% WKB-based scheme with adaptive step size control for the Schrödinger
% equation in the highly oscillatory regime
%
% See also: doc adaptive_WKB_marching_method
function bremer237(l, eps, N, outputf)

% Either give the exact phase as input or compute it numerically. Just
% comment or uncomment.
%phase = 'exact';
phase = 'numerical';

% Lambda parameter
%l = 10^5;

% Number of collocation points for the Clenshaw-Curtis algorithm in case
% the phase gets computed numerically.
n_int = 20; % 

% Airy example: a(x) = x
% Paramters
epsi = 1/l;           % parameter epsilon
Tol = eps;        % master tolerance

% Note that in this case there is a turning point at 0, therefore this
% point must be excluded from the computation intervall
x_start = -1;     % Start of the intervall
x_end = 1;         % End of the intervall 

% Airy functions evaluated at startigng point.
phi_0 = 0; 
dphi_0 = l;

% Initial values of the IVP (cf. subsection 5.1)
phi_init = [phi_0,dphi_0];

% Coefficient function and its derivatives, counts
aeval = 0;
da = @(x) -2*x.*cos(3.*x) + 3*x.^2.*sin(3.*x);
dda = @(x) (9*x.^2 - 2).*cos(3*x) + 12*x.*sin(3.*x);
ddda = @(x) (18 - 27*x.^2).*sin(3.*x) + 54*x.*cos(3.*x);
dddda = @(x) (108 - 81*x.^2).*cos(3.*x) - 216*x.*sin(3.*x);
ddddda = @(x) (-540 + 243*x.^2).*sin(3.*x) - 810*x.*cos(3.*x);

% Exact phase (corresponding to Eq. (2.7), only necessary if
% phase = 'exact')
%phi_eps = @(x) (2/3)*x.^(3/2)-(5/48)*x.^(-3/2) - ...
%    ((2/3)*x_start.^(3/2)-(5/48)*x_start.^(-3/2));

switch phase
    case 'exact'
        phase_info = phi_eps;
    case 'numerical'
        phase_info = n_int;
end

tic;
for i = 1:N
    [PhiSol, xGrid, scheme_flag_vec] = ...
        adaptive_WKB_marching_method(@a,da,dda,ddda,dddda,ddddda,x_start,...
        x_end,epsi,phi_init,Tol,phase,phase_info);
end
time = toc; % print the time needed for computing the solution
dtime = time/N;


% Reference solution at the end of the interval
switch l
    case 1e1
        refsol = 0.2913132934408612;
        errref = 7e-14;
    case 1e2
        refsol = 0.5294889561602804;
        errref = 5e-13;
    case 1e3
        refsol = -0.6028749132401260;
        errref = 3e-12;
    case 1e4
        refsol = -0.4813631690625038;
        errref = 5e-11;
    case 1e5
        refsol = 0.6558931145821987;
        errref = 3e-10;
    case 1e6
        refsol = -0.4829009413372087;
        errref = 5e-9;
    case 1e7
        refsol = -0.6634949630196019;
        errref = 4e-8;
end

relerr = abs((refsol - PhiSol(1,end))/refsol);
nsteps = length(PhiSol(1,:));
nrk = 0;
nwkb = 0;
nf = aeval/N;
alllines = [''];
fileempty = false;

% If file doesn't exist, create it and open for writing  
if exist(outputf)~=2
    % Doesn't exist
    fid = fopen(outputf, 'w');   
    fileempty = true;
% Otherwise, open it for reading
else 
    % Exists
    fid = fopen(outputf, 'r');
end
% Check to see if it's empty
if fseek(fid, 1, 'bof') == -1
    % File is empty
    fileempty = true;
else
    frewind(fid);
    tline = fgets(fid);
    alllines = [tline];
    while ischar(tline)
        tline = fgets(fid);
        if length(strfind(tline, 'end')) == 0 
            alllines = [alllines tline];
        end
    end
end
fclose(fid);
% If file was empty, write header:
fid = fopen(outputf, 'w');
if fileempty == true
    fprintf(fid, "# method, l, eps, relerr, tsolve, n_s_osc_att, n_s_osc_suc, n_s_slo_att, n_s_slo_suc, n_s_tot_att, n_s_tot_suc, n_f, n_LS, n_LU, n_sub, errlessref, params\n");
% Otherwise write contents we've just read in
else
    % Remove mysterious null characters appearing in vim 
    alllines = regexprep(alllines,'[\x0]','');
    fprintf(fid, alllines);
end
% Always write current results and ending line
if relerr <= errref
    errlessref = 'True';
else
    errlessref = 'False';
end
fprintf(fid, 'wkbmarching, %d, %d, %0.3g, %0.3g, , %d, , %d, , %d, %d, , , , %s, %s\n', l, Tol, relerr, dtime, nwkb, nrk, nsteps, nf, errlessref, sprintf('(nint = %d)', n_int));
fclose(fid);

function y = a(x)
    y = 1 - x.^2.*cos(3.*x);
    aeval=aeval+1;
end
end
