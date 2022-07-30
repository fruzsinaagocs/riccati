% This programm calls "adaptive_WKB_marching_method.m" in
% order to demonstrate the adaptive WKB-marching method applied to a
% simple example (Airy Equation).
%
% Comments in this script may refer to J. Körner, A. Arnold, K. Döpfner,
% WKB-based scheme with adaptive step size control for the Schrödinger
% equation in the highly oscillatory regime
%
% See also: doc adaptive_WKB_marching_method

%%%:::%%%
clc; clear;
%%%:::%%%

% Either give the exact phase as input or compute it numerically. Just
% comment or uncomment.
%phase = 'exact';
phase = 'numerical';

% Lambda parameter
l = 10^1;

% Number of collocation points for the Clenshaw-Curtis algorithm in case
% the phase gets computed numerically.
n_int = 20; % 

% Airy example: a(x) = x
% Paramters
epsi = 1/l;           % parameter epsilon
Tol = 10^(-12);        % master tolerance

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
global aeval;
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

N = 10;
tic
for i = 1:N
    [PhiSol, xGrid, scheme_flag_vec] = ...
        adaptive_WKB_marching_method(@a,da,dda,ddda,dddda,ddddda,x_start,...
        x_end,epsi,phi_init,Tol,phase,phase_info);
end
time = toc; % print the time needed for computing the solution
time/N


% Reference solution at the end of the interval
switch l
    case 1e1
        refsol = 0.2913132934408612;
    case 1e2
        refsol = 0.5294889561602804;
    case 1e3
        refsol = -0.6028749132401260;
    case 1e4
        refsol = -0.4813631690625038;
    case 1e5
        refsol = 0.6558931145821987;
    case 1e6
        refsol = -0.4829009413372087;
    case 1e7
        refsol = -0.6634949630196019;
end

relerr = abs((refsol - PhiSol(1,end))/refsol)
steps = length(PhiSol(1,:))
aeval



% Reference solution
RefSol = airy(0,-xGrid) + 1i*airy(2,-xGrid);

% Compute the global relative error (here only for the \varphi, NOT
% \dot{\varphi})
RelError = abs(RefSol - PhiSol(1,:))./abs(RefSol);

% For plot: Find indices based on which method was used 
scheme_flag_vec = scheme_flag_vec(2:end); % delete first entry
wkb_used = find(scheme_flag_vec == 1) + 1; % WKB step indices
rk45_used = find(scheme_flag_vec == 0) + 1; % RKF step indices

% Create a (only for plotting) reference solution on a fine grid
dx = 0.001;
xGrid_fine = x_start:dx:x_end;
[RefSol_fine]= airy(0,-xGrid_fine) + 1i*airy(2,-xGrid_fine);

% Plotting results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Renderer', 'painters', 'Position', [10 10 600 230.76]);

% Visualizaio of the reference solution (only real part)
subplot(1,2,1)
plot(xGrid_fine, real(RefSol_fine), '-', 'color', 'k')
hold on;
plot(xGrid(wkb_used), real(RefSol(wkb_used)), 's','MarkerSize',4,...
    'MarkerEdgeColor',[0, 0, 1],...
    'MarkerFaceColor',[0, 0, 1])
hold on;
plot(xGrid(rk45_used), real(RefSol(rk45_used)),'o','MarkerSize',3,...
    'MarkerEdgeColor','red',...
    'MarkerFaceColor','r')
xlabel('$x$','Interpreter','Latex')
ylabel('$\Re[\varphi(x)]$','Interpreter','Latex')
xlim([x_start x_end])
legend('reference solution', 'WKB step', 'RKF45 step','Interpreter','Latex')

% Global relative error plot
subplot(1,2,2)
semilogy(xGrid, RelError ,'-','color','k');
grid on;
hold on;
plot(xGrid(wkb_used), RelError(wkb_used), 's','MarkerSize',4,...
    'MarkerEdgeColor',[0, 0, 1],...
    'MarkerFaceColor',[0, 0, 1])
hold on;
plot(xGrid(rk45_used), RelError(rk45_used),'o','MarkerSize',3,...
    'MarkerEdgeColor','red',...
    'MarkerFaceColor','r')
xlabel('$x$','Interpreter','Latex')
ylabel('relative error','Interpreter','Latex')
xlim([x_start x_end])
legend('relative error','WKB step','RKF45 step','Interpreter','Latex','Location','southeast')


function y = a(x)
    y = 1 - x.^2.*cos(3.*x);
    global aeval;
    aeval += 1;
end

