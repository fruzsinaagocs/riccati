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
phase = 'exact';
%phase = 'numerical';

% Number of collocation points for the Clenshaw-Curtis algorithm in case
% the phase gets computed numerically.
n_int = 20; % 

% Airy example: a(x) = x
% Paramters
epsi = 1;           % parameter epsilon
Tol = 10^-5;        % master tolerance

% Note that in this case there is a turning point at 0, therefore this
% point must be excluded from the computation intervall
x_start = 0.01;     % Start of the intervall
x_end = 50;         % End of the intervall 


% Airy functions evaluated at startigng point.
phi_0 = airy(0,-x_start) + airy(2,-x_start)*1i;
dphi_0 = - (airy(1,-x_start) + airy(3,-x_start)*1i);

% Initial values of the IVP (cf. subsection 5.1)
phi_init = [phi_0,dphi_0];

% Coefficient function and its derivatives
a = @(x) x;
da = @(x) 1;
dda = @(x) 0;
ddda = @(x) 0;
dddda = @(x) 0;
ddddda = @(x) 0;

% Exact phase (corresponding to Eq. (2.7), only necessary if
% phase = 'exact')
phi_eps = @(x) (2/3)*x.^(3/2)-(5/48)*x.^(-3/2) - ...
    ((2/3)*x_start.^(3/2)-(5/48)*x_start.^(-3/2));

switch phase
    case 'exact'
        phase_info = phi_eps;
    case 'numerical'
        phase_info = n_int;
end

tic 
[PhiSol, xGrid, scheme_flag_vec] = ...
    adaptive_WKB_marching_method(a,da,dda,ddda,dddda,ddddda,x_start,...
    x_end,epsi,phi_init,Tol,phase,phase_info);
time = toc % print the time needed for computing the solution


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

