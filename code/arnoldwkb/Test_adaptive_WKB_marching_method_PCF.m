% This programm calls "adaptive_WKB_marching_method.m" and "PCFU.m" in
% order to demonstrate the adaptive WKB-marching method applied to a
% simple example (Parabolic Cylinder Function).
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

% PCF example: a(x) = K1*x^2 + K2*x
% Paramters
epsi = 2^-6;        % small parameter epsilon; small epsi may cause the
                    % evaluation with PCFU.m to take much time
Tol = 10^-6;       % relative error tolerance
K1 = -1/2;          % First coefficient in a
K2 = 1;             % Second coefficient in a

% Note that in this case there are turning points at 0 and -K2/K1, therefore
% these points must be excluded from the computation intervall
x_start = 0.05;      % Start of the intervall
x_end = -K2/K1-0.05; % End of the intervall 

% Transformation (cf. subsection 5.2)
nu = -K2^2/(8*epsi*sqrt(-K1^3));
z = @(x) (K2+2*K1*x)./(sqrt(2*epsi)*(-K1^3)^(1/4));

% PCF evaluated at transformed starting point.
[U_nu_0, dU_nu_0] = PCFU(nu,z(x_start));

% Scaling factor (cf. subsection 5.2)
kappa = 2./(U_nu_0-1i*sqrt(eps)*2^(3/4)*dU_nu_0); % depends on x_start

% Initial values of the IVP (cf. subsection 5.2)
phi_init = [kappa*U_nu_0, -kappa*sqrt(2/epsi)*(-K1)^(1/4)*dU_nu_0];

% Coefficient function and its derivatives
a = @(x) K1*x.^2+K2*x;
da = @(x) 2*K1*x+K2;
dda = @(x) 2*K1;
ddda = @(x) 0;
dddda = @(x) 0;
ddddda = @(x) 0;

% Antiderivative of sqrt(a(x)) (only necessary if phase = 'exact')
int_sqrt_a = @(x) (2*K1^(5/2)*x.^3+3*K1^(3/2)*K2*x.^2+sqrt(K1)*K2^2*x...
    -K2^(5/2)*sqrt(x).*sqrt((K1/K2).*x+1).*asinh(sqrt((K1/K2).*x)))./...
    (4*K1^(3/2).*sqrt(x.*(K1.*x+K2)));

% Antiderivative of b(x) (only necessary if phase = 'exact')
int_b =   @(x) (-8*K1^3.*x.^3-12*K2*K1^2.*x.^2+6*K2^2*K1.*x+5*K2^3)./...
    (48*K2^2.*(x.*(K1.*x+K2)).^(3/2));

% Exact phase (corresponding to Eq. (2.7), only necessary if
% phase = 'exact')
phi_eps = @(x) (int_sqrt_a(x)-epsi^2.*int_b(x))-...
    (int_sqrt_a(x_start)-epsi^2.*int_b(x_start));

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
RefSol = kappa*PCFU(nu,z(xGrid));

% Compute the global relative error (here only for the \varphi, NOT
% \dot{\varphi})
RelError = abs(RefSol - PhiSol(1,:))./abs(RefSol);

% For plot: Find indices based on which method was used 
scheme_flag_vec = scheme_flag_vec(2:end); % delete first entry
wkb_used = find(scheme_flag_vec == 1) + 1; % WKB step indices
rk45_used = find(scheme_flag_vec == 0) + 1; % RKF step indices

% Create a (only for plotting) reference solution on a fine grid
dx = 0.01;
xGrid_fine = x_start:dx:x_end;
[RefSol_fine]= kappa*PCFU(nu,z(xGrid_fine));

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
ylim([-4 5])
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
legend('relative error','WKB step','RKF45 step','Interpreter','Latex','Location','southeast')

