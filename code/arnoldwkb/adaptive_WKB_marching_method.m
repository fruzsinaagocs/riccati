function [PhiSol, xGrid, scheme_flag_vec] = adaptive_WKB_marching_method(a,da,dda,ddda,dddda,ddddda,x_start,x_end,epsi,phi_init,Tol,phase,phase_info)

% Adaptive WKB-marching method with a coupling mechanism to the
% well known Runge-Kutta-Fehlberg 4(5) scheme.
%
% This script is directly based on the article "WKB-based
% scheme with adaptive step size control for the Schrödinger equation in
% the highly oscillatory regime" from Jannis Körner, Anton Arnold and
% Kirian Döpfner. A preprint is available in the archive arXiv:2102.03107.
% Comments may refer to certain Equations from this work.
%
% For this programm there are test files called
% "Test_adaptive_WKB_marching_method_Airy.m" and
% "Test_adaptive_WKB_marching_method_PCF.m" with two simple
% examples.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Inputs:
%   a               The coefficient function a(x) as a function handle.
%   da              The first derivative of a(x) as a function handle.
%   dda             The second derivative of a(x) as a function handle.
%   ddda            The third derivative of a(x) as a function handle.
%   dddda           The fourth derivative of a(x) as a function handle.
%   ddddda          The fifth derivative of a(x) as a function handle.
%   x_start         Start of the interval.
%   x_end           End of the interval.
%   h_start         The initial (trial) step size.
%   epsi            Parameter \varepsilon.
%   phi_init        Initial values of the IVP. Two dimensional array
%                   including the values \varphi(x_{0}) and
%                   \dot{\varphi}(x_{0}).
%   RTol            Relative error tolerance.
%   phase           Flag for using/computing the phase (cf. Eq. (2.7)). It
%                   should either be 'exact' or 'numerical'.
%
%   The last input depends on "phase":
%
%   If phase = 'exact':
%   phi_eps         The phase (Eq. (2.7)) given exactly as a function
%                   handle.
%
%   If phase = 'numerical':
%   n_int           The number of Chebyshev collocations points for the
%                   numerical integration (positive integer).
%
% Outputs:
%   PhiSol          The solution array containing the values of \varphi as
%                   well as \dot{\varphi} at the computed spatial grid
%                   points.
%   xGrid           The computed spatial grid.
%   scheme_flag_vec A vector including only ones and zeros. The entries
%                   hint if either the WKB-marching method or the RKF
%                   scheme was used in the corresponding step. "1" means
%                   WKB was used, "0" means RKF was used.
% Note:
%   If phase = 'numerical' the file "clenshaw_curtis.m" is called by this
%   programm in order to integrate the phase numerically with the
%   Clenshaw-Curtis algorithm (e.g. see 
%   [5]: A. Arnold, C. Klein, B. Ujvari, WKB-method for the 1D Schrödinger
%   equation in the semi-classical limit: enhanced phase treatment,
%   submitted, (2019). arxiv.org/abs/1808.01887
%   [8]: C. W. Clenshaw, A. R. CurtisA method for numerical integration on
%   an automatic computer, Numerische Mathemathik 2, 197-205 (1960))
%                   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Contact:  jannis.koerner@tuwien.ac.at (Jannis Körner)
%           anton.arnold@tuwien.ac.at (Anton Arnold)
%           kirian.doepfner@gmail.com (Kirian Döpfner)
%
% Institute of Analysis and Scientific Computing, Technische Universität
% Wien, Wiedner Hauptstr. 8-10, 1040 Wien, Austria
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


switch phase
    % In this case "phase_info" should be the exact phase as a function handle
    case 'exact'
        phi_eps = phase_info;
    
    % In this case "phase_info" should be the number of collocations points for
    % the Clenshaw-Curtis algorithm
    case 'numerical'
        n_int = phase_info;
    
    otherwise
        error('The input variable "phase" was not assigned correctly')
end


%%%%%%%%%%%%%%%%%%%%%% Define global parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Matrix P and its inverse
P = 1/sqrt(2)*[1i    1
    1    1i ];
P_inv = 1/sqrt(2)*[-1i    1
    1    -1i ];

% Order of the coupled numerical methods (in h)
k_wkb = 2; % order of WKB-marching method
k_rkf = 5; % order of RKF-scheme

% Limitations for the step size growth
gf_min = 0.5; % min growth factor
gf_max = 2.0; % max growth factor

% Initialize the solution array with the given initial values of the IVP
PhiSol = phi_init.';

% Initialize the grid
xGrid(1) = x_start;

% Certain flags used in the algorithm
flag = 0; % condition for breaking the main while loop
scheme_flag_vec = 0; % storing information about which method was chosen
reduce_flag = 0; % flag if step was unsuccesful and the step size decreases

% Initial step size
h = 0.5;

% Tolerances for local error control
eta = 10^-2;
ATol = Tol*eta;
RTol = Tol;

% safety parameter
rho = 0.9; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% MAIN LOOP WITH ADAPTIVE STEPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while 1                                                                   %

    % check if the the step size is so large that the algorithm would jump
    % over the end of the interval
    if(xGrid(end) + h > x_end)
        h = x_end - xGrid(end);
        flag = 1;                                                         %
    end                                                                   %
                                                                          %
    while 1
        
        % Compute the needed phase values, i.e., phi^eps(x) and phi^eps(x+h) 
        % (corresponding to Eq. (2.7) and store them in the 2-dim vector 
        % phi_eps_vec
        switch phase
            
            case 'exact'
                
                phi_eps_vec = [phi_eps(xGrid(end)), phi_eps(xGrid(end)+h)];
                
            case 'numerical'
                dphi_eps = @(y) sqrt(a(y))-epsi^2*b(y); % Integrant of Eq. (2.7)
                % Computing phi^eps(x) (here: phi_eps_vec_1)
                if xGrid(end) == xGrid(1) % at start
                    phi_eps_vec_1 = 0;
                else 
                    if reduce_flag == 0 % previous step was succesful
                    % Since phi^eps(x_{n+1}) = phi^eps(x_{n}+h):
                    phi_eps_vec_1 = phi_eps_vec_2;
                    end
                    % else: previous step was not succesful, but phi^eps(x) can
                    % be used again, no computation necessary, phi_eps_vec_1
                    % stays unchanged.
                end
                
                % Computing phi^eps(x+h) (here: phi_eps_vec_2)
                if reduce_flag == 1 % previous step was not succesful
                    % h_old > h, and we already have phi^eps(x+h_old),
                    % hence, phi^eps(x+h)=phi^eps(x+h_old)-int_{x+h}^{x+h_old}dphi_eps
                    phi_eps_vec_2 = phi_eps_vec_2 - clenshaw_curtis(dphi_eps,xGrid(end)+h,xGrid(end)+h_old,n_int);
                else % previous step was succesful
                    % Since phi^eps(x+h)=phi^eps(x)+int_{x}^{x+h}dphi_eps, we only need
                    % to compute the latter term (phase difference)
                    phi_eps_vec_2 = phi_eps_vec_1 + clenshaw_curtis(dphi_eps,xGrid(end),xGrid(end)+h,n_int);
                end
               
                phi_eps_vec = [phi_eps_vec_1, phi_eps_vec_2];
        end
        
        % call the numerical schemes
        [est_wkb,~,wkb2] = wkb_AdSt(PhiSol(:,end),xGrid(end),h);
        [est_rkf,~,rkf5] = rkf45_scheme(PhiSol(:,end),xGrid(end),h);
                
          
        % avoid dividing by zero
        if(est_wkb == 0)
            est_wkb = 10^(-16);
        end
        if(est_rkf == 0)
            est_rkf = 10^(-16);
        end
            
        % Compute the theta_n for both schemes via Eq. (3.2)
        theta_wkb = max(gf_min, min(gf_max,rho*((ATol + RTol*norm(wkb2,inf))/est_wkb)^(1/(k_wkb))));
        theta_rkf = max(gf_min, min(gf_max,rho*((ATol + RTol*norm(rkf5,inf))/est_rkf)^(1/(k_rkf))));
            
        if((est_wkb < ATol + RTol*norm(wkb2,inf) && est_rkf < ATol + RTol*norm(rkf5,inf)) ... 
                || (est_wkb >= ATol + RTol*norm(wkb2,inf) && est_rkf >= ATol + RTol*norm(rkf5,inf)))
            if(theta_wkb > theta_rkf)
                Phi_switch = wkb2;
                est_switch = est_wkb;
                scheme_flag = 1; % flag 1 for wkb used
                h_next = theta_wkb*h;
            else
                Phi_switch = rkf5;
                est_switch = est_rkf;
                scheme_flag = 0; % flag 0 for rkf45 used
                h_next = theta_rkf*h;
            end
        end
        if((est_wkb < ATol + RTol*norm(wkb2,inf) && est_rkf >= ATol + RTol*norm(rkf5,inf)))
                Phi_switch = wkb2;
                est_switch = est_wkb;
                scheme_flag = 1; % flag 1 for wkb used
                h_next = theta_wkb*h;
        end
        if((est_wkb >= ATol + RTol*norm(wkb2,inf) && est_rkf < ATol + RTol*norm(rkf5,inf)))
                Phi_switch = rkf5;
                est_switch = est_rkf;
                scheme_flag = 0; % flag 0 for rkf45 used
                h_next = theta_rkf*h;
        end
        

        % Check, if tolerance is satisfied
        if(est_switch < ATol + RTol*norm(Phi_switch,inf))
            reduce_flag = 0; % no step size reduction will be performed
            break;
        else

        % Important for numerical phase computation in the next step    
        reduce_flag = 1; % flag for unsuccesful step.
        h_old = h; % store the h-value, which was to large.
            
        % step was not succesful, reduce step size    
        h = h_next;
                       
        flag = 0;
        end 
        
    end
    
    % Step was succesful, update the solution array
    PhiSol = [PhiSol, Phi_switch];
    
    % Grid update
    xGrid = [xGrid, xGrid(end) + h];
    
    % Storing the information about which method was used in this step
    scheme_flag_vec = [scheme_flag_vec, scheme_flag];
     
    % flag == 1 & succesful step => we are done
    if(flag == 1)
        break;
    end

    h = h_next;  % trial step size for next step
    
end

%% :::::::::: MAIN  FUNCTIONS ::::::::::

%%%%%%%%%%%%%%%%%%%%%%%%%% Runge-Kutta-Fehlberg %%%%%%%%%%%%%%%%%%%%%%%%%%%

% RKF45 Scheme function including error estimator
function [err_est,rkf4,rkf5] = rkf45_scheme(y,t,h)
    
    % The Runge-Kutta-Fehlberg scheme can be found, e.g., in:
    % E. Hairer, S. Nørsett, G. Wanner, Solving Ordinary Differential
    % Equations I, Springer, Berlin, 2000
    
    k1 = h*odefun(t,y);
    k2 = h*odefun(t+(1/4)*h,y+(1/4)*k1);
    k3 = h*odefun(t+(3/8)*h,y+(3/32)*k1+(9/32)*k2);
    k4 = h*odefun(t+(12/13)*h,y+(1932/2197)*k1-(7200/2197)*k2...
        +(7296/2197)*k3);
    k5 = h*odefun(t+h,y+(439/216)*k1-8*k2+(3680/513)*k3-(845/4104)*k4);
    k6 = h*odefun(t+(1/2)*h,y-(8/27)*k1+2*k2-(3544/2565)*k3...
        +(1859/4104)*k4-(11/40)*k5);
    
    % 4th-order solution
    rkf4 = y+(25/216)*k1+(1408/2565)*k3+(2197/4104)*k4-(1/5)*k5;
    
    % 5th-order solution
    rkf5 = y+(16/135)*k1+(6656/12825)*k3+(28561/56430)*k4...
        -(9/50)*k5+(2/55)*k6;
    
    % Compute the error estimator
    err_est = norm(rkf5-rkf4,Inf);
    
end
     
% Transform the 2nd order ODE into a system in order to use RKF45
function dydt = odefun(t,y)
    dydt = [y(2); -(1/epsi^2)*a(t)*y(1)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%% WKB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% error estimator (plus first and second order wkb solution of that step)
function [err_est,wkb1,wkb2] = wkb_AdSt(phi,x,h)
        
    % a and its first 5 derivatives evaluated at x and x+h
    af = [a(x), a(x+h)];
    aP = [da(x), da(x+h)];
    a2P = [dda(x), dda(x+h)];
    a3P = [ddda(x), ddda(x+h)];
    a4P = [dddda(x), dddda(x+h)];
    a5P = [ddddda(x), ddddda(x+h)];
    
    % b, b_0, b_1, b_2 and b_3 evaluated at x and x+h
    b_Only = b_only(af,aP,a2P);
    b_0 = b_zero(af,aP,a2P);
    b_1 = b_one(af,aP,a2P,a3P);
    b_2 = b_two(af,aP,a2P,a3P,a4P);
    b_3 = b_three(af,aP,a2P,a3P,a4P,a5P);
    
    % Short notation for the term a(x)^(1/4)
    a014 = af(1)^(1/4);
    
    % Vector notation in variable U corresponding to Eq. (2.8)
    U = [a014*phi(1) ; epsi*(aP(1)*phi(1)/(4*a014^5) + phi(2)/a014)];
    
    % Often used values in matrices
    i_eps_phi_eps1 = exp((1i/epsi).*phi_eps_vec(1));
    i_eps_phi_eps2 = exp((1i/epsi).*phi_eps_vec(2));
    
    % The matrix exponential from Eq. (2.10) ...
    EXP_1 = diag([conj(i_eps_phi_eps1) i_eps_phi_eps1]);
    
    % ... and its inverse
    EXP_2 = diag([i_eps_phi_eps2, conj(i_eps_phi_eps2)]);
    
    % Transformation from U to Z corresponding to Eq. (2.10)
    Z = EXP_1*P*U;
    
    % Short notation for the phase difference
    s_n = phi_eps_vec(2) - phi_eps_vec(1);
    
    % Building matrix A^1_n
    A_1 = zeros(2,2);
    
    % Often used values in matrices
    i2_eps_phi_eps1 = i_eps_phi_eps1^-2;
    i2_eps_phi_eps2 = i_eps_phi_eps2^-2;
    h_1_minus_sn = exp(1i*(-(2/epsi)*s_n)) - 1;
    h_2_minus_sn = h_1_minus_sn - 1i.*(-(2/epsi)*s_n);
    
    % Only offdiagonal entries:
    A_1(1,2) = 1i*epsi^2* (...
       b_0(2) * i2_eps_phi_eps2 - b_0(1) * i2_eps_phi_eps1...
       )...
       + epsi^3 * b_1(2) * (...
       i2_eps_phi_eps1*h_1_minus_sn...
       );

    A_1(2,1) = conj(A_1(1,2)); % abusing hermite property
    
    % Scheme iteration via Eq. (2.13) and back transformation to U
    U = P_inv*EXP_2*(eye(2,2) + A_1)*Z;
        
    % Initialize solution vector
    wkb1 = zeros(2,1);
    
    % Back transformation from U to phi via Eq. (2.8)
    wkb1(1) = U(1).*af(2).^(-1/4);
    wkb1(2) = U(2).*(af(2).^(1/4))/epsi - aP(2).*wkb1(1)./(4*af(2)); 
     
    % Building matrix A^1_mod,n
    A_1mod = zeros(2,2);
    
    % Only offdiagonal entries:
     A_1mod(1,2) = -1i*epsi^2*(...
        b_0(1)*i2_eps_phi_eps1 - b_0(2)*i2_eps_phi_eps2...
        )...
        + epsi^3*(...
        b_1(2)*i2_eps_phi_eps2 - b_1(1)*i2_eps_phi_eps1...
        )...
        + 1i*epsi^4*b_2(2)*(...
        -i2_eps_phi_eps1*h_1_minus_sn...
        )...
        - epsi^5*b_3(2)*(...
        i2_eps_phi_eps1*h_2_minus_sn...
        );

      A_1mod(2,1) = conj(A_1mod(1,2)); % abusing hermite property
    
    % Building matrix A^2_n
     A_2 = - (1i*epsi^3*h/2)*(b_Only(2)*b_0(2) + b_Only(1)*b_0(1))*[ 1 0 ; 0 -1 ] ...
         - epsi^4*b_0(1)*b_0(2) * [ h_1_minus_sn 0 ; 0 conj(h_1_minus_sn) ] ...
         + 1i*epsi^5*b_1(2)*(b_0(1) - b_0(2))*[ h_2_minus_sn 0 ; 0 -conj(h_2_minus_sn) ];
    
    % Scheme iteration via Eq. (2.13) and back transformation to U
    U = P_inv*EXP_2*(eye(2,2) + A_1mod + A_2)*Z;
    
    % Initialize solution vector
    wkb2 = zeros(2,1);
    
    % Back transformation from U to phi via Eq. (2.8)
    wkb2(1) = U(1).*af(2).^(-1/4);
    wkb2(2) = U(2).*(af(2).^(1/4))/epsi - aP(2).*wkb2(1)./(4*af(2));
    
    % compute error estimator
    err_est = norm(wkb2 - wkb1,Inf);
    
end

%%%%%% Auxilliary functions %%%%%%%%%

% Function b(x) as function handle needed for computing the phase if
% Input phase = 'numerical'
function b = b(x)
    
    b = (-5/32)*a(x).^(-5/2).*(da(x)).^2+(1/8)*a(x).^(-3/2).*dda(x);
    
end

% Functions b, b_0, b_1, b_2 and b_3 for evaluating at x and x+h
function b_only = b_only(af,aP,a2P)
    
    [b_only] = a2P./(8.*af.^(3/2)) - (5.*aP.^2)./(32.*af.^(5/2));
    
end

function b_zero = b_zero(af,aP,a2P)
    
    [b_zero] = 0.5.*(4.*af.*a2P - 5.*aP.^2)./(32.*af.^3 - 4.*epsi^2.*af.*a2P ...
        + 5.*epsi^2.*aP.^2);
    
end

function b_one = b_one(af,aP,a2P,a3P)
    
    [b_one] = 256.*af.^(9/2).*(4.*af.^2.*a3P - 18.*af.*aP.*a2P + 15.*aP.^3)./((32.*af.^3 ...
        - 4.*epsi^2.*af.*a2P + 5.*epsi^2.*aP.^2).^3);
    
end

function b_two = b_two(af,aP,a2P,a3P,a4P)
    
    [b_two] = 2048.*af.^6.*(675.*epsi^2.*aP.^6 - 1620.*epsi^2.*af.*aP.^4.*a2P - 128.*af.^5.*(9.*a2P.^2 ...
        + 14.*aP.*a3P) + 20.*epsi^2.*af.^2.*aP.^2.*(45.*a2P.^2 + 22.*aP.*a3P) + 256.*af.^6.*a4P ...
        - 8.*af.^3.*(540.*aP.^4 - 18.*epsi^2.*a2P.^3 + 80.*epsi^2.*aP.*a2P.*a3P - 5.*epsi^2.*aP.^2.*a4P) ...
        + 32.*af.^4.*(216.*aP.^2.*a2P + epsi^2.*(3.*a3P.^2 - a2P.*a4P)))./((32.*af.^3 + 5.*epsi^2.*aP.^2 ...
        - 4.*epsi^2.*af.*a2P).^5);
    
end

function b_three = b_three(af,aP,a2P,a3P,a4P,a5P)
    
    [b_three] = 65536.*af.^(15/2).*(10125.*epsi^4.*aP.^9 - 36450.*epsi^4.*af.*aP.^7.*a2P ...
        + 20.*epsi^4.*af.^2.*aP.^5.*(2034.*a2P.^2 + 575.*aP.*a3P) - 8192.*af.^9.*(8.*a2P.*a3P ...
        + 5.*aP.*a4P) + 400.*epsi^2.*af.^3.*aP.^3.*(-486.*aP.^4 - 18.*epsi^2.*a2P.^3 ...
        - 83.*epsi^2.*aP.*a2P.*a3P + 5.*epsi^2.*aP.^2.*a4P) + 4096.*af.^10.*a5P ...
        - 256.*af.^7.*(3240.*aP.^3.*a2P + 26.*epsi^2.*a2P.^2.*a3P + 10.*epsi^2.*aP.*(14.*a3P.^2 ...
        + 5.*a2P.*a4P) - 5.*epsi^2.*aP.^2.*a5P) + 4.*epsi^2.*af.^4.*aP.*(136080.*aP.^4.*a2P ...
        - 2088.*epsi^2.*a2P.^4 + 5840.*epsi^2.*aP.*a2P.^2.*a3P + 100.*epsi^2.*aP.^2.*(13.*a3P.^2 ...
        - 11.*a2P.*a4P) + 25.*epsi^2.*aP.^3.*a5P) + 1024.*af.^8.*(288.*aP.*a2P.^2 + 220.*aP.^2.*a3P ...
        + epsi^2.*(5.*a3P.*a4P - a2P.*a5P)) - 32.*epsi^2.*af.^5.*(12780.*aP.^3.*a2P.^2 ...
        + 4700.*aP.^4.*a3P - 58.*epsi^2.*a2P.^3.*a3P + 5.*epsi^2.*aP.*a2P.*(53.*a3P.^2 ...
        - 14.*a2P.*a4P) + 5.*epsi^2.*aP.^2.*(-5.*a3P.*a4P + a2P.*a5P)) + 64.*af.^6.*(6480.*aP.^5 ...
        + 468.*epsi^2.*aP.*a2P.^3 + 3480.*epsi^2.*aP.^2.*a2P.*a3P + 100.*epsi^2.*aP.^3.*a4P ...
        + epsi^4.*(15.*a3P.^3 - 10.*a2P.*a3P.*a4P + a2P.^2.*a5P)))./((32.*af.^3 + 5.*epsi^2.*aP.^2 ...
        - 4.*epsi^2.*af.*a2P).^7);
    
end

end
