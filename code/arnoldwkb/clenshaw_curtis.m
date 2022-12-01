function I = clenshaw_curtis(f,a,b,N)
%
% I = clenshaw_curtis(f,a,b,N)
%
% This program integrates numerically a function f on the intervall [a,b]
% using the well-known Clenshaw-Curtis algorithm (cf. C. W. Clenshaw, A. R.
% Curtis "A method for numerical integration on an automatic computer",
% Numerische Mathemathik 2, 197-205 (1960)). Comments in this program may
% refer to Equations from this article.
%
% A Chebyshev grid on [-1,1] with N collocation points gets created and the
% wanted integral gets computed by transforming the integral with the
% function gamma:[-1,1]->[a,b].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Inputs:
%   f   Function to be integrated as a function handle
%   a   Lower bound of intervall
%   b   Upper bound of intervall
%   N   Number of Chebyshev collocation points
%
% Outputs:
%   I   The approximated value of the integral of f from a to b
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Contact:  jannis.koerner@tuwien.ac.at (Jannis Körner)
%
% Institute of Analysis and Scientific Computing, Technische Universität
% Wien, Wiedner Hauptstr. 8-10, 1040 Wien, Austria
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Chebyshev grid on [-1,1] with N collocation points.
xGrid = cos(pi*(0:N)/N)';

% Transforming the integration domain gamma([-1,1])=[a,b]
gamma = @(y) (b/2)*(1+y)+(a/2)*(1-y); % Transformation
gamma_det = (b-a)/2; % Jacobian determinant

f_tilde = @(y) f(gamma(y));
F = f_tilde(xGrid);% F_s
a_n_vec = zeros(1,length(xGrid));

% Compute the coefficients a_n using Eq. (11)
for n=1:N+1
    summ = 0;
    for s = 2:N
        summ = summ + F(s)*cos((pi*(s-1)*(n-1))/N);
    end
    summ = summ + (1/2)*F(1) + (1/2)*F(N+1)*cos(pi*(n-1));
    a_n_vec(n) = (2/N)*summ;
end

% Compute the coefficients b_n using Eq. (7)
b_n_vec = zeros(1,length(a_n_vec));
for n = 2:1:N
    b_n_vec(n) = (1/(2*(n-1)))*(a_n_vec(n-1)-a_n_vec(n+1));
end
b_n_vec(N+1) = (1/(2*N))*a_n_vec(N);

% Applying Eq. (9)
I = 2*sum(b_n_vec(2:2:end)); % Eq. (9)
I = gamma_det*I; % Multiplying the Jacobian from the transformation

end
