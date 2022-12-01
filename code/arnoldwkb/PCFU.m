function [PCFU,dPCFU] = PCFU(nu,z)

% Parabolic Cylinder Function U(nu,z)
% Version 05.09.2020
%
% [PCFU,dPCFU] = PCFU(nu,z)
%
% We evaluate values of the parabolic cylinder function by using 
% U(a,z) = U(a,0)*u1(a,z)+U'(a,0)*u2(a,z) with functions
% u1(a,z)= exp(-z^2/4)*M(a/2+1/4,1/2,z^2/2)
% u2(a,z)= z*exp(-z^2/4)*M(a/2+3/4,3/2,z^2/2)
% Here M denotes the confluentel hypergeometric function of first
% kind. Since these functions are multivalued one can not simply replace z
% with -z. Therefore, we use the relation
% U(a,-z) = -sin(pi*a)*U(a,z)+pi/gamma(1/2+a)*V(a,z)
% with V(a,z)= V(a,0)*u1(a,z)+V'(a,0)*u2(a,z).
% Knowing asymptotic values of U,U',V and V' for z=0 as well as using the
% formula d/dz M(a,b,z) = (a/b)*M(a+1,b+1,z), we can eventually evaluate U
% for every real argument.
% Formulas etc. are taken from the Digital Library of Mathematical
% Functions (DLMF).
% 
% There are other values of U known "exactly" besides z=0, which
% could easily be implemented seperately here.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Inputs:
%   nu      real valued parameter
%   z       real valued argument
%
% Outputs:
%   PCFU    U(nu,z): Parabolic Cylinder Function at (nu,z)
%   dPCFU   U'(nu,z): Derivative of the PCF at (nu,z)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Contact:  jannis.koerner@tuwien.ac.at (Jannis Körner)
%
% Institute of Analysis and Scientific Computing, Technische Universität
% Wien, Wiedner Hauptstr. 8-10, 1040 Wien, Austria
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initializations
zn = length(z);
indx0 = find(abs(z)<1e-15);
indx1 = find(z>1e-15);
indx2 = find(z<-1e-15);

PCFU = zeros(1,zn);
dPCFU = zeros(1,zn);
PCFV = zeros(1,zn);
dPCFV = zeros(1,zn);

% Exact values for z=0 (here abs(z)<1e-15)    
U_nu_0 = ( sqrt(pi)/( 2^((nu+0.5)/2)*gamma(3/4+nu/2) ) );
dU_nu_0 = -( sqrt(pi)/( 2^((nu-0.5)/2)*gamma(1/4+nu/2) ) );
V_nu_0 = pi*2^((1/2)*nu+1/4)/((gamma(3/4-(1/2)*nu))^2*gamma(1/4+(1/2)*nu));
dV_nu_0 = pi*2^((1/2)*nu+3/4)/((gamma(1/4-(1/2)*nu))^2*gamma(3/4+(1/2)*nu));


% Value at z=0 (abs(z)<1e-15)
PCFU(indx0) = U_nu_0;
dPCFU(indx0)  = dU_nu_0;

u1 = @(y) exp((-1/4)*y.^2).*hypergeom((1/2)*nu+1/4,1/2,(1/2)*y.^2);
u2 = @(y) y.*exp((-1/4)*y.^2).*hypergeom((1/2)*nu+3/4,3/2,(1/2)*y.^2);

du1 = @(y) y.*exp((-1/4)*y.^2).*((-1/2).*hypergeom((1/2)*nu+1/4,1/2,(1/2)*y.^2)...
    +(nu+1/2).*hypergeom((1/2)*nu+5/4,3/2,(1/2)*y.^2));
du2 = @(y) exp((-1/4)*y.^2).*(hypergeom((1/2)*nu+3/4,3/2,(1/2)*y.^2)...
    -(1/2)*y.^2.*hypergeom((1/2)*nu+3/4,3/2,(1/2)*y.^2)...
    +y.^2.*(nu/3+1/2).*hypergeom((1/2)*nu+7/4,5/2,(1/2)*y.^2));

% Values at z>0 (z>1e-15)
PCFU(indx1) = U_nu_0.*u1(z(indx1))+dU_nu_0.*u2(z(indx1));
dPCFU(indx1) = U_nu_0.*du1(z(indx1))+dU_nu_0.*du2(z(indx1));


PCFV(indx2) = V_nu_0*u1(-z(indx2))+dV_nu_0*u2(-z(indx2));
dPCFV(indx2) = V_nu_0*du1(-z(indx2))+dV_nu_0*du2(-z(indx2));

% Values at z<0 (z<-1e-15)
PCFU(indx2) = -sin(pi*nu)*(U_nu_0.*u1(-z(indx2))+dU_nu_0.*u2(-z(indx2)))...
    +(pi/gamma((1/2)+nu))*PCFV(indx2);
dPCFU(indx2) = sin(pi*nu)*(U_nu_0.*du1(-z(indx2))+dU_nu_0.*du2(-z(indx2)))...
    -(pi/gamma((1/2)+nu))*dPCFV(indx2);
  
end