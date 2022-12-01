 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  This file contains code for constructing phase functions that represent solutions
!  of Legendre's equation on the interval [-1,1].
!
!  The Legendre function of the first kind of degree nu (P_nu) is defined for x in
!  [-1,1] in terms of Gauss's hypergeometric function F via the formula
!
!     P_nu(x) = F(-nu, nu+1; 1; 1/2 - 1/2x).
!
!  When nu is not an integer, the function of the second kind (Q_nu) is
!
!     Q_nu(z) = 1/2 P_nu(x) ( \log( (x+1)/(x-1) ) - 2 \psi(\nu+1) - \psi(1)) 
!
!       sin(pi nu)  \infty  gamma (l - nu) gamma(nu + l +1) ( \psi(l+1) - \psi(1))  (1-x)^l
!     - ----------   sum    ------------------------------------------------------  ------- ,
!           pi       l=1                         (l!)^2                               2^l
!
!  where \psi denotes the logarithmic derivative of the gamma function.  When nu = n > 0,
!
!     Q_n(z) = 1/2 P_n(x) ( \log( (x+1)/(x-1) ) - 2 \psi(n+1) - \psi(1)) 
!
!           n           (n+l)!( \psi(n+1) - \psi(1))   (1-x)^l
!     +    sum (-1)^l   ------------------------------ -------.
!          l=1                   (l!)^2                  2^l
!
!  Both are solutions of Legendre's equation
!
!     (1-t)^2 y''(t) - 2 t y'(t) + nu (nu+1) y(t) = 0                                   (1)
!
!  Rather than construct a phase functions for (1), this code constructs phase 
!  function for the equation
!
!     z''(u) + ( ) z(u) = 0                                                             (2)
!
!  which is obtained from (1) via a change of variables.  Specifically,
!  if y is a solution of (1) then the function z defined via
!
!     z(u) = y ( 1 - exp(u)) ( 2 - exp(u) )^(1/2)                                       (3)
!
!  is  solution of (2).  The advantage of Equation (3) over (1) is that the 
!  coefficient in (3) is smooth, which is useful when representing phase
!  functions as tensor products of polynomials in the variables t and dnu.
!
!  Real values a1, a2, b1, and b2 such that
!
!    P_nu ( 1 - exp(u) ) (2 - exp(u) )^(1/2) = a_1 sin ( \alpha(t,nu) - a_2)
!
!  and
!
!    Q_nu ( 1 - exp(u) ) (2 - exp(u) )^(1/2) = a_1 sin ( \alpha(t,nu) - a_2)
!
!  are computed using the observation that 
! 
!
!  The following subroutines should be regarded as public:
!
!  kummer_legendre_phase - construct a phase function which represents solutions
!    of (2) 
!
!  kummer_legendre_pcoefs - return coefficients a1 and a2 such that (3) represents
!    the function P_nu
!
!  kummer_legendre_qcoefs - return coefficients a1 and a2 such that (4) represents
!    the function Q_nu
!
!  kummer_legendre_precurrence - evaluate the function P_nu using the well-known
!    3-term recurrence relation satisfied by solutions of (1)
!
!  kummer_legendre_qrecurrence - evaluate the function Q_nu using the well-known
!    3-term recurrence relation satisfied by solutions of (1)
!
!  kummer_legendre_pvalue - evaluate the function P_nu at the point 0 in
!    the event that nu is large using an asymptotic expansion
!
!  kummer_legendre_qvalue - evaluate the function Q_nu at the point 0 in
!    the event that nu is large using an asymptotic expansion
!
!  kummer_legendre_phasecoefs - construct an expansion of the phase function
!    in tensor products of dnu and t 
!
!  kummer_legendre_phasecoefs_eval - evaluate an expansion constructed by
!    kummer_legnedre_phasecoefs
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module kummer_legendre

use utils
use chebyshev
use tensor
use kummer

implicit double precision (a-h,o-z)

double precision, private     :: dlambdasq, pi, eulergamma

data pi          / 3.14159265358979323846264338327950288d0 /
data eulergamma  / 0.577215664901532860606512090082402431d0 /

contains


subroutine kummer_legendre_phase(eps,dnu,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alpha,alphap,alphapp)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: k
double precision, intent(in)               :: eps, dnu
double precision, intent(in)               :: xscheb(k),chebintl(k,k),chebintr(k,k),ucheb(k,k)
integer, intent(out)                       :: nints
double precision, allocatable, intent(out) :: ab(:,:),alpha(:,:),alphap(:,:),alphapp(:,:)

!
!  Construct a phase functions for the differential equation (2).  
!
!  Input parameters:
!    dnu - the degree of the Legendre function
!    (k,xscheb,chebintl,chebintr,ucheb) - k is the order of the Chebyshev expansion
!      to use and 
!
!  Output parameters:
!

a         =  0.0d0
b         = 15.0d0

ifleft    = 1
dlambdasq = 2.0d0 + 4*dnu + 4*dnu**2

!
!  Construct the phase function and its inverse 
!

call elapsed(t1)

if (dnu .le. 1.0d4) then
call kummer_adap(eps,a,1.0d0,qlegendre,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)
call kummer_extend(eps,a,b,qlegendre,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)
else
call kummer_adap(eps,b/2,b,qlegendre,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)
call kummer_extend(eps,a,b,qlegendre,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)
endif

call kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alpha,alphap,alphapp)

call elapsed(t2)

t_phase = t2-t1

! call prini("after kummer_adap, nints = ",nints)
! call prin2("after kummer_adap, ab = ",ab)

end subroutine



subroutine kummer_legendre_pcoefs(dnu,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alpha,alphap,alphapp,a1,a2)
implicit double precision (a-h,o-z)

integer, intent(in)             :: k,nints
double precision, intent(in)    :: dnu
double precision, intent(in)    :: xscheb(k),chebintl(k,k),chebintr(k,k),ucheb(k,k)
double precision, intent(in)    :: ab(k,nints),alpha(k,nints),alphap(k,nints),alphapp(k,nints)
double precision, intent(out)   :: a1,a2

a1 = sqrt(2.0d0/pi)
a2 = pi/2-mod(pi/2*dnu,2*pi)

! call kummer_legendre_pvalue(dnu,val1)
! call kummer_legendre_pvalue(dnu+1.0d0,val2)

! ifleft = 1
! ya     = val1
! ypa    = val1/2-(dnu+1.0d0)*val2

! call kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,ya,ypa,a1,a2)


end subroutine


subroutine kummer_legendre_qcoefs(dnu,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alpha,alphap,alphapp,a1,a2)
implicit double precision (a-h,o-z)

integer, intent(in)             :: k,nints
double precision, intent(in)    :: dnu
double precision, intent(in)    :: xscheb(k),chebintl(k,k),chebintr(k,k),ucheb(k,k)
double precision, intent(in)    :: ab(k,nints),alpha(k,nints),alphap(k,nints),alphapp(k,nints)
double precision, intent(out)   :: a1,a2



a1 = sqrt(pi/2)
a2 = -mod(pi/2*dnu,2*pi)


! call kummer_legendre_qvalue(dnu,val1)
! call kummer_legendre_qvalue(dnu+1.0d0,val2)

! ifleft = 1
! ya     = val1
! ypa    = val1/2-(dnu+1.0d0)*val2


! call kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,ya,ypa,a1,a2)



end subroutine


subroutine qlegendre(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)   :: t
double precision, intent(out)  :: val

dd  = 1.0d0/(2*exp(t)-1.0d0)
val = 0.25d0*dd**2 + 0.25d0*dd*dlambdasq

end subroutine


subroutine kummer_legendre_qvalue(dnu,val)
implicit double precision (a-h,o-z)

!
!  Evaluate the function Q_dnu at the point 0 in the case that dnu is large.
!  This value, which is equal to
!
!       Pi^(1/2) gamma( (1+dnu)/2 )
!     -------------------------------- sin (pi/2 dnu)
!        2 gamma((2+dnu)/2)
!
! is computed via a power series in dnu around infinity.
!

dd = -sin(pi/2 * dnu)*sqrt(pi)/(2*sqrt(2.0d0))

val = -(1/dnu)**1.5q0/2.q0+(1/dnu)**2.5q0/1.6q1+  &
(5*(1/dnu)**3.5q0)/6.4q1-(21*(1/dnu)**4.5q0)/1.024q3-  &
(399*(1/dnu)**5.5q0)/4.096q3+2.q0*sqrt(1/dnu)

val = dd * val 

end subroutine



subroutine kummer_legendre_pvalue(dnu,val)
implicit double precision (a-h,o-z)

!
!  Evaluate the function P_dnu at the point 0 in the case that dnu is large.
!  This value, which is equal to
!
!                Pi^(1/2)
!     --------------------------------,
!      Gamma(1/2-dnu/2) Gamma(1+dnu/2)
!
! is computed via a power series in dnu around infinity.
!

dd = cos(pi/2 * dnu)/ sqrt(2*pi)

val =  ((-399-8.4q1*dnu+3.2q2*dnu**2+2.56q2*dnu**3-2.048q3*dnu**4+  &
8.192q3*dnu**5))/(4.096q3*dnu**5.5q0)

val = dd * val 

end subroutine




subroutine kummer_legendre_precurrence(dnu,t,val)
implicit double precision (a-h,o-z)

!
!  Use the recurrence relation satisfied by solutions of Legendre's equation
!  to evaluate P_dnu(t) with t in the interval [-1,1].
!
!  Input parameters:
!    dnu - the degree of the Legendre function to evaluation
!    t - the point at which to evaluate it in the interval [-1,1]
!
!  Output parameters:
!    val - the value of P_dnu(t)
!

if (dnu .le. 2) then
call kummer_legendre_ptaylor(dnu,t,val)
return
endif

n     = floor(dnu)
delta = dnu-n

call kummer_legendre_ptaylor(delta,t,val0)
call kummer_legendre_ptaylor(delta+1.0d0,t,val1)

dd = delta+1

do i=2,n
val  = (2*dd+1)*t*val1 - dd * val0
val  = val / (dd+1)
dd   = dd+1
val0 = val1
val1 = val
end do

end subroutine


subroutine kummer_legendre_qrecurrence(dnu,t,val)
implicit double precision (a-h,o-z)

!
!  Use the recurrence relation for Legendre functions of the first
!  kind in order to evaluate P_dnu(t) with t in the interval [-1,1]
!  and dnu a positive real number.
!
!
!  Input parameters:
!    dnu - the degree of the Legendre function to evaluation
!    t - the point at which to evaluate it in the interval [-1,1]
!
!  Output parameters:
!    val - the value of P_dnu(t)

!
!  Handle the case when dnu is small
!

if (dnu <= 2.0d0) then
call kummer_legendre_qtaylor(dnu,t,val)
return
endif

n     = floor(dnu)
delta = dnu-n

call kummer_legendre_qtaylor(delta,t,val0)
call kummer_legendre_qtaylor(delta+1.0d0,t,val1)

dd = delta+1

do i=2,n
val  = (2*dd+1)*t*val1 - dd * val0
val  = val / (dd+1)
dd   = dd+1
val0 = val1
val1 = val
end do

end subroutine


subroutine kummer_legendre_ptaylor(dnu,t,val)
implicit double precision (a-h,o-z)

!
!  Use a Taylor series in order to evaluate the the Legendre function of the first kind of
!  order dnu at a point t in the interval [-1,1].  The degree dnu must be in the
!  interval [0,2].
!

maxterms = 100

if (dnu < 0 .OR. dnu >  2) then
print *,"legendre_taylor called with dnu not in the range 0 < dnu < 2"
stop
endif

a   = 1
val = 0
dd  = 1

do i=1,maxterms

delta = a*dd
if (abs(delta) .lt. 1.0d-30) exit

val  = val + delta
dd  = dd  * (1-t)/2
a   = a * (-dnu+i-1)*(dnu+i)/(i**2)

end do


end subroutine


subroutine kummer_legendre_qtaylor(dnu,t,val)
implicit double precision (a-h,o-z)


!
!  Use a Taylor series in order to evaluate the the Legendre function of the 
!  second kind of order dnu at a point t in the interval [-1,1] in the event
!  that dnu is relatively small (< 10).
!

maxterms = 100

call kummer_legendre_digamma(dnu+1.0d0,r1)
r   = 0.5d0*log( (1+t)/(1-t)) -  eulergamma - r1

a   = 1
val = 0
dd  = 1

do i=1,maxterms

delta = a*dd * r
if (abs(delta) .lt. 1.0d-30) exit

val  = val + delta
dd  = dd  * (1-t)/2
a   = a * (-dnu+i-1)*(dnu+i)/(i**2)
r   = r + 1.0d0/i

end do

end subroutine


subroutine kummer_legendre_digamma(t,val)
implicit double precision (a-h,o-z)

double precision, intent(in)   :: t
double precision, intent(out)  :: val

!
!  Evaluate the logarithm derivative of the gamma function (also known as digamma)
!  at a point t in the interval (0,\infty).
!

!
!  Use the recurrence
!
!    phi(t) = phi(t+1) - 1/t
!
!  to find val0 such that
!
!    phi(t) = -val0 + phi(t0) 
!
!  with t0 large enough so that the well-known asymptotic expansion
!  is applicable to phi(t0).
!

val0 = 0.0d0
t0   = t

do while (t0 .lt. 100)
val0 = val0 + 1.0d0/t0
t0   = t0 + 1
end do

!
!  Now use the well-known asymptotic expansion to approximate phi(t0) and
!  add it to val0.
!

dd  = 1.0d0/(t0)
val = -val0  + log(t0) - 0.5d0*dd
dd  = dd*dd
val = val - 1.0d0/12.0d0  * dd
val = val + 1.0d0/120.0d0 * dd**2
val = val - 1.0d0/252.0d0 * dd**3
val = val + 1.0d0/240.0d0 * dd**4
val = val - 1.0d0/132.0d0 * dd**5

end subroutine



subroutine kummer_legendre_expansion(eps,nx,ny,c,d,nints,ab,coefs,coefsder)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nx,ny
double precision, intent(in)  :: c,d,eps

double precision, allocatable, intent(out) :: ab(:,:),coefs(:,:),coefsder(:,:)
integer, intent(out)                       :: nints

!
!  
!
!  Input parameters:
!
!  Output parameters:
!
!


double precision, allocatable :: xscheb(:),whtscheb(:),chebintl(:,:),chebintr(:,:), &
   ucheb(:,:),vcheb(:,:)

double precision, allocatable :: xscheb1(:),whtscheb1(:),chebintl1(:,:),chebintr1(:,:), &
   ucheb1(:,:),vcheb1(:,:)

double precision, allocatable :: xscheb2(:),whtscheb2(:),chebintl2(:,:),chebintr2(:,:), &
   ucheb2(:,:),vcheb2(:,:)

double precision, allocatable :: umatrix(:,:)
double precision, allocatable :: vals(:,:),valsder(:,:)

double precision, allocatable :: alpha(:,:),alphap(:,:),alphapp(:,:),alphappp(:,:)
double precision, allocatable :: alpha2(:,:),alphap2(:,:),alphapp2(:,:),alphappp2(:,:)
double precision, allocatable :: ab2(:,:)



ifleft = 1

a         =  0.0d0
b         = 15.0d0

!
!  Fetch the three Chebyshev quadratures and the tensor product coefficient
!  matrix
!

k = (nx*2)/3
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)

call chebexps(nx,xscheb1,whtscheb1,ucheb1,vcheb1,chebintl1,chebintr1)
call chebexps(ny,xscheb2,whtscheb2,ucheb2,vcheb2,chebintl2,chebintr2)

call tensor_umatrix(nx,ny,xscheb1,xscheb2,umatrix)

!
!  Construct a phase function for the smallest value of dnu with somewhat
!  higher accuracy in order to fix a discretization scheme for the phase
!  functions.
!

dnu = c
call prin2("before kummer_legendre_phase, dnu = ",dnu)
call kummer_legendre_phase(eps/10,dnu,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alpha,alphap,alphapp)
call prini("nints = ",nints)

!
!  Form the matrix of values
!

allocate(vals(nx*ny,nints),valsder(nx*ny,nints),coefs(nx*ny,nints),coefsder(nx*ny,nints))

c0 = 1/d
d0 = 1/c

do j = 1,ny

dnu = (d0-c0)/2 * xscheb2(j) + (d0+c0)/2
dnu = 1.0d0/dnu

call prin2("before kummer_legendre_phase, dnu = ",dnu)
call kummer_legendre_phase(eps,dnu,nx,xscheb1,chebintl1,chebintr1,ucheb1, &
  nints2,ab2,alpha2,alphap2,alphapp2)
call prini("nints2 = ",nints2)

!
!  Evaluate the resulting phase function 
!

do int=1,nints
a0 = ab(1,int)
b0 = ab(2,int)
do i=1,nx
x = (b0-a0)/2 * xscheb1(i) + (b0+a0)/2

call chebpw_eval(nints2,ab2,nx,xscheb1,alpha2,x,val)
call chebpw_eval(nints2,ab2,nx,xscheb1,alphap2,x,der)

vals(i+(j-1)*nx,int) = val/dnu
valsder(i+(j-1)*nx,int) = der/dnu
end do

end do

end do

!
!  Compute the coefficient expansions
!

do int=1,nints
coefs(:,int)    = matmul(umatrix,vals(:,int))
coefsder(:,int) = matmul(umatrix,valsder(:,int))
end do

end subroutine


subroutine kummer_legendre_evalp(nx,ny,c,d,nints,ab,coefs,coefsder,dnu,t,val)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nx,ny,nints
double precision, intent(in)  :: c,d,t,dnu
double precision, intent(in)  :: ab(2,nints),coefs(nx*ny,nints),coefsder(nx*ny,nints)
double precision, intent(out) :: val


a1 = sqrt(2.0d0/pi)
a2 = pi/2-mod(pi/2*dnu,2*pi)

u  = -log(1-t)

c0 = 1/d
d0 = 1/c


dnu0 = 1.0d0/dnu
call tensorpw_eval(nx,ny,nints,ab,c0,d0,coefs,u,dnu0,alpha)
call tensorpw_eval(nx,ny,nints,ab,c0,d0,coefsder,u,dnu0,alphap)

alpha       = alpha * dnu
alphap      = alphap * dnu

val = a1*sin(alpha+a2)/sqrt(alphap)
val = val /sqrt(1+t)

end subroutine


end module
