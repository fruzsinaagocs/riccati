!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  This module contains code for constructing nonoscillatory phase functions which
!  represent solutions of second order linear ordinary differential equations of
!  the form
!        
!       y''(t) + q(t) y(t) = 0   for all a <= t <= b,                                   (1)
!
!  where q is smooth and strictly positive on the interval (a,b).  It implements a 
!  variant of the windowing algorithm described in 
!
!      "On the numerical solution of second order linear ordinary differential 
!      equations in the high-frequency regime"
!      James Bremer, arxiv:1409.6049
!
!  We say that a function alpha is a phase function for (1) if the functions
!
!          cos(alpha(t))                         sin(alpha(t))
!        -----------------        and          -----------------
!        |alpha'(t)|^(1/2)                     |alpha'(t)|^(1/2)
!
!  form a basis in the space of solutions of (1).  Any phase function
!  satisfies Kummer's equation
!
!                             1   alpha'''(t)     3   (alpha''(t) )^2
!      (alpha'(t))^2 = q(t)- ---  ----------   + ---  ---------------                   (2)
!                             2    alpha'(t)      4   (alpha'(t) )^2
!      
!  Moreover, it follows that any solution of (1) can, in fact, be represented in
!  the form
!
!                   a1 sin( alpha(t) + a2 ) 
!      y(t) =      -------------------------                                            (3)
!                     (alpha'(t))^(1/2)
!
!  with 0 < a2 <= pi.  Note that since the magnitude of alpha is on the order of lambda,
!  formula (3) involves the evaluation of trigonometric functions of large arguments.
!  A loss of relative precision on the order of O(lambda) is expected as a consequence.
!  
!  The expression (3) is highly conducive to computing the roots of y.  If 
!  
!     t_1 < t_1 < t_2 < ... < t_m
!
!  are the zeros of the function y defined by (3) in the interval [a,b], then
!
!     t_k = \alpha^{-1}(\pi k - a_2),                                                   (4)
!
!  and the value of y' at the point t_k is
!
!     y'(t_k) = (-1)^k a_1 sqrt(alpha'(t_k)).                                           (5)
!
!  Note that formulas (4) and (5) can be evaluated without calculating trigonometric
!  functions of large arguments and the concominant loss of precision.
!
!  The phase function alpha and its derivatives are represented via piecewise
!  Chebyshev expansions; more specifically, their values at the nodes
!  of the k-point Chebyshev grid on a collection of subintervals are stored.
!  We refer to such a scheme for representing functions as a "discretization
!  scheme" (see the file chebyshev.f90 for more information).   The discretization 
!  schemes used to represent phase functions and their derivatives are
!  constructed adaptively in this code.
!
!  The following subroutines should be regarded as publically callable:
!
!   kummer_adap - construct the derivative of a nonoscillatory function 
!     which satisfies Kummer's equation (2) using the windowing algorithm.    
!
!   kummer_extend - extend the derivative alpha' of a phase function satisfying
!     (2) to a larger interval.  By combining this subroutine with kummer_adap
!     the user can  control the interval on which the windowing algorithm is applied.
!     This is useful because in some cases 
!
!   kummer_phase - integrate the derivative of a nonoscillatory phase function 
!     in order to produce a phase function which is 0 at one of the endpoints
!     the interval [a,b].
!
!   kummer_phase_inverse - compute the inverse of a phase function via 
!     Newton's method; this routine also returns a discretrization scheme 
!     for representing the inverse function.
!
!   kummer_coefs - compute the coefficients a1 and a2 in the representation (3) 
!     of a solution y of (1) given the values of y and its derivative at the 
!     whichever endpoint of [a,b] the phase function is 0 at
!
!   kummer_coefs2 - compute the coefficients in the representing (3) of
!     a solution y of (1) given the values of y and its derivative at
!     a point c in the interval [a,b]
!
!     IMPORTANT NOTE:  The accuracy of the coefficients computed by this 
!     routine depends  on the magnitude of alpha at the point c.  If alpha(c) 
!     is large in magnitude, then a loss of precision will be incurred
!     due to the evaluation of trigonometric functions of large arguments.
!
!   kummer_eval - evaluate a function of the form (3) at a specified point in
!     the interval [a,b] 
!
!   kummer_zeros_count - return the number of zeros of a function of the form
!     (3) on the interval [a,b] given the phase function alpha and the
!     coefficients a1 and a2
!
!   kummer_zero - calculate the jth zero of a function of the form (3) on the
!     interval [a,b] given the inverse of the phase function representing 
!     solutions of (1) and the coefficients a1 and a2
!
!   kummer_zeroder - calculate the jth zero of a function of the form (3) on
!     the interval [a,b] as well as the value of its derivative at that point
!     given the inverse of the phase function representing solutions of (1),
!     the derivative of the phase function representing solutions of (1),
!     and the coefficients a1 and a2
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module kummer

use utils
use odesolve

!
!  The following variables are shared between the subroutines kummer_adap,
!  kummer_solve and odekummer (which is passed to odesolve).
!

integer, private                       :: s_nints,s_k,s_ifwindowed
double precision, private              :: s_a,s_b,s_c,s_alpha,s_val0

procedure(kummerfun), pointer :: qfunptr

interface

subroutine kummerfun(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)  :: t
double precision, intent(out) :: val
end subroutine

end interface

contains


subroutine odekummer(t,y,yp,f,dfdy,dfdyp)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: t,y,yp
double precision, intent(out) :: f,dfdy,dfdyp


call qfunptr(t,qval)

if (s_ifwindowed .eq. 1) then
phi  = (erf(s_alpha*(t-s_c))+1.0d0)/2
qval = s_val0*(1.0d0-phi) + phi * qval
else
endif

!
!  Evaluate f and its derivatives.
!

f     = -2*y**3 + 2*qval * y + 1.5d0 * (yp)**2/y
dfdy  = -6*y**2 - 3*yp**2/(2*y**2) + 2 * qval
dfdyp = 3*yp/y

end subroutine



subroutine kummer_adap(eps,a,b,qfun,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)
implicit double precision (a-h,o-z)

double precision, intent(in)               :: a,b
integer, intent(in)                        :: k
integer, intent(out)                       :: nints
double precision, allocatable, intent(out) :: ab(:,:)
double precision,intent(in)                :: xscheb(k),chebintl(k,k),chebintr(k,k),ucheb(k,k)
double precision, allocatable, intent(out) :: alphap(:,:),alphapp(:,:)
procedure (kummerfun)                      :: qfun

!
!  Construct a nonoscillatory function alpha' which satisfies Kummer's equation (2).
!  The solution is represented via a discretization scheme which is constructed 
!  adaptively.
!
!  Input parameters:
!    eps - the precision for the calculations
!    (a,b) - the interval on which to compute the phase function
!    qfun - a user-specified subroutine for evaluting the coefficient q in (1);
!      it must conform to the interface kummerfun
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!    chebintl - the left Chebyshev spectral integration matrix as returned by 
!      chebexps
!    chebintr - the right Chebyshev spectral integration matrix as returned by 
!      chebexps
!    ucheb - the values-to-coefficients matrix returned by chebexps
!   
!
!  Output parameters:
!    (nints,ab) - the discretization scheme used to represent the solution
!   alphap - a (k,nints) array specifying the values of the derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!   alphapp - a (k,nints) array specifying the values of the second derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!

double precision, allocatable   :: ab00(:,:),ab0(:,:),alphappp(:,:)
data pi /3.14159265358979323846264338327950288d0/

call mach_zero(eps0)

qfunptr => qfun

!
!  Copy the windowing parameters for use by odekummer.
!
s_a          = a
s_b          = b
s_alpha      = 12.0d0/(b-a)
s_c          = (a+b)/2
call qfun(s_c,s_val0)

!
!  Adjust the coefficient in the bump function in the event that
!  calculations are being performed in extended precision arithmetic.
!

if (eps0 .lt. 1.0d-20) then
s_alpha      = 2d0*s_alpha
endif

!
!  Adptively discretize q.  
!


call chebadap(ier,eps,a,b,qfun,k,xscheb,ucheb,nints00,ab00)
if (ier .ne. 0) then
call prini("in kummer_adap, after chebadap, ier = ",ier)
stop
endif

!
!  Solve the windowed problem with the adaptive version of ode_solve_ivp on one
!  small interval only
!

s_ifwindowed = 1

ya    = sqrt(s_val0)
ypa   = 0.0d0

call ode_solve_ivp_adap(ier,eps,nints00,ab00,k,xscheb,chebintl,ucheb,nints0,ab0,  &
   alphap,alphapp,alphappp,odekummer,ya,ypa)

if (ier .ne. 0) then
call prini("after ode_solve_ivp_adap, ier = ",ier)
stop
endif


!
!  Set the desired terminal boundary conditions.
!

s_ifwindowed = 0

yb  = alphap(k,nints0)
ypb = alphapp(k,nints0)

!
!  Solve the original problem, going backwards.
!

call ode_solve_tvp_adap(ier,eps,nints0,ab0,k,xscheb,chebintr,ucheb,nints,ab,  &
   alphap,alphapp,alphappp,odekummer,yb,ypb)

if (ier .ne. 0) then
call prini("after ode_solve_tvp_adap, ier = ",ier)
stop
endif


end subroutine



subroutine kummer_extend(eps,a,b,qfun,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)
implicit double precision (a-h,o-z)

double precision, intent(in)                  :: a,b
integer, intent(in)                           :: k
integer, intent(inout)                        :: nints
double precision, allocatable, intent(inout)  :: ab(:,:)
double precision,intent(in)                   :: xscheb(k),chebintl(k,k),chebintr(k,k),ucheb(k,k)
double precision, allocatable, intent(inout)  :: alphap(:,:),alphapp(:,:)
procedure (kummerfun)                         :: qfun

!
!  Extend an existing solution of Kummer's equation given on an interval [c,d]
!  to the interval [a,b].
!
!  Input parameters:
!    eps - the precision for the calculations
!    (a,b) - the extents of the new interval
!    (nints,ab) - the discretization scheme used to represent the existing
!      solution of Kummer's equation on the interval [c,d].
!    qfun - a user-specified subroutine for evaluting the coefficient q in (1);
!      it must conform to the interface kummerfun
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!    chebintl - the left Chebyshev spectral integration matrix as returned by 
!      chebexps
!    chebintr - the right Chebyshev spectral integration matrix as returned by 
!      chebexps
!    ucheb - the values-to-coefficients matrix returned by chebexps
!    alphap - a (k,nints) array specifying the values of the derivative of
!      the existing solution at the nodes of the k-point Chebyshev grids on the intervals 
!      specified by ab
!    alphapp - a (k,nints) array specifying the values of the second derivative of
!      alpha at the nodes of the k-point Chebyshev grids on the intervals 
!      specified by ab
!
!  Output parameters:
!   (nints,ab) - a new discretization scheme on the interval [a,b]
!   alphap - a (k,nints) array specifying the values of the derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!   alphapp - a (k,nints) array specifying the values of the second derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!
double precision, allocatable   :: ab0(:,:),alphap0(:,:),alphapp0(:,:)
double precision, allocatable   :: ab1(:,:),alphap1(:,:),alphapp1(:,:),alphappp1(:,:)
double precision, allocatable   :: ab2(:,:),alphap2(:,:),alphapp2(:,:),alphappp2(:,:)

double precision, allocatable   :: alphappp(:,:),ab00(:,:)

data pi /3.14159265358979323846264338327950288d0/

call mach_zero(eps0)


!
!  Make a copy of the existing solutions
!
nints0 = nints
allocate(ab0(2,nints),alphap0(k,nints),alphapp0(k,nints))
ab0      = ab
alphap0  = alphap
alphapp0 = alphapp

!
!  Find the extents of the interval on which the old solution is given
!

c = ab(1,1)
d = ab(2,nints)


!
!  Extend the solution to the left
!

if (a .lt. c) then


call chebadap(ier,eps,a,c,qfun,k,xscheb,ucheb,nints00,ab00)

s_ifwindowed = 0

yb    = alphap0(1,1)
ypb   = alphapp0(1,1)

call ode_solve_tvp_adap(ier,eps,nints00,ab00,k,xscheb,chebintr,ucheb,nints1,ab1,  &
   alphap1,alphapp1,alphappp1,odekummer,yb,ypb)

if (ier .ne. 0) then
call prini("after ivp_adap, ier = ",ier)
stop
endif

endif


!
!  Extend the solution to the right
!

if (d .lt. b) then


call chebadap(ier,eps,d,b,qfun,k,xscheb,ucheb,nints00,ab00)

s_ifwindowed = 0

ya    = alphap0(k,nints)
ypa   = alphapp0(k,nints)

call ode_solve_ivp_adap(ier,eps,nints00,ab00,k,xscheb,chebintl,ucheb,nints2,ab2,  &
   alphap2,alphapp2,alphappp2,odekummer,ya,ypa)

if (ier .ne. 0) then
call prini("after ivp_adap, ier = ",ier)
stop
endif

endif

!
!  Copy out the newly obtained solution 
!

deallocate(alphap,alphapp,ab)

if (a .lt. c .AND. d .lt. b) then

nints = nints0 + nints1 + nints2
allocate(ab(2,nints),alphap(k,nints),alphapp(k,nints))

ab(:,1:nints1)               = ab1
ab(:,nints1+1:nints1+nints0) = ab0
ab(:,nints1+nints0+1:nints)  = ab2

alphap(:,1:nints1)               = alphap1
alphap(:,nints1+1:nints1+nints0) = alphap0
alphap(:,nints1+nints0+1:nints)  = alphap2

alphapp(:,1:nints1)               = alphapp1
alphapp(:,nints1+1:nints1+nints0) = alphapp0
alphapp(:,nints1+nints0+1:nints)  = alphapp2

elseif (a .lt. c) then

nints = nints0 + nints1 
allocate(ab(2,nints),alphap(k,nints),alphapp(k,nints))

ab(:,1:nints1)               = ab1
ab(:,nints1+1:nints1+nints0) = ab0

alphap(:,1:nints1)               = alphap1
alphap(:,nints1+1:nints1+nints0) = alphap0

alphapp(:,1:nints1)               = alphapp1
alphapp(:,nints1+1:nints1+nints0) = alphapp0


else if (d .lt. b) then

nints = nints0 + nints2
allocate(ab(2,nints),alphap(k,nints),alphapp(k,nints))

ab(:,1:nints0)        = ab0
ab(:,nints0+1:nints)  = ab2

alphap(:,1:nints0)        = alphap0
alphap(:,nints0+1:nints)  = alphap2

alphapp(:,1:nints0)        = alphapp0
alphapp(:,nints0+1:nints)  = alphapp2

else

nints = nints0
allocate(ab(2,nints),alphap(k,nints),alphapp(k,nints))

ab(:,1:nints0)        = ab0
alphap(:,1:nints0)    = alphap0
alphapp(:,1:nints0)   = alphapp0

endif


end subroutine


subroutine kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alpha,alphap,alphapp)
implicit double precision (a-h,o-z)

integer, intent(in)                         :: k,nints,ifleft
double precision, intent(in)                :: ab(2,nints),alphap(k,nints),alphapp(k,nints)
double precision, intent(in)                :: xscheb(k),chebintl(k,k),chebintr(k,k),ucheb(k,k)
double precision, allocatable, intent(out)  :: alpha(:,:)

!
!  Construct a phase function alpha by integrating a solution alpha' of Kummer's
!  equation (2).
!
!  The user must specify whether or not the phase function alpha is to be 0 at
!  the point a (the left-hand side of the interval on which the ODE is given)
!  or at the point b (the  right-hand side of the interval on which the
!  ODE is given).
!
!  Input parameters:
!    ifleft - an integer parameter specifying whether the function alpha should
!     be zero on the left-hand side of the interval [a,b] or on the right-hand
!     side of the interval [a,b]
!
!      ifleft = 1  indicates that alpha should satisfy alpha(a) = 0
!      ifleft = 0  indicates that alpha should satisfy alpha(b) = 0 
!
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!    chebintl - the left Chebyshev spectral integration matrix as returned by 
!      chebexps
!    chebintr - the right Chebyshev spectral integration matrix as returned by 
!      chebexps
!    ucheb - the values-to-coefficients matrix returned by chebexps
!    (nints,ab) - the discretization scheme used to represent the solution
!   alphap - a (k,nints) array specifying the values of the derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!   alphapp - a (k,nints) array specifying the values of the second derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab   
!
!  Output parameters:
!

data pi /3.14159265358979323846264338327950288d0/
!
!  Construct alpha via spectral integration
!

allocate(alpha(k,nints))

if (ifleft .eq. 1) then
do int=1,nints
a0 = ab(1,int)
b0 = ab(2,int)

if (int .eq. 1) then
dd  = 0
dd0 = 0
else
dd = alpha(k,int-1)
endif 
alpha(:,int)   = matmul(chebintl*(b0-a0)/2,alphap(:,int))+dd
end do

else
do int=nints,1,-1
a0 = ab(1,int)
b0 = ab(2,int)

if (int .eq. nints) then
dd  = 0
dd0 = 0
else
dd = alpha(1,int+1)
endif 
alpha(:,int)   = matmul(chebintr*(b0-a0)/2,alphap(:,int))+dd
end do

endif

end subroutine



subroutine kummer_phase_inverse(nints,ab,k,xscheb,chebintl,ucheb,alpha,alphap,&
    nintsinv,abinv,alphainv,alphainvp)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: nints,k
double precision, intent(in)               :: xscheb(k),ab(2,nints),chebintl(k,k)
double precision, intent(in)               :: alpha(k,nints),alphap(k,nints),ucheb(k,k)
double precision, intent(out), allocatable :: alphainv(:,:),alphainvp(:,:),abinv(:,:)
integer, intent(out)                       :: nintsinv

!
!  Compute the inverse of the phase function alpha via Newton's method.
!  
!  Input parameters:
!    (nints,ab) - the discretization scheme for representing the phase
!      function alpha
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!    chebintl - the left Chebyshev spectral integration matrix as returned by 
!      chebexps
!    chebintr - the right Chebyshev spectral integration matrix as returned by 
!      chebexps
!    ucheb - the values-to-coefficients matrix returned by chebexps
!
!  Output parameters:
!
!    (nintsab,abinv) - the discretization scheme used to represent the inverse of
!      alpha
!   alphainv - a (k,nints) array specifying the values of the inverse of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals
!     in the discretization scheme     
!
!   alphainvp - a (k,nints) array specifying the values of the derivative of
!     the inverse of alpha at the nodes of the k-point Chebyshev grids on the 
!     intervals in the discretization scheme        
!
!

nintsinv = nints
allocate(alphainv(k,nints),alphainvp(k,nints),abinv(2,nints))

call mach_zero(eps0)
maxiters = 20
eps0     = sqrt(eps0)
nextra   = 3

!
!  Form the initial list of intervals for the inverse function.
!

do int=1,nints
a = ab(1,int)
b = ab(2,int)

call chebpw_eval(nints,ab,k,xscheb,alpha,a,a0)
call chebpw_eval(nints,ab,k,xscheb,alpha,b,b0)

abinv(1,int) = a0
abinv(2,int) = b0

end do


!
!  Use Newton's method to evaluate the inverse at each of the Chebyhev nodes 
!  on the grid defined by abinv; start at the right-hand side since alpha
!  is monotonically increasing.
!

do int = nints,1,-1

a0 = abinv(1,int)
b0 = abinv(2,int)
a  = ab(1,int)
b  = ab(2,int)


do i = k,1,-1

x  = (b0-a0)/2*xscheb(i) + (b0+a0)/2
t  = (b-a)/2*xscheb(i) + (b+a)/2

do iter=1,maxiters+1

if (iter .eq. maxiters+1) then
call prina("in kummer_phase_invert: maximum number of Newton iterations exceeded")
stop
endif

call chebpw_eval(nints,ab,k,xscheb,alpha,t,val)
call chebpw_eval(nints,ab,k,xscheb,alphap,t,der)

delta = (val-x)/der

if (abs(delta) .lt. eps0*(1+abs(t))) exit
t     = t - delta

end do


do iextra=1,nextra

call chebpw_eval(nints,ab,k,xscheb,alpha,t,val)
call chebpw_eval(nints,ab,k,xscheb,alphap,t,der)

delta = (val-x)/der
t     = t - delta

end do

alphainv(i,int)  = t
alphainvp(i,int) = 1.0d0/der

end do
end do

end subroutine



subroutine kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,y,yp,a1,a2)
implicit double precision (a-h,o-z)

integer, intent(in)              :: nints,k
double precision, intent(in)     :: xscheb(k),ab(2,nints),y,yp
double precision, intent(in)     :: alphap(k,nints),alphapp(k,nints)
double precision, intent(out)    :: a1,a2

!
!  Return the coefficients a1 and a2 in the representation (3) of a solution y
!  of the ordinary differential equation (1) given the values of y and its
!  derivative at either the point a (in the event that alpha(a) = 0) or
!  at the point b (in the event that alpha(b) = 0).
!
!  Input parameters:
!    ifleft -  an integer parameter indicating whether or not alpha(a) = 0
!      or alpha(b) = 0
!
!       ifleft = 0  indicates that alpha(b) = 0
!       ifleft = 1  indicates that alpha(a) = 0
!
!    (nints,ab) - the discretization scheme for representing the phase
!      function alpha
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!   alphap - a (k,nints) array specifying the values of the derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!   alphapp - a (k,nints) array specifying the values of the second derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!   y - the value of the desired solution y of (1) 
!   yp - the value of the derivative of the desired solution y of (1)
!
!  Output parameters:
!   a1,a2 - the coefficients in the representation (3)
!
!

data pi /3.14159265358979323846264338327950288d0/

if (ifleft .eq. 1) then
apval  = alphap(1,1)
appval = alphapp(1,1)
else
apval  = alphap(k,nints)
appval = alphapp(k,nints)
endif


c1 = y*sqrt(apval)
c2 = yp/sqrt(apval)+y*appval/(2*apval**(1.5d0))

a1  = sqrt(c1**2+c2**2)
a2  = atan2(c1,c2)

if(a2 .gt. pi) then
a1 = -a1
a2 = a2-pi
endif

if(a2 .le. 0) then
a1 = -a1
a2 = a2 + pi
endif

end subroutine 


subroutine kummer_coefs2(nints,ab,k,xscheb,alpha,alphap,alphapp,a,ya,ypa,a1,a2)
implicit double precision (a-h,o-z)

integer, intent(in)              :: nints,k
double precision, intent(in)     :: xscheb(k),ab(2,nints),a,ya,ypa
double precision, intent(in)     :: alpha(k,nints),alphap(k,nints),alphapp(k,nints)
double precision, intent(out)    :: a1,a2

!
!  Construct the coefficients 
!
!
!  Input parameters:
!
!  Output parameters:
!
!

pi   = acos(-1.0d0)

call chebpw_eval(nints,ab,k,xscheb,alpha,a,aval)
call chebpw_eval(nints,ab,k,xscheb,alphap,a,apval)
call chebpw_eval(nints,ab,k,xscheb,alphapp,a,appval)

ua  = sin(aval)/sqrt(apval)
upa = cos(aval)*sqrt(apval)-sin(aval)*appval/(2*apval**(1.5d0))

va  = cos(aval)/sqrt(apval)
vpa = -sin(aval)*sqrt(apval)-cos(aval)*appval/(2*apval**(1.5d0))

det = ua*vpa-upa*va
c2  = (vpa*ya-va*ypa)/det
c1  = (ua*ypa-upa*ya)/det

a1  = sqrt(c1**2+c2**2)
a2  = atan2(c1,c2)

if(a2 .gt. pi) then
a1 = -a1
a2 = a2-pi
endif

if(a2 .le. 0) then
a1 = -a1
a2 = a2 + pi
endif

end subroutine 




subroutine kummer_zeros_count(nints,ab,k,xscheb,alpha,a1,a2,nroots)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: nints,k
double precision, intent(in)               :: ab(2,nints),xscheb(k),a1,a2
double precision, intent(in)               :: alpha(k,nints)
integer, intent(out)                       :: nroots

!
!  Return the number of zeros of a function of the form (3) in the interval
!  [a,b].
!
!  Input parameters:
!    (nints,ab) - the discretization scheme for representing the phase
!      function alpha
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!   alpha - a (k,nints) array specifying the values of alpha at the nodes
!     of the k-point Chebyshev grids on the intervals specified by ab
!   a1,a2 - the coefficients in the representation (3)
!
!  Output parameters:
!    nroots - the number of roots of y in the interval [a,b]
!

data pi /3.14159265358979323846264338327950288d0/

!
!  Count the number of roots on the interval.
!

a    = ab(1,1)
b    = ab(2,nints)
call chebpw_eval(nints,ab,k,xscheb,alpha,a,alpha0)
call chebpw_eval(nints,ab,k,xscheb,alpha,b,alpha1)

k1     = floor((alpha0+a2)/pi)
k2     = floor((alpha1+a2)/pi)
nroots = k2-k1

end subroutine 



subroutine kummer_zero(nintsinv,abinv,k,xscheb,alphainv,a1,a2,n,root)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: nintsinv,k,n
double precision, intent(in)               :: abinv(2,nintsinv),xscheb(k)
double precision, intent(in)               :: alphainv(k,nintsinv)
double precision, intent(out)              :: root

!
!  Return the nth root of the function y defined via (3) on the interval
!  [a,b].
!
!  Input parameters:
!    (nintsinv,abinv) - the discretization scheme for representing the inverse
!      of the phase function
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!   alphainv - a (k,nints) array specifying the values of the inverse of alpha
!     at the nodes of the k-point Chebyshev grids on the intervals specified by ab
!   a1,a2 - the coefficients in the representation (3)
!
!  Output parameters:
!    root - the value of the n^th root of y on the interval [a,b]
!
!

double precision :: pi
data pi /3.14159265358979323846264338327950288d0/

xx = pi*n-a2
call chebpw_eval(nintsinv,abinv,k,xscheb,alphainv,xx,root)
end subroutine 



subroutine kummer_zeroder(ifleft,k,xscheb,nintsinv,abinv,alphainv,nints,ab,alpha,alphap,&
   a1,a2,n,root,der)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: k,nintsinv,nints,ifleft
double precision, intent(in)               :: xscheb(k),abinv(k,nintsinv),ab(k,nints)
double precision, intent(in)               :: alphainv(k,nintsinv),alphap(k,nints)
double precision, intent(in)               :: alpha(k,nints)
double precision, intent(out)              :: root,der

!
!  Return the nth root of the function y defined via (3) on the interval
!  [a,b] and the value of the derivative of y at that point.
!
!  Input parameters:
!    (nints,ab) - the discretization scheme for representing the phase
!      function alpha
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!   alphainv - a (k,nints) array specifying the values of the inverse of alpha
!     at the nodes of the k-point Chebyshev grids on the intervals specified by ab
!   a1,a2 - the coefficients in the representation (3)
!
!  Output parameters:
!    root - the value of the n^th root of y on the interval [a,b]
!    der output the value 
!
!

double precision :: pi
data pi /3.14159265358979323846264338327950288d0/


if (ifleft .eq. 1) then
xx = pi*n-a2
else
xx = -pi*n+a2
endif

call chebpw_eval(nintsinv,abinv,k,xscheb,alphainv,xx,root)
call chebpw_eval(nints,ab,k,xscheb,alphap,root,apval)

der = (-1.0d0)**n * a1 * sqrt(apval)


end subroutine 



subroutine kummer_eval(nints,ab,k,xscheb,alpha,alphap,a1,a2,t,val)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nints,k
double precision, intent(in)  :: ab(2,nints),xscheb(k),a1,a2,t
double precision, intent(in)  :: alpha(k,nints),alphap(k,nints)
double precision, intent(out) :: val

!
!  Evaluate a function of the form (3) at a specified point in the inverval
!  [a,b].
!
!  Input parameters:
!    (nints,ab) - the discretization scheme for representing the phase
!      function alpha
!    k - the number of terms in the piecewise Chebyshev expansions used to
!      represent the solution
!    xscheb - the nodes of the k-point Chebyshev grid on the interval [-1,1]
!   alpha - a (k,nints) array specifying the values of alpha at the nodes
!     of the k-point Chebyshev grids on the intervals specified by ab
!   alphap - a (k,nints) array specifying the values of the derivative of
!     alpha at the nodes of the k-point Chebyshev grids on the intervals 
!     specified by ab
!   a1,a2 - the coefficients in the representation (3)
!
!  Output parameters:
!    val - the value of the function y at the point t
!

call chebpw_eval(nints,ab,k,xscheb,alpha,t,aval)
call chebpw_eval(nints,ab,k,xscheb,alphap,t,apval)

val = a1*sin(aval+a2)/sqrt(abs(apval))

end subroutine 

end module
