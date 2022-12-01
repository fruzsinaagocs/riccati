!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  This file contains code for solving quasilinear second ordinary ordinary differential
!  equations of the form
!
!     y''(t) = f(t,y(t),y'(t))                                                            (1)
!
!  on the interval [a,b] subject to either initial boundary conditions or terminal
!  boundary conditions (that is, the values of y and y' can be given either at the 
!  left-hand endpoint of the interval a or at the right-hand endpoint b).
!
!  Solution are represented via their values at the k-point Chebyshev grids
!  on a collection of subintervals of [a,b], where k is a user-specified
!  integer.
!
!  The following routines should be regarded as public:
!
!    ode_solve_ivp  - solve an initial value problem for an ordinary differential
!     equation of the form (1).  The solution is represented via its values on 
!     at the nodes of the Chebyshev grids on a collection of intervals specified 
!     by the user.
!
!    ode_solve_tvp  - solve a terminal value problem for an ordinary differential
!     equation of the form (1).  The solution is represented via its values on 
!     at the nodes of the Chebyshev grids on a collection of intervals specified 
!     by the user.
!
!    ode_solve_ivp_adap - solve an initial value problem for an ordinary differential
!     equation of the form (1).  This version of the solver adaptively discretizes
!     the solution.
!
!    ode_solve_tvp_adap - solve a terminal value problem for an ordinary differential
!     equation of the form (1); this version of the solver routine adaptively
!     discretizes the solution.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module odesolve
use utils
use chebyshev

interface


subroutine odefunction(t,y,yp,f,dfdy,dfdyp)
implicit double precision (a-h,o-z)
double precision, intent(in)  :: t,y,yp
double precision, intent(out) :: f,dfdy,dfdyp
!
!  Return the value of f(t,y,y') as well as the derivatives of f
!  with respect to  y and with respect to  y' at a specified point.
!
end subroutine

end interface

contains



subroutine ode_solve_ivp(ier,nints,ab,k,xscheb,chebintl,ys,yders,yder2s,odefun)
implicit double precision (a-h,o-z)

integer, intent(in)              :: nints
double precision, intent(in)     :: ab(2,nints),xscheb(k),chebintl(k,k)
double precision ,intent(out)    :: ys(k,nints), yders(k,nints), yder2s(k,nints)
procedure(odefunction)           :: odefun

!
!  Solve an initial value problem for a second order nonlinear ordinary differential
!  equation of the form
!
!    y''(t) = f(t,y'(t),y(t))                                                             (2)
! 
!  on an interval [a,b] using a Chebyshev collocation method.  This procedure takes
!  as input a partition of the interval [a,b].  On each subinterval, it uses
!  the implicit trapezoidal method procedure to construct a crude approximation 
!  of the solution and then refines it via Newton's method.  The linearized equation 
!  is solved via a Chebyshev spectral method and the restriction y of the solution to 
!  each of the subintervals of [a,b] is represented via its values at a k-point Chebyshev
!  grid.
!
!  Input parameters:
!
!    nints - the number of intervals into which [a,b] is subdivided
!    ab - a (2,nints) array each column of which specifies one of the subintervals
!    xscheb - the nodes of the k-point Chebyshev grid on [-1,1] as returned by
!     the subroutine chebexps (see below)
!    chebintl - the "left" Chebyshev spectral integration matrix as constructed
!      by the subroutine chebexps (see below)
!    odefun - a user-supplied external procedure of type "odefunction" which
!     supplied the values of the function f as well as the derivatives of
!     f w.r.t. y' and y  (see the definition of odefunction above)
!    ys(1,1) - the first entry of this array specifies the initial value of the
!     solution y of (2)
!    yders(1,1) - the first entry of this array specifies the initial value of the
!     derivative y' 
!
!  Output parameters:
!
!    ier - an error return code;
!       ier =  0    indicates successful execution
!       ier =  4    means that 
!
!    ys - the values of the solution y of (2) at the Chebyshev nodes
!    yders - the values of the solution y' of (2) at the Chebyshev nodes
!      Chebyshev grid on [a,b]
!    yder2s - the values of the solution y'' of (2) at the Chebyshev nodes
!

double precision, allocatable :: ts(:),ps(:),qs(:),fs(:),amatr(:,:)
double precision, allocatable :: hs(:),hders(:),hder2s(:)
double precision, allocatable :: ys0(:),yders0(:),yder2s0(:)

ier      = 0
maxiters = 25

!
!  Set algorithm parameters and allocate memory for the procedure.
!

allocate(ts(k),ps(k),qs(k),fs(k),amatr(k,k),hs(k),hders(k),hder2s(k))
allocate(ys0(k),yders0(k),yder2s0(k))

!
!  Traverse the intervals in increasing order.
!

do int = 1 ,nints

a  = ab(1,int)
b  = ab(2,int)
ts = (b+a)/2 + (b-a)/2 * xscheb

!
!  Copy the initial values to the beginning of the output arrays for the
!  interval
!

if (int .gt. 1) then
ys(1,int)    = ys(k,int-1)
yders(1,int) = yders(k,int-1)
endif

!
!  Use the implicit trapezoidal method to construct an appropriate initial
!  approximation.
!

call ode_trap_ivp(ier,k,ts,odefun,ys(:,int),yders(:,int),yder2s(:,int))


if (ier .ne. 0) then
ier = 1024
return
endif

!  The values of y, y' and y'' calculated using the implicit trapezoidal method satisfy
!  the differential equation at the specified nodes, but they are not consistent with eachother.  
!  That is, y' is not the derivative of y in any reasonable sense.
!
!  We integrate the second derivative twice in order to produce consistent
!  approximations.
!
!  Skipping this step is fatal.
!

yders(:,int)  = matmul(chebintl*(b-a)/2,yder2s(:,int)) + yders(1,int)
ys(:,int)     = matmul(chebintl*(b-a)/2,yders(:,int))  + ys(1,int)

!
!  Form the coefficients for the linearized problem and compute
!  the error in the current solution.
!

do i=1,k
call odefun(ts(i),ys(i,int),yders(i,int),f,dfdy,dfdyp)
ps(i)    = -dfdyp
qs(i)    = -dfdy
fs(i)    = f-yder2s(i,int)
end do

dd0 = norm2(fs)

!
!  Perform Newton iterations.
!

do inewt = 1, maxiters


!
!  Call the ode_linear_ivp routine to solve the linearized system.
!

hs(1)    = 0
hders(1) = 0
call ode_linear_ivp(a,b,k,xscheb,chebintl,ps,qs,fs,hs,hders,hder2s)

!
!  Check the error in the newly obtained solution
!

ys0     = ys(:,int) + hs
yders0  = yders(:,int) + hders
yder2s0 = yder2s(:,int) + hder2s

do i=1,k
call odefun(ts(i),ys0(i),yders0(i),f,dfdy,dfdyp)
ps(i)    = -dfdyp
qs(i)    = -dfdy
fs(i)    = f-yder2s0(i)
end do
dd1 = norm2(fs)

!
!  Accept the new solution if the error has decreased.
!


if (abs(dd0) .le. abs(dd1)) exit

dd0           = dd1
ys(:,int)     = ys0
yders(:,int)  = yders0
yder2s(:,int) = yder2s0


end do

end do

end subroutine



subroutine ode_solve_tvp(ier,nints,ab,k,xscheb,chebintr,ys,yders,yder2s,odefun)
implicit double precision (a-h,o-z)

integer, intent(in)              :: nints
double precision, intent(in)     :: ab(2,nints),xscheb(k),chebintr(k,k)
double precision ,intent(out)    :: ys(k,nints), yders(k,nints), yder2s(k,nints)
procedure(odefunction)           :: odefun

!
!  Solve a terminal value problem for a second order nonlinear ordinary differential
!  equation of the form
!
!    y''(t) = f(t,y'(t),y(t))                                                             (3)
! 
!  on an interval [a,b] using a Chebyshev spectral method.  This procedure takes
!  as input a partition of the interval [a,b].  One each subinterval, it uses
!  the implicit trapezoidal method to construct a crude approximation of the solution
!  and then refines it via Newton's method.  The linearized equation is solved via
!  a Chebyshev spectral method and the restriction y of the solution to each of
!  the subintervals of [a,b] is represented via its values at a k-point Chebyshev
!  grid.
!
!  Input parameters:
!
!    nints - the number of intervals into which [a,b] is subdivided
!    ab - a (2,nints) array each column of which specifies one of the subintervals
!    xscheb - the nodes of the k-point Chebyshev grid on [-1,1] as returned by
!     the subroutine chebexps (see below)
!    chebintr - the "right" Chebyshev spectral integration matrix as constructed
!      by the subroutine chebexps (see below)
!    odefun - a user-supplied external procedure of type "odefunction" which
!     supplied the values of the function f as well as the derivatives of
!     f w.r.t. y' and y  (see the definition of odefunction above)
!
!    ys(k,nints) - the last entry of this array specifies the terminal value of the
!     solution y of (3)
!    yders(k,nints) - last first entry of this array specifies the initial value of the
!     derivative y' 
!
!  Output parameters:
!
!    ys - the values of the solution y of (3) at the Chebyshev nodes
!    yders - the values of the solution y' of (3) at the Chebyshev nodes
!      Chebyshev grid on [a,b]
!    yder2s - the values of the solution y'' of (3) at the Chebyshev nodes
!

double precision, allocatable :: ts(:),ps(:),qs(:),fs(:),amatr(:,:)
double precision, allocatable :: hs(:),hders(:),hder2s(:)
double precision, allocatable :: ys0(:),yders0(:),yder2s0(:)

!
!  Set algorithm parameters and allocate memory for the procedure.
!

maxiters = 25

allocate(ts(k),ps(k),qs(k),fs(k),amatr(k,k),hs(k),hders(k),hder2s(k))
allocate(ys0(k),yders0(k),yder2s0(k))

!
!  Traverse the intervals in decreasing order.
!

do int = nints,1, -1

a  = ab(1,int)
b  = ab(2,int)
ts = (b+a)/2 + (b-a)/2 * xscheb

!
!  Copy the initial values to the beginning of the output arrays for the
!  interval
!

if (int .lt. nints) then
ys0(k)    = ys(1,int+1)
yders0(k) = yders(1,int+1)
else
ys0(k)    = ys(k,nints)
yders0(k) = yders(k,nints)
endif

!
!  Use the implicit trapezoidal method to construct an appropriate initial
!  approximation.
!

call ode_trap_tvp(ier,k,ts,odefun,ys0,yders0,yder2s0)

if (ier .ne. 0) then
ier = 1024
return
endif

!  The values of y, y' and y'' calculated by the implicit trapezoidal method satisfies
!  the differential equation at the specified nodes, but they are not consistent with eachother.  
!  That is, y' is not the derivative of y in any reasonable sense.
!
!  We integrate the second derivative twice in order to produce consistent
!  approximations.
!
!  Skipping this step is fatal.
!
yders0  = matmul(chebintr*(b-a)/2,yder2s0) + yders0(k)
ys0     = matmul(chebintr*(b-a)/2,yders0)  + ys0(k)


!
!  Perform Newton iterations.
!

dd0 = 1d300
do inewt = 1, maxiters

!
!  Form the coefficients and right-hand side of the linearized equation, which is:
!
!  h''(t) - D_{y'} f(t,y(t),y'(t)) h'(t) - D_{y} f(t,y(t),y'(t)) h(t) = f(t,y(t),y'(t)) - y''(t)
!
!  Note that the right-hand side is nothing more than the residual.
!  

do i=1,k
call odefun(ts(i),ys0(i),yders0(i),f,dfdy,dfdyp)
ps(i)    = -dfdyp
qs(i)    = -dfdy
fs(i)    = f-yder2s0(i)
end do

dd1 = norm2(fs)

if (isNaN(dd1)) exit
if( dd0 .le. dd1) exit


dd0           = dd1
ys(:,int)     = ys0
yders(:,int)  = yders0
yder2s(:,int) = yder2s0

!
!  Solve the linearized system with 0 boundary conditions
!

hs(k)    = 0
hders(k) = 0
call ode_linear_tvp(a,b,k,xscheb,chebintr,ps,qs,fs,hs,hders,hder2s)

!
!  Update the solution.
!

ys0      = ys(:,int) + hs
yders0   = yders(:,int) + hders
yder2s0  = yder2s(:,int) + hder2s

 
end do

! if (dd0 .eq. 1d300) then
! ier = 4
! return
! endif

end do


end subroutine



subroutine ode_solve_ivp_adap(ier,eps,nintsin,abin,k,xscheb,chebintl,ucheb,nints,ab,  &
   ys,yders,yder2s,odefun, ya,ypa)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: nintsin
double precision, intent(in)               :: abin(2,nintsin)
double precision, intent(in)               :: xscheb(k),ucheb(k,k),chebintl(k,k)
procedure(odefunction)                     :: odefun
integer, intent(out)                       :: nints
double precision, allocatable, intent(out) :: ab(:,:), ys(:,:),yders(:,:),yder2s(:,:)
double precision, intent(in)               :: ya,ypa

!
!  Solve an initial value problem for a second order nonlinear ordinary differential
!  equation of the form
!
!    y''(t) = f(t,y'(t),y(t))                                                             (2)
! 
!  on an interval [a,b] using a Chebyshev spectral method.  This procedure takes
!  as input a partition of the interval [a,b].  On each subinterval, it uses
!  the implicit trapezoidal method to construct a crude approximation of the solution
!  and then refines it via Newton's method.  The linearized equation is solved via
!  a Chebyshev spectral method and the restriction y of the solution to each of
!  the subintervals of [a,b] is represented via its values at a k-point Chebyshev
!  grid.
!
!  Input parameters:
!
!    nintsin - the number of intervals into which [a,b] is initially subdivided
!    abin - a (2,nints) array each column of which specifies one of the subintervals
!      into which (a,b) is initially divided
!    xscheb - the nodes of the k-point Chebyshev grid on [-1,1] as returned by
!     the subroutine chebexps (see below)
!    chebintl - the "left" Chebyshev spectral integration matrix as constructed
!      by the subroutine chebexps (see below)
!    odefun - a user-supplied external procedure of type "odefunction" which
!     supplied the values of the function f as well as the derivatives of
!     f w.r.t. y' and y  (see the definition of odefunction above)
!    ya - the initial value of the solution y of (2)
!    ypa - the initial value of the solution yp of (2)
!
!  Output parameters:
!    ier - an error return code; 
!      ier = 0      indicates successful execution
!      ier = 4      means that the maximum number of intervals was exceeded
!      ier = 1024   means that the maximum number of recursion steps was
!                   exceeded
!      ier = 2048   the number of input intervals exceeded the preset
!                   maximum number of intervals
!
!    ys - the values of the solution y of (2) at the Chebyshev nodes
!    yders - the values of the solution y' of (2) at the Chebyshev nodes
!      Chebyshev grid on [a,b]
!    yder2s - the values of the solution y'' of (2) at the Chebyshev nodes
!

double precision, allocatable :: ab0(:,:),ys0(:),yders0(:),yder2s0(:),coefs0(:)
double precision, allocatable :: about(:,:),ysout(:,:),ydersout(:,:),yder2sout(:,:)
double precision, allocatable :: ys1(:),yders1(:),yder2s1(:)

double precision, allocatable :: ts(:),ps(:),qs(:),fs(:),amatr(:,:)
double precision, allocatable :: hs(:),hders(:),hder2s(:)

ier = 0
pi  = acos(-1.0d0)
!
!  Set algorithm parameters and allocate memory for the procedure.
!

maxiters = 25
maxints  = 1000000

allocate(ts(k),ps(k),qs(k),fs(k),amatr(k,k),hs(k),hders(k),hder2s(k))
allocate(ab0(2,maxints))

allocate(about(2,maxints),ysout(k,maxints),ydersout(k,maxints),yder2sout(k,maxints))
allocate(ys0(k),yders0(k),yder2s0(k),coefs0(k))
allocate(ys1(k),yders1(k),yder2s1(k))


if (nintsin .gt. maxints) then
ier = 2048
return
endif

!
!  Initialize the list of intervals which need to be processed.
!

nintsout        = 0
nints0          = nintsin

do i =1,nints0
ab0(:,i) = abin(:,nints0-i+1)
enddo


do while (nints0 > 0) 

ifsplit = 0
a0      = ab0(1,nints0)
b0      = ab0(2,nints0)
nints0  = nints0 - 1
ts      = (b0+a0)/2 + (b0-a0)/2 * xscheb

if (abs(b0-a0) .eq. 0) then
ier = 1024
return
endif

!
!  Use the implicit trapezoidal method to construct an approximation
!  to use as an initial guess for Newton's method.
!

ys0 = 0

if (nintsout .eq. 0) then
ys1(1)    = ya
yders1(1) = ypa
else
ys1(1)    = ysout(k,nintsout)
yders1(1) = ydersout(k,nintsout)
endif


call ode_trap_ivp(ier,k,ts,odefun,ys1,yders1,yder2s1)

if (ier .ne. 0) then
ifsplit = 1
goto 1000
endif


!  The values of y, y' and y'' calculated by the implicit trapezoidal method satisfy 
!  the differential equation at the specified nodes, but they are not consistent with eachother.  
!  That is, y' is not the derivative of y in any reasonable sense.
!
!  We integrate the second derivative twice in order to produce consistent
!  approximations.
!
!  Skipping this step is fatal.
!

yders1  = matmul(chebintl*(b0-a0)/2,yder2s1) + yders1(1)
ys1     = matmul(chebintl*(b0-a0)/2,yders1)  + ys1(1)


dd0 = 1d300

!
!  Use Newton iterations to refine the solution as much as is practical.
! 

do inewt = 1, maxiters

!
!  Form the coefficients and right-hand side of the linearized equation, which is:
!
!  h''(t) - D_{y'} f(t,y(t),y'(t)) h'(t) - D_{y} f(t,y(t),y'(t)) h(t) = f(t,y(t),y'(t)) - y''(t)
!
!  Note that the right-hand side is nothing more than the residual.
!  


do i=1,k
call odefun(ts(i),ys1(i),yders1(i),f,dfdy,dfdyp)
ps(i)    = -dfdyp
qs(i)    = -dfdy
fs(i)    = f-yder2s1(i)
end do

!
!  Compare the accuracy of the proposed solution to that of the currently accepted
!  solution
!

dd1 = norm2(fs)

if (isNaN(dd1) .OR. dd0 .le. dd1) then
exit
endif

dd0     = dd1
yders0  = yders1
ys0     = ys1
yder2s0 = yder2s1


!
!  Call the ode_linear_ivp routine to solve the linear system.
!

hs(1)    = 0
hders(1) = 0
call ode_linear_ivp(a0,b0,k,xscheb,chebintl,ps,qs,fs,hs,hders,hder2s)

!
!  Form a new proposed solution
!

ys1     = ys0     + hs
yders1  = yders0  + hders
yder2s1 = yder2s0 + hder2s

if (dd1 .eq. 0) exit

end do

!
!  If we did not perform even one Newton iteration, split the interval.
!

! if (dd0 .eq. 1d300) then
! ifsplit = 1
! goto 1000
! endif

yders0  = yders1
ys0     = ys1
yder2s0 = yder2s1





!
!  Compute the Chebyshev expansion for the solution
!

coefs0 = matmul(ucheb,ys0)
coefs0 = coefs0/coefs0(1)


!  relative Maxmimum value of trailing coefficients

nn = k/2

! dd1   = sum(abs(coefs0(k-nn+1:k)))
dd1   = maxval(abs(coefs0(k-nn+1:k)))
! dd2   = maxval(abs(coefs0))+1
! dd    = dd1/dd2


! dd  = dd1*(b0-a0)

dd = dd1

if (dd .gt. eps) then
ifsplit = 1
endif

!
!  We jump here if the decision to split has already been made
!
1000 continue

!
!  If the interval does not need to be split, copy it to the output list and
!  copy the computed solution there as well.
!

if (ifsplit .eq. 0) then


if (nintsout+1 .ge. maxints) then
ier = 4
return
endif


nintsout                 = nintsout+1
about(1,nintsout)        = a0
about(2,nintsout)        = b0
ysout(1:k,nintsout)      = ys0
ydersout(1:k,nintsout)   = yders0
yder2sout(1:k,nintsout)  = yder2s0

else

if (nints0+2 .ge. maxints) then
ier = 4
return
endif

!
!  Otherwise update the list of intervals.
!

c0 = (a0+b0)/2
nints0 = nints0 + 1
ab0(1,nints0) = c0
ab0(2,nints0) = b0
nints0 = nints0 + 1
ab0(1,nints0) = a0
ab0(2,nints0) = c0
endif

end do

!
!  Copy the obtained solution out.
!

nints = nintsout
allocate(ab(2,nints))
allocate(ys(k,nints),yders(k,nints),yder2s(k,nints))

ab     = about(:,1:nints)
ys     = ysout(:,1:nints)
yders  = ydersout(:,1:nints)
yder2s = yder2sout(:,1:nints)


end subroutine



subroutine ode_solve_tvp_adap(ier,eps,nintsin,abin,k,xscheb,chebintr,ucheb,nints,ab,  &
   ys,yders,yder2s,odefun, yb,ypb)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: nintsin
double precision, intent(in)               :: abin(2,nintsin)
double precision, intent(in)               :: xscheb(k),ucheb(k,k),chebintr(k,k)
procedure(odefunction)                     :: odefun
integer, intent(out)                       :: nints
double precision, allocatable, intent(out) :: ab(:,:), ys(:,:),yders(:,:),yder2s(:,:)
double precision, intent(in)               :: yb, ypb

!
!  Solve a terminal value problem for a second order nonlinear ordinary differential
!  equation of the form
!
!    y''(t) = f(t,y'(t),y(t))                                                             (2)
! 
!  on an interval [a,b] using a Chebyshev spectral method.  This procedure takes
!  as input a partition of the interval [a,b].  On each subinterval, it uses
!  the implicit trapezoidal method to construct a crude approximation of the solution
!  and then refines it via Newton's method.  The linearized equation is solved via
!  a Chebyshev spectral method and the restriction y of the solution to each of
!  the subintervals of [a,b] is represented via its values at a k-point Chebyshev
!  grid.
!
!  Input parameters:
!
!    nintsin - the number of intervals into which [a,b] is initially subdivided
!    abin - a (2,nints) array each column of which specifies one of the subintervals
!    xscheb - the nodes of the k-point Chebyshev grid on [-1,1] as returned by
!     the subroutine chebexps (see below)
!    chebintl - the "left" Chebyshev spectral integration matrix as constructed
!      by the subroutine chebexps (see below)
!    odefun - a user-supplied external procedure of type "odefunction" which
!     supplied the values of the function f as well as the derivatives of
!     f w.r.t. y' and y  (see the definition of odefunction above)
!    yb - the terminal value of the solution y of (2)
!    ypb - the terminal value of the solution yp of (2)
!
!  Output parameters:
!
!    ier - an error return code; 
!      ier = 0      indicates successful execution
!      ier = 4      means that the maximum number of intervals was exceeded
!      ier = 1024   means that the maximum number of recursion steps was
!                   exceeded
!      ier = 2048   the number of input intervals exceeded the preset
!                   maximum number of intervals
!
!    ys - the values of the solution y of (2) at the Chebyshev nodes
!    yders - the values of the solution y' of (2) at the Chebyshev nodes
!      Chebyshev grid on [a,b]
!    yder2s - the values of the solution y'' of (2) at the Chebyshev nodes
!

double precision, allocatable :: ab0(:,:),ys0(:),yders0(:),yder2s0(:),coefs0(:)
double precision, allocatable :: ys1(:),yders1(:),yder2s1(:)
double precision, allocatable :: about(:,:),ysout(:,:),ydersout(:,:),yder2sout(:,:)


double precision, allocatable :: ts(:),ps(:),qs(:),fs(:),amatr(:,:)
double precision, allocatable :: hs(:),hders(:),hder2s(:)

ier = 0
pi  = acos(-1.0d0)

!
!  Set algorithm parameters and allocate memory for the procedure.
!

maxints  = 1000000
maxiters = 25

allocate(ts(k),ps(k),qs(k),fs(k),amatr(k,k),hs(k),hders(k),hder2s(k))
allocate(ab0(2,maxints))

allocate(about(2,maxints),ysout(k,maxints),ydersout(k,maxints),yder2sout(k,maxints))
allocate(ys0(k),yders0(k),yder2s0(k),coefs0(k))
allocate(ys1(k),yders1(k),yder2s1(k))

!
!  Initialize the list of intervals which need to be processed.
!


if (nintsin .gt. maxints) then
ier = 2048
return
endif


nintsout   = 0
nints0     = nintsin
do i=1,nints0
ab0(:,i) = abin(:,i)
end do


do while (nints0 > 0) 

ifsplit = 0
a0      = ab0(1,nints0)
b0      = ab0(2,nints0)
nints0  = nints0 - 1
ts      = (b0+a0)/2 + (b0-a0)/2 * xscheb

if (abs(b0 - a0) .eq. 0) then
ier = 1024
return
endif


!
!  Use the  implicit trapezoidal method procedure to construct an appropriate initial
!  approximation.
!


if (nintsout .eq. 0) then
ys1(k)    = yb
yders1(k) = ypb
else
ys1(k)    = ysout(1,nintsout)
yders1(k) = ydersout(1,nintsout)
endif


call ode_trap_tvp(ier,k,ts,odefun,ys1,yders1,yder2s1)
if (ier .ne. 0) then
ifsplit = 1
goto 1000
endif


!  The values of y, y' and y'' calculated by the implicit trapezoidal method satisfy 
!  the differential equation at the specified nodes, but they are not consistent with eachother.  
!  That is, y' is not the derivative of y in any reasonable sense.
!
!  We integrate the second derivative twice in order to produce consistent
!  approximations.
!
!  Skipping this step is fatal.
!

yders1  = matmul(chebintr*(b0-a0)/2,yder2s1) + yders1(k)
ys1     = matmul(chebintr*(b0-a0)/2,yders1)  + ys1(k)

!
!  Perform Newton iterations.
! 

dd0     = 1d300
do inewt = 1, maxiters


!
!  Form the coefficients and right-hand side of the linearized equation, which is:
!
!  h''(t) - D_{y'} f(t,y(t),y'(t)) h'(t) - D_{y} f(t,y(t),y'(t)) h(t) = f(t,y(t),y'(t)) - y''(t)
!
!  Note that the right-hand side is nothing more than the residual.
!  


do i=1,k
call odefun(ts(i),ys1(i),yders1(i),f,dfdy,dfdyp)
ps(i)    = -dfdyp
qs(i)    = -dfdy
fs(i)    = f-yder2s1(i)
end do

dd1 = norm2(fs)
if (isNaN(dd1) .OR.  dd0 .le. dd1) exit

dd0     = dd1
ys0     = ys1
yders0  = yders1
yder2s0 = yder2s1



!
!  Call the ode_linear_ivp routine to solve the linear system.
!


hs    = 0
hders = 0


call ode_linear_tvp(a0,b0,k,xscheb,chebintr,ps,qs,fs,hs,hders,hder2s)

!
!  Update the solution.
!

ys1     = ys0     + hs
yders1  = yders0  + hders
yder2s1 = yder2s0 + hder2s

end do

!
!  If we did not perform even one Newton iteration, split the interval.
!


!
!  If we did not perform even one Newton iteration, split the interval.
!

if (dd0 .eq. 1d300) then
ifsplit = 1
goto 1000
endif

!
!  Compute the Chebyshev expansion for the solution
!



coefs0 = matmul(ucheb,ys0)
coefs0 = coefs0/coefs0(1)


nn = k/2

dd   = maxval(abs(coefs0(k-nn+1:k)))
! dd2 = maxval(abs(coefs0))+1
! dd  = dd1/dd2


if (dd .gt. eps) then
ifsplit = 1
endif

1000 continue

!
!  If the interval does not need to be split, copy it to the output list and
!  copy the computed solution there as well.
!

if (ifsplit .eq. 0) then

if (nintsout+1 .ge. maxints) then
ier = 4
return
endif

nintsout                 = nintsout+1
about(1,nintsout)        = a0
about(2,nintsout)        = b0
ysout(1:k,nintsout)      = ys0
ydersout(1:k,nintsout)   = yders0
yder2sout(1:k,nintsout)  = yder2s0

else


!
!  Otherwise update the list of intervals.
!

if (nints0+2 .ge. maxints) then
ier = 4
return
endif

c0 = (a0+b0)/2
nints0 = nints0 + 1
ab0(1,nints0) = a0
ab0(2,nints0) = c0
nints0 = nints0 + 1
ab0(1,nints0) = c0
ab0(2,nints0) = b0
endif

end do


!
!  Copy the obtained solution out (in reverse order)
!

nints = nintsout
allocate(ab(2,nints))
allocate(ys(k,nints),yders(k,nints),yder2s(k,nints))

do i=1,nints
j             = nints-i+1
ab(:,i)       = about(:,j)
ys(1:k,i)     = ysout(:,j)
yders(1:k,i)  = ydersout(:,j)
yder2s(1:k,i) = yder2sout(:,j)
end do



end subroutine



subroutine ode_linear_ivp(a,b,k,xscheb,chebintl,ps,qs,fs,ys,yders,yder2s)
implicit double precision (a-h,o-z)

integer, intent(in)           :: k
double precision, intent(in)  :: xscheb(k),chebintl(k,k)
double precision, intent(in)  :: ps(k),qs(k),fs(k)
double precision, intent(out) :: ys(k),yders(k),yder2s(k)

!
!  Solve an initial value for the ordinary differential equation
!
!     y''(t) + p(t) y'(t) + q(t) y(t) = f(t)                                              (3)
!
!  on the interval [a,b] using a standard Chebyshev spectral method.
!
!  Input parameters:
!
!    (a,b) - the interval on which the ODE (3) is given
!    k - the number of Chebyshev points on the interval [a,b]
!    xscheb - the nodes of the k-point Chebyshev grid on [-1,1] as returned by
!     the subroutine chebexps (see below)
!    chebintl - the "left" Chebyshev spectral integration matrix as constructed
!      by the subroutine chebexps (see below)
!    ps - an array specifying the values of the function p(t) appearing in (3)
!      at the k Chebyshev nodes on [a,b]
!    qs - an array specifying the values of the function q(t) appearing in (3)
!      at the k Chebyshev node on [a,b]
!    fs - an array speciying the values of the function f(t) appearing in (3)
!      at the k Chebyshev nodes on [a,b]
!
!    ys(1) - the value of y(a)
!    yders(1) - the value of y'(a)
!
!  Output parameters:
!
!    ys - the values of the solution y of (3) at the nodes of the k-point
!      Chebyshev grid on [a,b]
!    yders - the values of the solution y' of (3) at the nodes of the k-point
!      Chebyshev grid on [a,b]
!    yder2s - the values of the solution y'' of (3) at the nodes of the k-point
!      Chebyshev grid on [a,b]
!

double precision, allocatable :: amatr(:,:),xs(:),chebint2(:,:),sigma(:),rhs(:)

!
!  Allocate memory for the procedure and setup some parameters.
!

allocate(amatr(k,k),xs(k),chebint2(k,k),sigma(k),rhs(k))


chebint2 = matmul(chebintl,chebintl)
xs       = (b-a)/2 *xscheb + (b+a)/2 
alpha    = ys(1)
beta     = yders(1)

!
!  We represent the solution in the form
!
!      y(t) = alpha + beta (t-a) + \int_a^t (t-s) sigma(s) ds,
!
!  insert this representation into (1), and solve the resulting system of
!  linear equations in order to obtain the values of sigma.  
!    

amatr  = 0
do i=1,k
amatr(i,i) = 1.0d0
sigma(i)   = 0.0d0
end do

!
! Handle the p(t) * y'(t) term.
!
do i=1,k
amatr(i,:) = amatr(i,:) + ps(i) * chebintl(i,:)*(b-a)/2
sigma(i)   = sigma(i) - ps(i)*beta
end do

!
!  Handle the q(t) y(t) term
!

do i=1,k
amatr(i,:) = amatr(i,:) + qs(i) * chebint2(i,:)*((b-a)/2)**2
sigma(i)   = sigma(i) - qs(i) * (alpha + beta*(xs(i)-a))
end do


!
!  Form the right-hand side.
!
do i=1,k
sigma(i) = sigma(i) + fs(i)
end do

!
!  Use a QR decomposition to invert the linear system
!

 call qrsolv(amatr,k,sigma)
!call gesv(amatr,sigma)

!
!  Calculate y(t) and y'(t) from sigma.
!

yder2s = sigma
yders  = (b-a)/2*matmul(chebintl,sigma)
ys     = ((b-a)/2)**2*matmul(chebint2,sigma)

do i=1,k
ys(i)     = ys(i) + alpha + beta*(xs(i)-a)
yders(i)  = yders(i) + beta
end do

end subroutine



subroutine ode_linear_tvp(a,b,k,xscheb,chebintr,ps,qs,fs,ys,yders,yder2s)
implicit double precision (a-h,o-z)

integer, intent(in)           :: k
double precision, intent(in)  :: xscheb(k),chebintr(k,k)
double precision, intent(in)  :: ps(k),qs(k),fs(k)
double precision, intent(out) :: ys(k),yders(k),yder2s(k)

!
!  Solve a terminal value for the ordinary differential equation
!
!     y''(t) + p(t) y'(t) + q(t) y(t) = f(t)                                              (4)
!
!  on the interval [a,b] using a standard Chebyshev spectral method.
!
!  Input parameters:
!
!    (a,b) - the interval on which the ODE (4) is given
!    k - the number of Chebyshev points on the interval [a,b]
!    xscheb - the nodes of the k-point Chebyshev grid on [-1,1] as returned by
!     the subroutine chebexps (see below)
!    chebintr - the "right" Chebyshev spectral integration matrix as
!      constructed by the subroutine chebexps (below)
!    ps - an array specifying the values of the function p(t) appearing in (4)
!      at the k Chebyshev nodes on [a,b]
!    qs - an array specifying the values of the function q(t) appearing in (4)
!      at the k Chebyshev node on [a,b]
!    fs - an array speciying the values of the function f(t) appearing in (4)
!      at the k Chebyshev nodes on [a,b]
!
!    ys(k) - the value of y(b)
!    yders(k) - the value of y'(b)
!
!  Output parameters:
!
!    ys - the values of the solution y of (4) at the nodes of the k-point
!      Chebyshev grid on [a,b]
!    yders - the values of the solution y' of (4) at the nodes of the k-point
!      Chebyshev grid on [a,b]
!    yder2s - the values of the solution y'' of (4) at the nodes of the k-point
!      Chebyshev grid on [a,b]
!

double precision, allocatable :: amatr(:,:),xs(:),chebint2(:,:),sigma(:)

!
!  Allocate memory for the procedure and setup some parameters.
!

allocate(amatr(k,k),xs(k),chebint2(k,k),sigma(k))

chebint2 = matmul(chebintr,chebintr)
xs       = (b-a)/2 *xscheb + (b+a)/2 
alpha    = ys(k)
beta     = yders(k)

!
!  We represent the solution in the form
!
!      y(t) = alpha + beta (t-b) + \int_b^t (t-s) sigma(s) ds,
!
!  insert this representation into (2), and solve the resulting system of
!  linear equations in order to obtain the values of sigma.  
!    

amatr  = 0
do i=1,k
amatr(i,i) = 1.0d0
sigma(i)   = 0.0d0
end do

!
! Handle the p(t) * y'(t) term.
!
do i=1,k
amatr(i,:) = amatr(i,:) + ps(i) * chebintr(i,:)*(b-a)/2
sigma(i)   = sigma(i) - ps(i)*beta
end do

!
!  Handle the q(t) y(t) term
!

do i=1,k
amatr(i,:) = amatr(i,:) + qs(i) * chebint2(i,:)*((b-a)/2)**2
sigma(i)   = sigma(i) - qs(i) * (alpha + beta*(xs(i)-b))
end do


!
!  Form the right-hand side.
!
do i=1,k
sigma(i) = sigma(i) + fs(i)
end do

!
!  Use a QR decomposition to invert the linear system
!

 call qrsolv(amatr,k,sigma)
!call gesv(amatr,sigma)

!
!  Calculate y(t) and y'(t) from sigma.
!

yder2s = sigma
yders  = (b-a)/2*matmul(chebintr,sigma)
ys     = ((b-a)/2)**2*matmul(chebint2,sigma)

do i=1,k
ys(i)     = ys(i) + alpha + beta*(xs(i)-b)
yders(i)  = yders(i) + beta
end do

end subroutine



subroutine ode_trap_ivp(ier,k,ts,odefun,ys,yders,yder2s)
implicit double precision (a-h,o-z)

integer, intent(out)          :: ier
integer, intent(in)           :: k
double precision, intent(in)  :: ts(k)
double precision, intent(out) :: ys(k),yders(k),yder2s(k)
procedure (odefunction)       :: odefun

!
!  Use the implicit trapezodial method to crudely approximate the solution of an
!  initial value problem for the nonlinear ordinary differential equation
!
!    y''(t) = f(t,y(t),y'(t))                                                             (5)
! 
!  on the interval [a,b].  The user specifies the nodes on which the solution
!  of is to be computed.
!
!  Input parameters:
!
!    k - the number of nodes at which the solution of (5) is to be computed
!    ts - an array of length k which supplies the a sorted list of nodes at which 
!     (4) is to be solved
!    odefun - a user-supplied external procedure of type "odefunction" which
!     supplied the values of the function f as well as the derivatives of
!     f w.r.t. y' and y  (see the definition of odefunction above)
!
!    ys(1) - the first entry of this array is the initial value for the solution y
!    yders(1) - the first entry of this array is the initial value for the solution y'
!
!  Output parameters:
!
!    ys - the values of the obtained approximation of the solution of (5) at
!     the specified nodes
!    yders - the values of the obtained approximation of the derivative of the solution of 
!     (5) at the specified nodes
!    yder2s - the values of the obtained approximation of the second derivative of the solution 
!     of  (5) at the specified nodes
!  

ier      = 0
maxiters = 25

!
!  Evaluate the second derivative at the left-hand endpoint of the interval.
!

call odefun(ts(1),ys(1),yders(1),yder2s(1),dfdy,dfdyp)


do i=2,k
t0 = ts(i-1)
t  = ts(i)
h  = t-t0

y0   = ys(i-1)
yp0  = yders(i-1)
ypp0 = yder2s(i-1)


!
!  Set the initial guess.
!

! yp1 = yp0 + h *ypp0
yp1 = yp0 
y1  = y0 + h/2*(yp0+yp1)
call odefun(t,y1,yp1,ypp1,dfdy,dfdyp)
dd    = yp1 - yp0 - h/2*(ypp0+ypp1)

!
!  Conduct Newton iterations in an attempt to improve the siutation.
!

do iter=1,maxiters

!
!  Record the current approximation.
!
dd0       = dd
ys(i)     = y1
yders(i)  = yp1
yder2s(i) = ypp1

!
!  Take a Newton step.
!

val   = dd
der   = 1.0d0-h/2*(dfdy*h/2+dfdyp)
delta = val/der

yp1   = yp1-delta
y1    = y0 + h/2*(yp0+yp1)
call odefun(t,y1,yp1,ypp1,dfdy,dfdyp)
dd    = yp1 - yp0 - h/2*(ypp0+ypp1)

if (abs(dd) .gt. abs(dd0))  exit

end do

ys(i)     = y1
yders(i)  = yp1
yder2s(i) = ypp1

end do

end subroutine



subroutine ode_trap_tvp(ier,k,ts,odefun,ys,yders,yder2s)
implicit double precision (a-h,o-z)

integer, intent(out)          :: ier
integer, intent(in)           :: k
double precision, intent(in)  :: ts(k)
double precision, intent(out) :: ys(k),yders(k),yder2s(k)
procedure (odefunction)       :: odefun
!
!  Use the implicit trapezoidal method to crudely approximate the solution of a
!  terminal value problem for the nonlinear ordinary differential equation
!
!    y''(t) = f(t,y(t),y'(t))                                                             (5)
! 
!  on the interval [a,b].  The user specifies the nodes on which the solution
!  of is to be computed.
!
!  Input parameters:
!
!    k - the number of nodes at which the solution of (5) is to be computed
!    ts - an array of length k which supplies the a sorted list of nodes at which 
!     (5) is to be solved
!    odefun - a user-supplied external procedure of type "odefunction" which
!     supplied the values of the function f as well as the derivatives of
!     f w.r.t. y' and y  (see the definition of odefunction above)
!
!    ys(k) - the last entry of this array is the terminal value for the solution y
!    yders(k) - the first entry of this array is the terminal value for the solution y'
!
!  Output parameters:
!
!    ier - an error return code;
!       ier = 0  indicates successful execution
!
!    ys - the values of the obtained approximation of the solution of (5) at
!     the specified nodes
!    yders - the values of the obtained approximation of the derivative of the solution of 
!     (5) at the specified nodes
!    yder2s - the values of the obtained approximation of the second derivative of the solution 
!     of  (5) at the specified nodes
!  

ier        = 0
maxiters   = 25

call odefun(ts(k),ys(k),yders(k),yder2s(k),dfdy,dfdyp)

do i=k-1,1,-1
t  = ts(i)
h  = ts(i+1)-ts(i)

y1   = ys(i+1)
yp1  = yders(i+1)
ypp1 = yder2s(i+1)

!
!  Set the initial guess.
!

yp0 = yp1
! yp0 = yp1 - ypp1*h
y0  = y1 - h/2*(yp0+yp1)

call odefun(t,y0,yp0,ypp0,dfdy,dfdyp)
dd  = yp1 - yp0 - h/2*(ypp0+ypp1)

!
!  Try to improve it via Newton iterations
!

do iter=1,maxiters

!
!  Record the current guess
!

dd0       = dd
ys(i)     = y0
yders(i)  = yp0
yder2s(i) = ypp0

!
!  Take a Newton step
!
val   = dd
der   = -1.0d0-h/2*(-dfdy*h/2+dfdyp)
delta = val/der

yp0   = yp0 -delta
y0    = y1 - h/2*(yp0+yp1)
call odefun(t,y0,yp0,ypp0,dfdy,dfdyp)
dd  = yp1 - yp0 - h/2*(ypp0+ypp1)

if (abs(dd) .gt. abs(dd0) .OR. abs(dd) .gt. abs(dd0)) exit

end do
end do

end subroutine


end module
