module test_odesolve_routines

use utils
use chebyshev
use odesolve

double precision :: gamma

contains

subroutine leges(n,x,pols,ders,ders2)
implicit double precision (a-h,o-z)

double precision, intent(out) :: pols(n+1),ders(n+1),ders2(n+1)
double precision x
integer n
!
!  Evaluate the n+1 Legendre polynomials of degree 0 through n at the
!  point x using the standard 3-term recurrence relation.  Return the values
!  of their first and second derivatives at the point x as well.
!
!  Input parameters:
!
!    n - an integer specifying the order of polynomials which are to
!      be evaluated
!    x - the point at which the polynomials and their derivatives
!
!  Output parameters:
!
!    pols - the ith entry of this user-supplied array of length n+1
!      will contain the value of  Legendre polynomial of degree
!      i-1 at x
!    ders - the ith entry of this user-supplied array will contain the
!      value of the derivative of the  Legendre polynomial
!      of degree i-1 at x
!    ders2 - the ith entry of this user-supplied array will contain
!      the value of the second derivative of the Legendre
!      polynomial of degree i-1 at x
!
!

eps0 = 1.0d-14

!
!  Handle the boundary case ...
!

if (x <= -1.0d0+eps0) then

pols(1) = 1
ders(1) = 0

do j=0,n

dd        = sqrt((2*j+1)/2.0d0)
pols(j+1) = dd * (-1.0d0)**j
ders(j+1) = (j*(1.0d0+j)/2)*dd*(-1.0d0)**(j+1)
ders2(j+1) = j/8.0d0*(j**3+2*j**2-j-2.0d0) *dd *(-1.0d0)**j
end do

return
endif


if (x >=  1-eps0) then
do j=0,n
dd      = sqrt((2*j+1)/2.0d0)
pols(j+1) = dd
ders(j+1) = (j*(1.0d0+j)/2)*dd
ders2(j+1) = j/8.0d0*(j**3+2*j**2-j-2.0d0) *dd
end do

return
endif

!
!  Otherwise ...
!

pols(1)  = 1
ders(1)  = 0
ders2(1) = 0

if (n >= 1) then
pols(2)  = x
ders(2)  = 1
ders2(2) = 0
end if

!
!  Calculate the values of the unnormalized polynomials
!
do j=2,n
   pols(j+1)=((2.0d0*j-1.0d0)*x*pols(j)-(j-1.0d0)*pols(j-1))/j
end do

!
!  Compute the derivatives of the unnormalized polynoials
!
d=x**2.0d0-1.0d0
do j=3,n+1
   ders(j)=(j-1.0d0)*(x*pols(j)-pols(j-1))/d
end do

!
!  Compute the second derivatives of the unnormalized polynomials
!
do j=3,n+1
  ders2(j) = (j-1)/d * (pols(j)+ ( (j-1)*x-2*x) /(j-1) * ders(j) - ders(j-1))
end do

end subroutine


subroutine ode1(t,y,yp,f,dfdy,dfdyp)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: t,y,yp
double precision, intent(out) :: f,dfdy,dfdyp

f     =  -y**4-yp**2+exp(t)+exp(2*t)+exp(4*t)
dfdy  =  -4*y**2
dfdyp =  -2*yp

end subroutine


subroutine ode2(t,y,yp,f,dfdy,dfdyp)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: t,y,yp
double precision, intent(out) :: f,dfdy,dfdyp

qval = 1 + cos(t)**2
phi  = (erf(12*(t-0.5d0))+1.0d0)/2
qval = (1-phi) + phi*qval

f     = -2*y**3 + 2*gamma**2 * qval * y + 1.5d0 * (yp)**2/y
dfdy  = -6*y**2 - 3*yp**2/(2*y**2) + 2*gamma**2 * qval
dfdyp = 3*yp/y

end subroutine


subroutine ode3(t,y,yp,f,dfdy,dfdyp)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: t,y,yp
double precision, intent(out) :: f,dfdy,dfdyp

qval = 1 + cos(t)**2-sin(t)/2
phi  = (erf(12*(t-1.5d0))+1.0d0)/2
qval = (phi) + (1-phi) * qval

f     = -2*y**3 + 2*gamma**2 * qval * y + 1.5d0 * (yp)**2/y
dfdy  = -6*y**2 - 3*yp**2/(2*y**2) + 2*gamma**2 * qval
dfdyp = 3*yp/y

end subroutine


subroutine test_nonlinear_adap()
implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),ucheb(:,:),vcheb(:,:), &
   chebintl(:,:),chebintr(:,:)

double precision, allocatable :: ab(:,:),ys(:,:),yders(:,:),yder2s(:,:),abin(:,:)


!
!  Fetch the Chebyshev quadrature and associated matrices.
!

k  = 30
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)

!
!  Setup the problem
!

gamma = 10**7

allocate(abin(2,2))
nintsin   = 1
abin(1,1) = 0.0d0
abin(2,1) = 2.0d0

call mach_zero(eps0)
eps         = 1.0d-14

ya          = gamma
ypa         = 0

call ode_solve_ivp_adap(ier,eps,nintsin,abin,k,xscheb,chebintl,ucheb,nints,ab,&
   ys,yders,yder2s,ode2,ya,ypa)


if (ier .ne. 0) then
call prini("after ode_solve_ivp_adap, ier = ",ier)
stop
endif

call prin2("after ode_solve_ivp_adap, ab = ",ab)
call prini("nints = ",nints)

call chebpw_plot("alpha*",1,nints,ab,k,xscheb,ys)
call chebpw_plot("alpha'*",2,nints,ab,k,xscheb,yders)


deallocate(ab,ys,yders,yder2s)

!
!  Test the terminal value version of the routine
!

yb          = gamma
ypb         = 0

call ode_solve_tvp_adap(ier,eps,nintsin,abin,k,xscheb,chebintr,ucheb,nints,ab,&
   ys,yders,yder2s,ode3,yb,ypb)

if (ier .ne. 0) then
call prini("after ode_solve_tvp_adap, ier = ",ier)
stop
endif

call prin2("after ode_solve_tvp_adap, ab = ",ab)


call chebpw_plot("alpha*",3,nints,ab,k,xscheb,ys)
call chebpw_plot("alpha'*",4,nints,ab,k,xscheb,yders)

end subroutine


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  Test the full nonlinear ODE solver
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine test_nonlinear()
implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),ucheb(:,:),vcheb(:,:), &
   chebintl(:,:),chebintr(:,:)

double precision, allocatable :: ab(:,:),xs(:,:),ys(:,:),yders(:,:),yder2s(:,:)


!
!  Fetch the Chebyshev quadrature and associated matrices.
!

k  = 30
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)

!
!  Form the list of intervals.
!
nints = 10
a     = 0.0d0
b     = 1.0d0

allocate(ab(2,nints))

dd = (b-a)/nints
do i=1,nints
ab(1,i) = a + dd * (i-1) 
ab(2,i) = a + dd * (i) 
end do

call prin2("before ode_solve_ivp, ab = ",ab)

!
!  Form the list of all the discretization nodes for plotting.
!
allocate(xs(k,nints))

do int = 1,nints
a0 = ab(1,int)
b0 = ab(2,int)

do i = 1, k
xs(i,int) = (b0+a0)/2 +  (b0-a0)/2 * xscheb(i)
end do
end do

!
!  Set the initial values and solve the problem.
!

allocate(ys(k,nints),yders(k,nints),yder2s(k,nints))

ys(1,1)     = 1.0d0
yders(1,1)  = 1.0d0

call elapsed(t1)
call ode_solve_ivp(ier,nints,ab,k,xscheb,chebintl,ys,yders,yder2s,ode1)
call elapsed(t2)

if (ier .ne. 0) then

call prini("after ode_solve_ivp, ier = ",ier)
stop
endif

!  Measure the error 
!

dd1 = maxval(abs(ys - exp(xs)))
dd2 = maxval(abs(yders - exp(xs)))
dd3 = maxval(abs(yder2s - exp(xs)))
dd1 = max(dd1,dd2,dd3)

call prina("")
call prin2("ode_solve_ivp error = ",dd1)


!
!  Test the terminal value version of the routine
!

ys     = 0
yders  = 0
yder2s = 0

ys(k,nints)     = exp(b)
yders(k,nints)  = exp(b)

call ode_solve_tvp(ier,nints,ab,k,xscheb,chebintr,ys,yders,yder2s,ode1)
if (ier .ne. 0) then
call prini("after ode_solve_ivp, ier = ",ier)
stop
endif


!
!  Measure the error 
!

dd1 = maxval(abs(ys - exp(xs)))
dd2 = maxval(abs(yders - exp(xs)))
dd3 = maxval(abs(yder2s - exp(xs)))
dd1 = max(dd1,dd2,dd3)
call prin2("ode_solve_tvp error = ",dd1)
call prina("")


end subroutine


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  Test the routines for solving linear ordinary differential equations
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

subroutine test_linear()
implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),ucheb(:,:),vcheb(:,:), &
   chebintl(:,:),chebintr(:,:)

double precision, allocatable :: xs(:),ps(:),qs(:),fs(:),ys(:),yders(:),yder2s(:),pols(:)
double precision, allocatable :: polders(:),polder2s(:)
!
!  Construct the k-point Chebyshev quadrature and the associated matrices.
!

k  = 30
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  Construct the Legendre polynomial of order 10 using   ode_linear_ivp
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

allocate(xs(k))

a = 0.0d0
b = 0.5d0

xs = (b+a)/2 + (b-a)/2*xscheb

!
!  Form the coefficients of the ODE and plot them.
!

allocate (ps(k),qs(k),fs(k),ys(k),yders(k),yder2s(k))

ys(1)    = -63.0d0/256.0d0
yders(1) = 0.0d0

ps = -2*xs/(1-xs**2)
qs = 110.0d0/(1-xs**2)
fs = 0

call ode_linear_ivp(a,b,k,xscheb,chebintl,ps,qs,fs,ys,yders,yder2s)

!
!  Report on the error 
!

!
!  Report on the error
!

errmax = 0

allocate(pols(100),polders(100),polder2s(100))


do i=1,k
t = xs(i)
call leges(10,t,pols,polders,polder2s)
dd1  = abs(pols(11)-ys(i))
dd2 = abs(polders(11)-yders(i))
dd3 = abs(polder2s(11)-yder2s(i))
errmax = max(dd1,max(dd2,max(dd1,errmax)))
end do

call prin2("ode_linear_ivp errmax = ",errmax)

!
!  Test the terminal value routine.
!

ys     = 0
yders  = 0
yder2s = 0

ys(k)    = -(49343d0/262144d0)
yders(k) = -(151855d0/65536d0)

call ode_linear_tvp(a,b,k,xscheb,chebintr,ps,qs,fs,ys,yders,yder2s)

do i=1,k
t = xs(i)
call leges(10,t,pols,polders,polder2s)
dd1  = abs(pols(11)-ys(i))
dd2 = abs(polders(11)-yders(i))
dd3 = abs(polder2s(11)-yder2s(i))
errmax = max(dd1,max(dd2,max(dd1,errmax)))
end do

call prin2("ode_linear_tvp errmax = ",errmax)
call prina("")

end subroutine

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  test trapezoidal routines
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


subroutine test_trap()
implicit double precision (a-h,o-z)

!
!  Tese the two subroutines implementing the trapezoidal methods for ordinary 
!  differential equations.  It uses these subroutines to solve the ordinary 
!  differential equation
!
!    y''(t) + (y'(t))^2 + (y(t))^4 = exp(t) + exp(2t) + exp(4t)
!
!  and measure the order of convergence of the obtained solutions.
!

double precision, allocatable :: ts(:),ys(:),yders(:),yder2s(:)

niters = 7
k      = 60
a      = 0.0d0
b      = 1.0d0

do iter = 1,niters
k = k*2

allocate(ts(k),ys(k),yders(k),yder2s(k))

do i=1,k
ts(i) = a + (b-a)/(k-1) * (i-1)
end do

ys       = 0
yders    = 0
yder2s   = 0
ys(1)    = exp(a)
yders(1) = exp(a)

call ode_trap_ivp(ier,k,ts,ode1,ys,yders,yder2s)

dd1 = maxval(abs(ys - exp(ts)))
dd2 = maxval(abs(yders - exp(ts)))
dd3 = maxval(abs(yder2s - exp(ts)))
dd1 = max(dd1,dd2,dd3)

if (iter .gt. 1) then
call prin2("convergence ratio = ",dd0/dd1)
endif
dd0 = dd1
deallocate(ts,ys,yders,yder2s)

end do

call prin2("ode_trap_ivp, final error = ",dd0)

call prina("")

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

k      = 12
a      = 0.0d0
b      = 1.0d0

do iter = 1,niters
k = k*2

allocate(ts(k),ys(k),yders(k),yder2s(k))


do i=1,k
ts(i) = a + (b-a)/(k-1) * (i-1)
end do

ys       = 0
yders    = 0
yder2s   = 0

ys(k)    = exp(b)
yders(k) = exp(b)

call ode_trap_tvp(ier,k,ts,ode1,ys,yders,yder2s)

dd1 = maxval(abs(ys - exp(ts)))
dd2 = maxval(abs(yders - exp(ts)))
dd3 = maxval(abs(yder2s - exp(ts)))
dd1 = max(dd1,dd2,dd3)


if (iter .gt. 1) then
call prin2("convergence ratio = ",dd0/dd1)
endif
dd0 = dd1
deallocate(ts,ys,yders,yder2s)

end do
call prin2("ode_trap_tvp, final error = ",dd0)

call prina("")

end subroutine


end module
program test_odesolve

use utils
use odesolve
use test_odesolve_routines

implicit double precision (a-h,o-z)

call test_trap()
call test_linear()
call test_nonlinear()
call test_nonlinear_adap()

end program
