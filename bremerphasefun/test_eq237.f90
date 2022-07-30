module test_eq237_routines

use utils
use chebyshev
use odesolve

double precision :: gamma

contains

subroutine ode(t,y,yp,f,dfdy,dfdyp)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: t,y,yp
double precision, intent(out) :: f,dfdy,dfdyp

f     = -y*gamma**2*(1 - t**2*cos(3*t))
dfdy  = -gamma**2*(1 - t**2*cos(3*t))
dfdyp = 0.0

end subroutine


subroutine test_nonlinear_adap()
implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),ucheb(:,:),vcheb(:,:), &
   chebintl(:,:),chebintr(:,:)

double precision, allocatable :: ab(:,:),ys(:,:),yders(:,:),yder2s(:,:),abin(:,:)


!
!  Fetch the Chebyshev quadrature and associated matrices.
!

k  = 16
call prini("Calling chebexps", k)
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)
call prini("Called chebexps", k)

!
!  Setup the problem
!

gamma = 10**2

allocate(abin(2,10))
nintsin   = 10
abin(1,1) = -1.0d0
abin(2,1) = -0.8d0
abin(1,2) = -0.8d0
abin(2,2) = -0.6d0
abin(1,3) = -0.6d0
abin(2,3) = -0.4d0
abin(1,4) = -0.4d0
abin(2,4) = -0.2d0
abin(1,5) = -0.2d0
abin(2,5) = 0.0d0
abin(1,6) = 0.0d0
abin(2,6) = 0.2d0
abin(1,7) = 0.2d0
abin(2,7) = 0.4d0
abin(1,8) = 0.4d0
abin(2,8) = 0.6d0
abin(1,9) = 0.6d0
abin(2,9) = 0.8d0
abin(1,10) = 0.8d0
abin(2,10) = 1.0d0

call mach_zero(eps0)
eps         = 1.0d-14

ya          = 0
ypa         = gamma


call prini("Calling solve_ivp", k)
call ode_solve_ivp_adap(ier,eps,nintsin,abin,k,xscheb,chebintl,ucheb,nints,ab,&
   ys,yders,yder2s,ode,ya,ypa)

if (ier .ne. 0) then
call prini("after ode_solve_ivp_adap, ier = ",ier)
stop
endif

call prin2("after ode_solve_ivp_adap, ab = ",ab)
call prini("nints = ",nints)

! call chebpw_plot("alpha*",1,nints,ab,k,xscheb,ys)
! call chebpw_plot("alpha'*",2,nints,ab,k,xscheb,yders)


deallocate(ab,ys,yders,yder2s)

end subroutine


end module
program test_eq237

use utils
use odesolve
use test_eq237_routines

implicit double precision (a-h,o-z)

call test_nonlinear_adap()

end program
