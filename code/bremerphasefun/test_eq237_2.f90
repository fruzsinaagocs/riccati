module test_eq237_2_subroutines

use utils
use chebyshev
use odesolve
use kummer

implicit double precision (a-h,o-z)

double precision :: lambda
double precision :: eps
character *100 :: arg
integer :: nfeval
integer :: nreps

contains


subroutine q237(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)   :: t
double precision, intent(out)  :: val
val = lambda**2*(1.0d0 - cos(3.0d0*t)*t**2) 
nfeval = nfeval + 1
end subroutine


end module



program test_eq237_2

use utils
use odesolve
use kummer
use test_eq237_2_subroutines

implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),chebintl(:,:),chebintr(:,:), &
   ucheb(:,:),vcheb(:,:)

double precision, allocatable :: ab(:,:),alpha(:,:),alphap(:,:),alphapp(:,:),alphappp(:,:)
double precision, allocatable :: alphainv(:,:),alphainvp(:,:),abinv(:,:)

double precision, allocatable :: alpha_coefs(:,:),alphainv_coefs(:,:),alphap_coefs(:,:)
double precision, allocatable :: ts(:),vals(:)

call getarg(1, arg)
read(arg, *) lambda
call getarg(2, arg)
read(arg, *) eps

call mach_zero(eps0)
pi = acos(-1.0d0)

nfeval   = 0
nreps = 1000
a        =  -1.0d0
b        =  1.0d0

!call prinl("before kummer_adap, norder = ",norder)

!
!  Fetch the Chebyshev quadrature and related matrices.
!

!eps    = 1.0d-14
k      = 16
ifleft = 1

if (eps0 .lt. 1.0d-20) eps = 1.0d-20

call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)


!
!  Construct the phase function and its inverse
!

call elapsed(t1)
do nreps = 1, nreps
call kummer_adap(eps,a,b,q237,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alphap,alphapp)
end do
call elapsed(t2)
t_phase = (t2-t1)/nreps
nfeval = nfeval/nreps


call kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alpha,alphap,alphapp)

call kummer_phase_inverse(nints,ab,k,xscheb,chebintl,ucheb,alpha,alphap, &
    nintsinv,abinv,alphainv,alphainvp)

!
!  Set initial conditions
!

ya = 0.0d0
ypa = lambda

call kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,ya,ypa,c1,c2)


!
!  Compute coefficient expansions
!

allocate(alpha_coefs(k,nints),alphap_coefs(k,nints),alphainv_coefs(k,nintsinv))

do int=1,nints
alpha_coefs(:,int)  = matmul(ucheb,alpha(:,int))
alphap_coefs(:,int) = matmul(ucheb,alphap(:,int))
end do

do int=1,nintsinv
alphainv_coefs(:,int) = matmul(ucheb,alphainv(:,int))
end do


!
!  Evaluate the solution at the endpoint of the interval.
!

nts = 1
allocate(ts(nts),vals(nts))

ts(1) = b

!call prind("ts=",ts)

do i=1,nts
t = ts(i)


call kummer_eval(nints,ab,k,xscheb,alpha,alphap,c1,c2,t,val)

vals(i)     = val
end do

call prind("",vals(1)) ! Solution at the end of the interval
call prin2("",t_phase) ! Phase function construction time
call prini("",nfeval) ! Number of function evaluations

end program
