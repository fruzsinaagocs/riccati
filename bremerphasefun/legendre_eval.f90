module test_kummer_legendre_subroutines

use utils
use chebyshev
use odesolve
use kummer
use kummer_legendre

implicit double precision (a-h,o-z)

contains


end module


program test_kummer_legendre

use utils
use chebyshev
use tensor
use odesolve
use kummer
use kummer_legendre
use test_kummer_legendre_subroutines

implicit double precision (a-h,o-z)


double precision, allocatable :: xscheb(:),whtscheb(:),chebintl(:,:),chebintr(:,:), &
  ucheb(:,:),vcheb(:,:)
double precision, allocatable :: ab(:,:),alpha(:,:),alphap(:,:),alphapp(:,:)
double precision, allocatable :: ts(:),vals(:),vals0(:),errs(:)

double precision, allocatable :: abcoefs(:,:),coefs(:,:),coefsder(:,:)
double precision, allocatable :: dlams(:),vals2(:,:),vals20(:,:),errs2(:,:)

call mach_zero(eps0)
pi = acos(-1.0d0)

!
!  Fetch the quadrature data
!

k = 20
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)
eps = 1.0d-13

!
!  Construct a phase function for Legendre functions
! 

dnu = 2.3232323232d5


call prin2("before kummer_legendre_phase, dnu = ",dnu)

call elapsed(t1)
call kummer_legendre_phase(eps,dnu,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alpha,alphap,alphapp)
call elapsed(t2)
t_phase = t2-t1

call kummer_legendre_pcoefs(dnu,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alpha,alphap,alphapp,a1,a2)

call kummer_legendre_qcoefs(dnu,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alpha,alphap,alphapp,b1,b2)

call prind("after kummer_legendre_pcoefs, a1 = ",a1)
call prind("after kummer_legendre_pcoefs, a2 = ",a2)

call prind("after kummer_legendre_pcoefs, b1 = ",b1)
call prind("after kummer_legendre_pcoefs, b2 = ",b2)


!
!  Evaluate P_nu at a collection of points in the interval [0,1]
!  and compare the results with those obtained via the recurrence relation.
!  Make sure to include some points near 1.
!

nts = 100
allocate(ts(nts),vals(nts),vals0(nts),errs(nts))
call random_seed()

do i=1,nts/2
call random_number(ts(i))
end do

do i=1,nts/2
call random_number(dd)
ts(i+nts/2) = 1-exp(-15*dd)
end do

call insort0(nts,ts)
call prind("ts=",ts)

call elapsed(t1)
do i=1,nts
t = ts(i)
call kummer_legendre_precurrence(dnu,t,vals0(i))
end do
call elapsed(t2)
t_recurrence =  (t2-t1)

call elapsed(t1)
do i=1,nts
t = ts(i)
u = -log(1-t)
call kummer_eval(nints,ab,k,xscheb,alpha,alphap,a1,a2,u,val)
vals(i)     = val / sqrt(1+t)
end do
call elapsed(t2)
t_eval =  (t2-t1)

errs = abs(vals0-vals)

call prin2("errors in P_nu = ",errs)


!
!  Evaluate Q_nu at a collection of points in the interval [0,1]
!  and compare the results with those obtained via the recurrence relation.
!  Make sure to include some points near 1.
!


call elapsed(t1)
do i=1,nts
t = ts(i)
call kummer_legendre_qrecurrence(dnu,t,vals0(i))
end do
call elapsed(t2)
t_recurrence =  t_recurrence+(t2-t1)

call elapsed(t1)
do i=1,nts
t = ts(i)
u = -log(1-t)
call kummer_eval(nints,ab,k,xscheb,alpha,alphap,b1,b2,u,val)
vals(i)     = val / sqrt(1+t)
end do
call elapsed(t2)
t_eval =  t_eval+(t2-t1)

t_eval       = t_eval/(2*nts)
t_recurrence = t_recurrence/(2*nts)

errs = abs(vals0-vals)

call prin2("errors in Q_nu = ",errs)
call prina("")
call prina("")
call prin2("order = ",dnu)
call prini("number of phase function coefficients = ",nints*k)
call prin2("phase function time = ",t_phase)
call prin2("average evaluatime time via recurrence relation = ",t_recurrence)
call prin2("average evaluation time via phase function  = ",t_eval)
call prina("")

!
!  Build coefficient expansions in both dnu and t for the phase function and
!  its derivative; use them to evaluate Legendre polynomials.
!

eps = 1.0d-13
nx  = 16
ny  = 4

c   = 1.0d4
d   = 1.0d8

call kummer_legendre_expansion(eps,nx,ny,c,d,nintscoefs,abcoefs,coefs,coefsder)


nlams = 8
nts   = 50
deallocate(ts)

allocate(ts(nts))
allocate(dlams(nlams),vals2(nts,nlams),vals20(nts,nlams),errs2(nts,nlams))

call random_seed()

do i=1,nlams
call random_number(dd)
dlams(i) = c + (d-c)*dd
end do

do i=1,nts
call random_number(dd)
ts(i) = dd
end do

call insort0(nlams,dlams)
call prind("dlams=",dlams)
call prind("ts=",ts)


call elapsed(t1)
do i=1,nts
do j=1,nlams
t    = ts(i)
dnu = dlams(j)
call kummer_legendre_precurrence(dnu,t,vals20(i,j))
end do
end do
call elapsed(t2)
t_recurrence =  (t2-t1)

call elapsed(t1)
do i=1,nts
do j=1,nlams
t   = ts(i)
dnu = dlams(j)
call kummer_legendre_evalp(nx,ny,c,d,nintscoefs,abcoefs,coefs,coefsder,dnu,t,val)
vals2(i,j)     = val
end do
end do
call elapsed(t2)
t_eval =  t_eval+(t2-t1)

do i=1,nts
do j=1,nlams
errs2(i,j) = abs(vals20(i,j)-vals2(i,j))
end do
end do

t_recurrence = t_recurrence / (nts*nlams)
t_eval       = t_eval / (nts*nlams)

call prin2("errs2 = ",errs2)
call prin2("maximum error = ",maxval(errs2))
call prin2("average evaluation time time via recurrence relation = ",t_recurrence)
call prin2(" average evaluation time via phase function  = ",t_eval)
call prini("size of expansion = ",nints*nx*ny)

end program
