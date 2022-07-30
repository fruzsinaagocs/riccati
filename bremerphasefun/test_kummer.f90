module test_kummer_subroutines

use utils
use chebyshev
use odesolve
use kummer

implicit double precision (a-h,o-z)

double precision :: dlambda


contains



subroutine legetayl(gamma,t,yt,ypt)
implicit double precision (a-h,o-z)

!
!  Approximate the function
!
!       f(t) = \sqrt(sin(t)) P_gamma(cos(t)),
!
!  where P_\gamma is the Legendre function of the first kind of order gamma,
!  and its derivative at a point a which is very close to 0 using third
!  order Taylor expansions.
!

yt = Sqrt(t) + (-0.08333333333333333 - gamma/4. - gamma**2/4.)*t**2.5 +  &
(0.0006944444444444445 + gamma/96. + (5*gamma**2)/192. + gamma**3/32. + gamma**4/64.)*t**4.5 + &
(-0.000041335978835978834 - gamma**2/2304. - gamma**3/768. - gamma**4/576. - gamma**5/768. &
- gamma**6/2304.)*t**6.5


ypt=  1/(2.*Sqrt(t)) + (-0.20833333333333334 - (5*gamma)/8. - (5*gamma**2)/8.)*t**1.5 + &
(0.003125 + (3*gamma)/64. + (15*gamma**2)/128. + (9*gamma**3)/64. + (9*gamma**4)/128.)*t**3.5 + &
 (-0.0002686838624338624 - (13*gamma**2)/4608. - (13*gamma**3)/1536. - (13*gamma**4)/1152. - &
(13*gamma**5)/1536. - (13*gamma**6)/4608.)*t**5.5


end subroutine


subroutine legezero(n,val)
implicit double precision (a-h,o-z)

!
!  Approximate the value of the Legendre polynomial of order n at the 
!  point 0 when n is large.
!

i = mod(n,4)

if (i .eq. 0) dd = 1
if (i .eq. 1) dd = 0
if (i .eq. 2) dd = -1
if (i .eq. 3) dd = 0

dn  = n

val = -1.9947114020071635q-1*(1/dn)**1.5q0+  &
2.4933892525089543q-2*(1/dn)**2.5q0+  &
3.116736565636193q-2*(1/dn)**3.5q0-  &
8.181433484795006q-3*(1/dn)**4.5q0-  &
3.886180905277628q-2*(1/dn)**5.5q0+  &
1.0579859670069733q-2*(1/dn)**6.5q0+  &
1.1969303265980789q-1*(1/dn)**7.5q0-  &
3.1813864021737577q-2*(1/dn)**8.5q0-  &
6.828657531754342q-1*(1/dn)**9.5q0+  &
sqrt(1/dn)*sqrt(6.366197723675814q-1)

val = dd * val 

end subroutine



subroutine lege0(n,x,pol)
implicit double precision (a-h,o-z)
integer n
double precision x,pol
!
!  Evaluate the Legendre polynomial of degree n at the point x using the
!  3-term recurrence relation.  As is well-known, this is somewhat inaccurate
!  when x is near the points +- 1.
!
!  Input parameters:
!
!    n - the degree of the polynomial to evaluate
!    x - the point at which the polynomial is to be evaluated
!
!  Output parameters:
!
!    pol - the value of P_n(x)
!

if (n == 0) then
pol = 1.0d0
else if (n == 1) then
pol = x
else
p1 = 1
p2 = x

do j=2,n
   p  = ((2*j-1)*x*p2-(j-1)*p1)/j
   p1 = p2
   p2 = p
end do

pol = p
endif

end subroutine


subroutine qlegendre(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)   :: t
double precision, intent(out)  :: val
val = 0.5d0 + dlambda + dlambda**2 + 0.25d0 * (cos(t)/sin(t))**2
end subroutine



end module



program test_kummer

use utils
use odesolve
use kummer
use test_kummer_subroutines

implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),chebintl(:,:),chebintr(:,:), &
   ucheb(:,:),vcheb(:,:)

double precision, allocatable :: ab(:,:),alpha(:,:),alphap(:,:),alphapp(:,:),alphappp(:,:)
double precision, allocatable :: alphainv(:,:),alphainvp(:,:),abinv(:,:)

double precision, allocatable :: alpha_coefs(:,:),alphainv_coefs(:,:),alphap_coefs(:,:)
double precision, allocatable :: ts(:),vals(:),vals0(:),errs(:)



call mach_zero(eps0)
pi = acos(-1.0d0)

norder   =  10**4
dlambda  =  norder

a        =  1.0d-15
b        =  pi/2

call prinl("before kummer_adap, norder = ",norder)

!
!  Fetch the Chebyshev quadrature and related matrices.
!

eps    = 1.0d-14
k      = 16
ifleft = 1

if (eps0 .lt. 1.0d-20) eps = 1.0d-20

call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)


!
!  Construct the phase function and its inverse
!

call elapsed(t1)
call kummer_adap(eps,a,b,qlegendre,k,xscheb,chebintl,chebintr,ucheb, &
  nints,ab,alphap,alphapp)
call elapsed(t2)
t_phase = t2-t1


call kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alpha,alphap,alphapp)

call prini("after kummer_adap, nints = ",nints)
call prin2("after kummer_adap, ab = ",ab)

call kummer_phase_inverse(nints,ab,k,xscheb,chebintl,ucheb,alpha,alphap, &
    nintsinv,abinv,alphainv,alphainvp)

!
!  Compute the coefficients c1 and c2 such that 
!
!   P_n(cos(t)) (sin(t))^(1/2) =  c1 sin ( alpha(t) + c2 ) (alpha'(t))^(-1/2)
!

if (ifleft .eq. 1) then
call legetayl(dlambda,a,ya,ypa)
call kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,ya,ypa,c1,c2)
else
call legezero(norder,yb)
call legezero(norder+1,ypb)
ypb = -ypb*(norder+1)
call kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,yb,ypb,c1,c2)
endif


call prind("after kummer_coefs, c1 = ",c1)
call prind("after kummer_coefs, c2 = ",c2)


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
!  Evaluate the solution at a collection of points in the interval [0,1]
!  and compare the results with those obtained via the recurrence relation.
!  Make sure the include points near 0 and points near 1.
!

nts = 200
allocate(ts(nts),vals(nts),vals0(nts),errs(nts))
call random_seed()

do i=1,nts/2
call random_number(ts(i))
end do

do i=1,nts/4
call random_number(dd)
ts(i+nts/2) = 1.0d0-exp(-15*dd)
end do

do i=1,nts/4
call random_number(dd)
ts(i+nts/2+nts/4) = exp(-30*dd)
end do

call insort0(nts,ts)

call prind("ts=",ts)

call elapsed(t1)
do i=1,nts
t = ts(i)
call lege0(norder,t,vals0(i))
end do
call elapsed(t2)
t_recurrence =  t2-t1


call elapsed(t1)
do i=1,nts
t = ts(i)
u = acos(t)


call kummer_eval(nints,ab,k,xscheb,alpha,alphap,c1,c2,u,val)

vals(i)     = val * (1-t**2)**(-0.25d0)
end do
call elapsed(t2)
t_phase_eval =  t2-t1


errs = abs(vals0-vals)


call prin2("errs = ",errs)
call prina("")

call prin2("phase function time = ",t_phase)
call prini("number of intervals = ",nints+nints2)
call prin2("largest absolute error = ",maxval(errs))
call prin2("evaluation via recurrence time =",t_recurrence)
call prin2("evaluation via phase function time =",t_phase_eval)


!
!  Calculate the Gauss-Legendre quadrature rule of order norder.
!

call elapsed(t1)
dsum = 0
dmax = 0

dd    = 2.0d0/c1**2

!
!  Construct the quadrature and report the time
!

call elapsed(t1)

dmax = 0
int1 = 1
int2 = 1
dsum = 0

do i=1,norder/2

xx = pi*i-c2
call chebpw_eval20(nintsinv,abinv,k,alphainv_coefs,xx,root,int1)
call chebpw_eval20(nints,ab,k,alphap_coefs,root,apval,int2)

! call chebpw_eval0(nintsinv,abinv,k,xscheb,alphainv,xx,root,int1)
! call chebpw_eval0(nints,ab,k,xscheb,alphap,root,apval,int2)

x   = cos(root)
wht = dd*sin(root)/apval
dsum = dsum + 2*wht


end do

call elapsed(t2)

t_quad = (t2-t1)

call prina("")
call prind("sum of weights = ",dsum)
call prin2("time for quadrature = ",2*t_quad)


end program
