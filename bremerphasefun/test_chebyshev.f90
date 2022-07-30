module test_chebyshev_subroutines

use utils
use chebyshev

contains

subroutine testfun(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)     :: t
double precision, intent(out)    :: val

val  = 0.25*(cos(t)/sin(t))

end subroutine

subroutine testfun00(t,val,der)
implicit double precision (a-h,o-z)
double precision, intent(in)     :: t
double precision, intent(out)    :: val

val  = 0.25*(cos(t)/sin(t))
der = -0.25d0 / sin(t)**2

end subroutine


subroutine testfun2(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)     :: t
double precision, intent(out)    :: val

val  = 1.0d0/(1-t**2)

end subroutine

end module

program test_chebyshev

use utils
use chebyshev
use test_chebyshev_subroutines

implicit double precision (a-h,o-z)

double precision, allocatable :: xs(:),whts(:),u(:,:),v(:,:),aintl(:,:),aintr(:,:)
double precision, allocatable :: ab(:,:),vals(:,:),coefs(:,:),xs0(:),errs(:)


!
!  Fetch the Clenshaw-Curtis quadrature.
!

k = 16
call chebexps(k,xs,whts,u,v,aintl,aintr)

call prin2("xs = ",xs)
call prin2("whts = ",whts)


!
!  Adaptively discretize the input function
!

eps   = 1.0d-14

a     = 1.0d-15
b     = 1.0d0

call chebadap(ier,eps,a,b,testfun2,k,xs,u,nints,ab)
if (ier .ne. 0) then
call prini("after chebadap, ier = ",ier)
stop
endif

call prin2("after chebadap, ab = ",ab)
call prini("after chebadap, nints = ",nints)

call chebrefine(ier,eps,nints,ab,testfun,k,xs,u)
if (ier .ne. 0) then
call prini("after chebrefine, ier = ",ier)
stop
endif


if (nints .lt. 1000) then
call prin2("after chebrefine, ab = ",ab)
endif

call prini("after chebrefine, nints = ",nints)


!
!  Evaluate it and its coefficients, and plot the function.
!

allocate(vals(k,nints),coefs(k,nints))

do int=1,nints
a0 = ab(1,int)
b0 = ab(2,int)

do i=1,k
t = (b0-a0)/2*xs(i)+(b0+a0)/2
call testfun(t,val)
vals(i,int) = val
end do
end do

do int=1,nints
coefs(:,int) = matmul(u,vals(:,int))
end do

call chebpw_plot("log of input fun*",1,nints,ab,k,xs,log(vals))


n = 100000
allocate (xs0(n),errs(n))
do i=1,n
xs0(i) = (b-a)/(n-1)*(i-1)
end do
xs0 = a + xs0 * (b-a)

!!!!!!!!!!!!!!!!!!!
call elapsed(t1)
do i=1,n
x0 = xs0(i)
call chebpw_eval(nints,ab,k,xs,vals,x0,val)
call testfun(x0,val0)
errs(i) = abs(val-val0)/abs(val0)
end do

call elapsed(t2)

call prin2("barycentric time = ",t2-t1)

dmax = maxval(errs)

call prin2("maximum relative  error = ",dmax)



!!!!!!!!!!!!!!!!!!


call elapsed(t1)
do i=1,n
x0 = xs0(i)
call chebpw_eval2(nints,ab,k,coefs,x0,val)
call testfun(x0,val0)
errs(i) = abs(val-val0)/abs(val0) 
end do
call elapsed(t2)

call prin2("coefficient time = ",t2-t1)

dmax = maxval(errs)
call prin2("maximum relative  error = ",dmax)


end program
