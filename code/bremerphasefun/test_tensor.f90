module test_tensor_functions

implicit double precision (a-h,o-z)

contains

subroutine testfun2(x,y,val)
implicit double precision (a-h,o-z)
double precision, intent(in)  :: x,y
double precision, intent(out) :: val
val = (1+x)*cos(y)/(0.1d0+x**2)
end subroutine

subroutine testfun2_0(x,val)
implicit double precision (a-h,o-z)
double precision, intent(in)  :: x
double precision, intent(out) :: val
val = (1.0d0+x)/(0.1d0+x**2)
end subroutine

subroutine testfun(x,y,val)
implicit double precision (a-h,o-z)
double precision, intent(in)  :: x,y
double precision, intent(out) :: val
val = cos(x**2)+y**2
end subroutine


end module

program test_tensor

use utils
use chebyshev
use tensor
use test_tensor_functions

implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb1(:),whtscheb1(:),ucheb1(:,:),vcheb1(:,:), &
   chebintl1(:,:),chebintr1(:,:)
double precision, allocatable :: xscheb2(:),whtscheb2(:),ucheb2(:,:),vcheb2(:,:), &
   chebintl2(:,:),chebintr2(:,:)
double precision, allocatable :: utensor(:,:)

double precision, allocatable :: vals(:),coefs(:)
double precision, allocatable :: ab(:,:),coefst(:,:),valst(:,:)

!
!  Build the Chebyshev expansions
!

k1 = 16
k2 = 24

call chebexps(k1,xscheb1,whtscheb1,ucheb1,vcheb1,chebintl1,chebintr1)
call chebexps(k2,xscheb2,whtscheb2,ucheb2,vcheb2,chebintl2,chebintr2)

call tensor_umatrix(k1,k2,xscheb1,xscheb2,utensor)

!
!  Test an expansion on a single 
!

allocate(vals(k1*k2),coefs(k1*k2))

a =  0d0
b =  1d0
c =  0d0
d =  1d0

do i=1,k1
x = xscheb1(i) * (b-a)/2 + (b+a)/2
do j=1,k2
y = xscheb2(j) * (d-c)/2 + (c+d)/2
call testfun(x,y,val)
vals(i + (j-1)*k1) = val
end do
end do

coefs = matmul(utensor,vals)

call prin2("vals = ",vals)
call prin2("coefs = ",coefs)

x = 0.1d0
y = 0.5d0
call tensor_evalder(k1,k2,a,b,c,d,coefs,x,y,val,derx,dery)
call testfun(x,y,val0)

call prin2("evaluation error = ",val0-val)


!
!  Adaptively discretize the test function
!

eps = 1.0d-14
call chebadap(ier,eps,a,b,testfun2_0,k1,xscheb1,ucheb1,nints,ab)
call prin2("ab = ",ab)

!
!  Form coefficient expansions
!

allocate( valst(k1*k2,nints), coefst(k1*k2,nints) )

do int=1,nints
a0 = ab(1,int)
b0 = ab(2,int)
do i=1,k1
do j=1,k2

x = (b0-a0)/2*xscheb1(i) + (b0+a0)/2
y = (d-c)/2*xscheb2(j) + (d+c)/2

call testfun2(x,y,val)
valst(i + (j-1)*k1,int) = val

end do
end do
end do

do int=1,nints
coefst(:,int) = matmul(utensor, valst(:,int))
end do

x = 0.10d0
y = 0.75d0

call tensorpw_evalder(k1,k2,nints,ab,c,d,coefst,x,y,val,derx,dery)
call testfun2(x,y,val0)

call prin2("evaluation error = ",val-val0)


end program

