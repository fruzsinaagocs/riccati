!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  
!  This module contains code for representing a function of two variables given on
!  a rectangle [a,b] x [c,d] as a tensor product of Chebyshev polynomials;  that is,
!  in the form
!       
!              nx-1  ny-1        
!     f(x,y) = sum  sum  c_ij T_{1,i}(x) T_{2,j}(y)                                       (1)
!              i=0   j=0            
!
!  where T_{1,i} denotes the Chebyshev polynomial of degree i on the interval [a,b]
!  and T_{2,i} denote the Chebyshev polynomial of degree i on the interval [c,d].
!
!  It also contains routines for forming and evaluating expansions of the form
!
!              nx-1 ny-1                                   
!     f(x,y) = sum  sum  sum c_{ijl} T_{i,l}(x) T_j(y)                                    (2)
!              i=0  j=0   l
!
!  where  [a,b] is the disjoint union of the intervals [a_l,b_l] and T_{i,l}
!  denotes the Chebyshev polynomial of degree i on [a_l,b_l].  That is, f(x,y)
!  is the tensor product of a Chebyshev expansion in the y variable and a 
!  piecewise Chebyshev expansion on the interval [a_l,b_l] in the x variable.
!
!  The following subroutines should be regarded as public:
!
!    tensor_exps - return a quadrature on the rectangle [-1,1] x [-1,1] and
!
!    tensor_umatrix - return the matrix which takes the values of a funtion f at
!      nodes of a tensor product of Clenshaw-Curtis quadrature formulas to the 
!      coefficients in the expansion (1)
!
!    tensor_eval - calculate the value of an expansion of the form (1)  at 
!      a specified point
!
!    tensor_eval der- calculate the value of an expansion of the form (1) as well
!      as well as its derivatives with respect to x and y at a specified point
!
!    tensorpw_eval - evaluate a function which is represented via a piecewise
!      Chebyshev expansion in the x variable and a "single" Chebyshev expansion 
!      in the variable; that is, one of the form (2)
!
!    tensorpw_evalder - evaluate a function of the form (2) and its derivatives
!      w.r.t. x and y at a specified point
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module tensor

use chebyshev

contains

subroutine tensor_exps(nx,ny,nfuns,nquad,xs,ys,whts,u)
implicit double precision (a-h,o-z)

integer, intent(in)                        :: nx,ny
integer, intent(out)                       :: nquad,nfuns
double precision, allocatable, intent(out) :: xs(:),ys(:),whts(:),u(:,:)

!
!  Return a quadrature formula on the rectangle [-1,1] x [-1,1] which
!  integrates functions of the form
!
!     T_i(x) T_j(y)   with 0 <= i < nx-1 and 0 <= j < ny-1
!
!  as well as a matrix which takes the values of an expansion of the
!  form (1) at those nodes to the coefficients in the expansion (1).
!
!  The user should not make assumptions about the order of such 
!
!  Input parameters:
!    nx - the parameter nx in (1)
!    ny - the parameter ny in (1)
!
!  Output parameters:
!    nquad - the number of nodes in the quadrature formula
!    nfuns - the number of functions in the expansion (1)
!    (xs,ys,whts) - 
!    u - a (nfuns,nquad) matrix which takes the values of an expansion
!     of the form (1) to the coefficients a_ij in the expansion
!

double precision, allocatable :: xscheb1(:),whtscheb1(:),ucheb1(:,:),vcheb1(:,:), &
   chebintl1(:,:), chebintr1(:,:)
double precision, allocatable :: xscheb2(:),whtscheb2(:),ucheb2(:,:),vcheb2(:,:), &
   chebintl2(:,:), chebintr2(:,:)

call chebexps(nx,xscheb1,whtscheb1,ucheb1,vcheb1,chebintl1,chebintr1)
call chebexps(ny,xscheb2,whtscheb2,ucheb2,vcheb2,chebintl2,chebintr2)

nquad = nx*ny
nfuns = nx*ny
allocate(xs(nquad),ys(nquad),whts(nquad))

idx=1
do j=1,ny
do i=1,nx
xs(idx) = xscheb1(i)
ys(idx) = xscheb2(j)
idx = idx + 1
end do
end do

call tensor_umatrix(nx,ny,xscheb1,xscheb2,u)


end subroutine


subroutine tensor_umatrix(nx,ny,xs,ys,u)
implicit double precision (a-h,o-z)

integer, intent(in)                         :: nx,ny
double precision, intent(in)                :: xs(:),ys(:)
double precision, allocatable, intent(out)  :: u(:,:)

!
!  Return the matrix which takes the vector of values
!
!      f( xs(1)    , ys(1) )
!      f( xs(2)    , ys(1) )
!               .
!               .
!      f( xs(nx)   , ys(1) )
!      f( xs(1)    , ys(2) )
!      f( xs(2)    , ys(2) )                                                      (2)
!               .
!               .
!
!      f( xs(1)    , ys(ny) )
!      f( xs(2)    , ys(ny) )
!               .
!               .
!      f( xs(nx)    , ys(ny) ),
!
!  where xs(1), ..., xs(nx) are the nodes of the nx-point Clenshaw-Curtis quadrature
!  formula and ys(1), ..., ys(ny) are the nodes of the ny-point Clenshaw-Curtis
!  quadrature formula, to the vector of coefficients 
!
!      a_00
!      a_10
!      a_20
!       .
!       .
!       .
!      a_{nx-1}0                                                                 (3)
!      a_01
!      a_11
!        .
!        .
!
!  in the expansion
!
!              nx-1  ny-1
!     f(x,y) = sum sum  a    T_i(x) T_j(y)
!              i=0 j=0   ij
!
!  of f in terms of tensor products of Chebyshev polynomials.
!
!
!  Input parameters:
!    nx - the parameter nx in (1)
!    ny - the parameter ny in (1)
!    xs - an array containing the points of the nx-point Chebyshev grid on the interval
!      [-1,1]
!    ys - an array containing the points of the ny-point Chebyshev grid on the interval
!      [-1,1]
!    
!  Output parameters:
!    u - the (nx*ny,nx*ny) matrix which takes the vector of values (2) 
!     to the vector of coefficients (3)
!
!

double precision, allocatable :: polsx(:),polsy(:)

allocate(polsx(nx+1), polsy(ny+1),u(nx*ny,nx*ny))

!
!  Use the identity
!
!
!              n  ''                        { \delta_ij * (n)   if i = 0 or i = n
!             \sum   T_i(x_k) T_j(x_k)  =   {
!              k=0                          { \delta_ij * (n/2) if 0 < i < n
!
!  to form the matrix.
!

do i1=1,nx
x    = xs(i1)
call chebs(x,nx,polsx)
if (i1 .eq. 1 .OR. i1 .eq. nx) polsx = polsx/2.0d0

do i2=1,ny
y    = ys(i2)
call chebs(y,ny,polsy)
if (i2 .eq. 1 .OR. i2 .eq. ny) polsy = polsy/2.0d0

do j1=1,nx
do j2=1,ny
val    = polsy(j2)*polsx(j1)

dscaley = 1.0d0/(ny-1)
if (j2 .gt. 1 .AND. j2 .lt. ny) dscaley=dscaley*2

dscalex = 1.0d0/(nx-1)
if (j1 .gt. 1 .AND. j1 .lt. nx) dscalex=dscalex*2

u(j1 + nx*(j2-1),i1+(i2-1)*nx) = val * dscaley * dscalex

end do
end do

end do
end do


end subroutine


subroutine tensor_evalder(nx,ny,a,b,c,d,coefs,x,y,val,derx,dery)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nx,ny
double precision, intent(in)  :: a,b,c,d,x,y,coefs(nx*ny)
double precision, intent(out) :: val,derx,dery

!
!  Evaluate an expansion of the form (1) and its derivatives with respect
!  to x and y at a specified point.
!
!  Input parameters:
!

!    nx - the number of terms in the expansion in the x variable
!    ny - the number of terms in the expansion in the y variable
!    (a,b) - the interval over which x varies
!    (c,d) - the interval over which y varies
!
!    coefs - the array of coefficients in the expansion
!    (x,y) - the point at which to evaluate the expansion
!
!  Output parameters:
!
!   val -the value of the expansion (1) at the specified point
!   derx - the derivative w.r.t. x of the expansion (1) at the specified point
!   dery - the derivative w.r.t. y of the expansion (1) at the specified point
!

double precision :: polsx(nx+1),polsy(ny+1),dersx(nx+1),dersy(ny+1)

xx = (x-(b+a)/2)*2/(b-a)
yy = (y-(d+c)/2)*2/(d-c)

call chebders(xx,nx,polsx,dersx)
call chebders(yy,ny,polsy,dersy)

val  = 0
derx = 0
dery = 0

idx = 0
do j2=1,ny
do j1=1,nx
idx = idx+1
val  = val  + coefs(idx) * polsx(j1)*polsy(j2)
derx = derx + coefs(idx) * dersx(j1)*polsy(j2)
dery = dery + coefs(idx) * polsx(j1)*dersy(j2)
end do
end do



derx = derx * 2/(b-a)
dery = dery * 2/(d-c)

end subroutine


subroutine tensor_eval(nx,ny,a,b,c,d,coefs,x,y,val)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nx,ny
double precision, intent(in)  :: a,b,c,d,x,y,coefs(nx*ny)
double precision, intent(out) :: val

!
!  Evaluate an expansion of the form (1) and its derivatives with
!  respect to x and y at a specified point.
!
!
!  Input parameters:
!
!    nx - the parameter nx in (1)
!    ny - the parameter ny in (1)
!    (a,b) - the interval over which x varies
!    (c,d) - the interval over which y varies
!    coefs - the array of coefficients in the expansion
!    (x,y) - the point at which to evaluate the expansion
!
!  Output parameters:
!
!   val - the value of the expansion at the desired point
!   derx - the value of the derivative of the expansion w.r.t. x
!   dery - the value of the derivative of the expansion w.r.t. y
!

double precision :: polsx(nx+1),polsy(ny+1)

xx = (x-(b+a)/2)*2/(b-a)
yy = (y-(d+c)/2)*2/(d-c)

call chebs(xx,nx,polsx)
call chebs(yy,ny,polsy)

val  = 0

idx = 0
do j2=1,ny
do j1=1,nx
idx = idx+1
val  = val  + coefs(idx) * polsx(j1)*polsy(j2)
end do
end do

end subroutine





subroutine tensorpw_eval(nx,ny,nints,ab,c,d,coefs,x,y,val)
implicit double precision (a-h,o-z)


integer, intent(in)          :: nx,ny,nints
double precision, intent(in) :: ab(2,nints),c,d
double precision, intent(in) :: coefs(nx*ny,nints)

!
!  Evaluate an expansion of the form (2) at a specified point.
!
!
!  Input parameters:
!    nx - the parameter nx in (2)
!    ny - the parameter ny in (2)
!    nints - the number of subintervals into which [a,b] is subdivided
!    ab - a (2,nints) array each column of which gives the endpoints of
!      one of the intervals
!    (c,d) - the interval over which y varies
!    coefs - the array of coefficients in the expansion
!    (x,y) - the point at which to evaluate the expansion
!
!  Output parameters:
!   val - the value of the expansion at the point x
!

!
!  Find the index of the interval containing the point x
!


niters = 6
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
cc     = ab(1,int)
if (x .gt. cc) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do

do int= intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do


a = ab(1,int)
b = ab(2,int)


!
!  Evaluate the expansions 
!


call tensor_eval(nx,ny,a,b,c,d,coefs(:,int),x,y,val)

end subroutine



subroutine tensorpw_evalder(nx,ny,nints,ab,c,d,coefs,x,y,val,derx,dery)
implicit double precision (a-h,o-z)


integer, intent(in)          :: nx,ny,nints
double precision, intent(in) :: ab(2,nints),c,d
double precision, intent(in) :: coefs(nx*ny,nints)


!  Evaluate an expansion of the form (2) and its derivatives
!  at a specified point.
!
!
!  Input parameters:
!    nx - the parameter nx in (2)
!    ny - the parameter ny in (2)
!    nints - the number of subintervals into which [a,b] is subdivided
!    ab - a (2,nints) array each column of which gives the endpoints of
!      one of the intervals
!    (c,d) - the interval over which y varies
!    coefs - the array of coefficients in the expansion
!    (x,y) - the point at which to evaluate the expansion
!
!  Output parameters:
!   val - the value of the expansion at the specified point
!   derx - the value of the derivative of (2) w.r.t. x at the specified point
!   dery - the value of the derivative of (2) w.r.t. y at the specified point
!
!

!
!  Find the index of the interval containing the point x
!


niters = 6
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
cc     = ab(1,int)
if (x .gt. cc) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do

do int= intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do


a = ab(1,int)
b = ab(2,int)


!
!  Evaluate the expansions 
!


call tensor_evalder(nx,ny,a,b,c,d,coefs(:,int),x,y,val,derx,dery)

end subroutine




end module

