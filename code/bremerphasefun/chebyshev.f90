!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  This module contains code for representing functions via piecewise Chebyshev 
!  expansions.  More accurately, functions on an interval [a,b] can be represented either 
!  via their values  at the Chebyshev grids on a collection of subintervals [a,b] or
!  via their Chebyshev expansions on said subintervals. 
!
!  The following routines should be regarded as public:
!    
!    chebexps - construct the n-point Clenshaw-Curtis quadrature on the interval [-1,1],
!      the matrix which takes the values of a function to the Chebyshev coefficients of the 
!      polynomial interpolating the function at the nodes of the quadrature rule, as well
!      as the "left" and "right" spectral integration matrices.
!
!    chebadap - recursively subdivide an interval [a,b] until a user-specified
!      input function is accurately represented via k-term Chebyshev expansions
!      on each subinterval
!
!    chebrefine - further subdivide an existing decomposition of [a,b] until a 
!      user-specified function is accurately represented via k-term Chebyshev
!      expansions on each subinterval
!
!    chebs - evaluate the Chebyshev polynomials of orders 0 through n at a specified
!      point in the interval [-1,1]
!
!    chebders - evaluate the Chebyshev polynomials of orders 0 through n and their
!      derivatives at a point in the interval [-1,1]
!
!    chebeval - evaluate a polynomial of degree n-1 whose values are given 
!      on the n-point Chebyshev grid on an interval [a,b] using the well-known 
!      barycentric interpolation formula
!
!    chebeval2 - use the Clenshaw algorithm in order to evaluate an n-term Chebyshev
!      expansion on the interval [a,b] at a specified point t in [a,b]
!
!    chebevalder - evaluate an n-term Chebyshev expansion on the interval [a,b]
!      and its derivative at a specified point x in [a,b]
!
!    chebpw_eval - evaluate a function specified by its values at k-point Chebyshev
!      grids on a collection of subintervals of [a,b] at an arbitrary point x in
!      [a,b]
!
!    chebpw_eval2 - evaluate a piecewise Chebyshev expansion given on a collection
!      of subintervals of [a,b] at a specified point x in [a,b]
!
!    chebpw_evalder - evaluate a piecewise Chebyshev expansion given on a collection
!      of subintervals of [a,b] and its derivative at a specified point x in [a,b]
!
!    chebpw_plot - produce a GNUplot file which generates a plot of a function 
!      represented via its values at the k-point Chebyshev grids on a collection 
!      of subintervals of [a,b]
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module chebyshev


use utils


interface

subroutine chebadapfun(t,val)

double precision, intent(in)      :: t
double precision, intent(out)     :: val
end subroutine

end interface

double precision, private :: cheby_eps00

contains


subroutine chebexps(n,xs,whts,u,v,aintl,aintr)
implicit double precision (a-h,o-z)
integer :: n
double precision, allocatable, intent(out) :: xs(:),whts(:),aintl(:,:),aintr(:,:)
double precision, allocatable, intent(out) :: u(:,:),v(:,:)
!
!
!  Construct the n-point Clenshaw-Curtis quadrature with nodes
!  
!      x_j = cos (pi j/(n-1))
!
!  on the interval [-1,1], as well the matrices taking the values of an 
!  n-term Chebyshev expansion at the nodes x_j to its coefficients and vice-versa, 
!  and two spectral integration matrices.
!
!  Input parameters:
!
!    n - the length of the quadrature formula to construct
!
!  Output parameters:
!
!    xs - an array containing the n quadrature nodes
!    whts - an array containing the n quadrature weights
!    u - the (n,n) matrix which takes the values of an n-term Chebyshev expansion
!      at the n quadrature nodes to the n expansion coefficients
!    v - the (n,n) matrix which takes the coefficients of an nterm-Chebyshev
!      expansion to its values at the n quadratue nodes
!    aintl - the "left" spectral integration matrix which takes the values
!      of a function f(t) on the Chebyshev nodes to the value of the function g(t) 
!      defined via the formula
!
!                     t
!          g(t) = \int  f(u) du
!                     a
!
!    aintr - the "right" spectral integration matrix which takes the values
!      of a function f(t) on the Chebyshev nodes to the value of the function g(t) 
!      defined via the formula
!
!                     t
!          g(t) = \int_ f(u) du
!                     b
!

double precision, allocatable :: pols(:),c(:,:),d(:,:),xx(:,:)

pi = acos(-1.0d0)
call mach_zero(eps00)

cheby_eps00 = eps00

allocate(xs(n),whts(n),u(n,n),v(n,n))
allocate(pols(n+1),c(1,n),d(1,n))

!

!  Construct the nodes
!

h = pi/(n-1)
do i=1,n
xs(n-i+1) = cos(h*(i-1))
end do

!
!  Construct the matrix u which takes values to coefficients
!

do i=1,n
x = xs(i)
call chebs(x,n-1,pols)
do j=1,n
u(j,i) = pols(j)
v(i,j) = pols(j)
end do
end do


u(1,:) = u(1,:)/2
u(n,:) = u(n,:)/2
u(:,1) = u(:,1)/2
u(:,n) = u(:,n)/2

u = u*2.0d0/(n-1)

!
!  Construct the weights by multiplying u^t on the left by the
!  integrals of the Chebyshev polynomials.
!

c=0
c(1,1) = 2.0d0
do i=2,n-1,2
c(1,i+1) = 1.0d0/(i+1)-1.0d0/(i-1)
end do

d = matmul(c,u)
whts = d(1:n,1)

!
!  Form the matrix which takes the values of a function f(t) to the values of
!
!              t
!    g(t) =  \int     f(u) du
!              a
!  

allocate(xx(n,n),aintr(n,n),aintl(n,n))

do i=1,n
call chebs(xs(i),n,pols)
xx(i,1) = xs(i)
xx(i,2) = xs(i)**2/2.0d0
do j=3,n
xx(i,j) = 0.5d0 * (pols(j+1)/j-pols(j-1)/(j-2))
end do
end do

do i=2,n
xx(i,:) = xx(i,:) - xx(1,:)
end do
xx(1,:) = 0

aintl = matmul(xx,u)

!
!  Form the matrix which takes the values of a function f(t) to the values of
!
!              t
!    g(t) =  \int     f(u) du
!              b
!  

xx = 0

do i=1,n
call chebs(xs(i),n,pols)
xx(i,1) = xs(i)
xx(i,2) = xs(i)**2/2.0d0
do j=3,n
xx(i,j) = 0.5d0 * (pols(j+1)/j-pols(j-1)/(j-2))
end do
end do

do i=1,n-1
xx(i,:) = xx(i,:) - xx(n,:)
end do
xx(n,:) = 0

aintr = matmul(xx,u)

end subroutine



subroutine chebadap(ier,eps,a,b,fun,k,xs,u,nints,ab)
implicit double precision (a-h,o-z)

double precision, intent(in)               :: eps,xs(k),u(k,k),a,b
integer, intent(in)                        :: k
integer, intent(out)                       :: nints
double precision, intent(out), allocatable :: ab(:,:)
procedure(chebadapfun)                     :: fun

!
!  This subroutine adaptively discretizes a user-specified function.  More 
!  specifically, it recursively subdivides the specified interval [a,b] until
!  it is represented via k-term polynomial expansions on each subinterval.
!
!  Input parameters:
!
!    eps - the precision with which the function should be represented
!    (a,b) - the interval on which the function is given
!    fun - an external procedure conforming to the interface chebadapfun
!      which returns the values of the function
!
!    k - the number of terms in the polynomial expansions
!    xs - the nodes of the k-point Clenshaw-Curtis quadrature returned by chebexps
!    u - the (k,k) matrix u which takes values to coefficients (as returned by
!      chebexps)
!   
!  Output parameters:
!
!    ier - an error return code;
!       ier =   0     indicates successful execution
!       ier =   4     means that the present maximum number of intervals
!                     was exceeded before an appropriate discretization scheme
!                     could be obtained
!       ier =   1024  means that the maximum number of recursion steps was
!                     exceeded 
!
!    nints - the number of subintervals in the discretization scheme
!    ab - an (2,nints) array each column of which specifies one of the
!      subintervals in the discretization scheme
!

double precision, allocatable :: ab0(:,:),ab1(:,:),vals(:),coefs(:),ts(:)

nn = k/2

ier     = 0
maxints = 1000000

allocate(ab0(2,maxints),ab1(2,maxints),vals(k),coefs(k),ts(k))

nints0   = 1
ab0(1,1) = a
ab0(2,1) = b

nints1   = 0

do while (nints0 .gt. 0) 

a0 = ab0(1,nints0)
b0 = ab0(2,nints0)
c0 = (a0+b0)/2

if (b0-a0 .eq. 0) then
ier = 1024
return
endif

nints0 = nints0-1

!
!  Evaluate the function at each of the Chebyshev nodes on [a0,b0]
!

ts = xs*(b0-a0)/2 + (b0+a0)/2

do i=1,k
call fun(ts(i),vals(i))
end do


coefs = matmul(u,vals)

!
!  Measure the relative error in the trailing coefficients
!


dd2 = maxval(abs(coefs))+1
dd1 = maxval(abs(coefs(k-nn+1:k)))
dd  = dd1/dd2


if (dd .gt. eps ) then

if (nints0+2 .gt. maxints) then
ier = 4
return
endif

c0 = (a0+b0)/2

nints0 = nints0 + 1
ab0(1,nints0) = c0
ab0(2,nints0) = b0
nints0 = nints0 + 1
ab0(1,nints0) = a0
ab0(2,nints0) = c0

else

if (nints1+1 .gt. maxints) then
ier = 4
return
endif

nints1 = nints1 + 1
ab1(1,nints1) = a0
ab1(2,nints1) = b0
endif

end do

nints = nints1
allocate(ab(2,nints))
ab = ab1(:,1:nints)


end subroutine



subroutine chebrefine(ier,eps,nints,ab,fun,k,xs,u)
implicit double precision (a-h,o-z)

double precision, intent(in)                 :: eps,xs(k),u(k,k)
integer, intent(in)                          :: k
integer, intent(inout)                       :: nints
double precision, intent(inout), allocatable :: ab(:,:)
procedure(chebadapfun)                       :: fun

!
!  This subroutine adaptively discretizes a user-specified function.  More 
!  specifically, it recursively subdivides the specified interval [a,b] until
!  it is represented via k-term polynomial expansions on each subinterval.  It
!  differs from chebadap in that it uses an existing discretization scheme as a 
!  starting point.
!
!  Input parameters:
!
!    eps - the precision with which the function should be represented
!    (nints,ab) - the subintervals of the existing discretization scheme 
!    fun - an external procedure conforming to the interface chebadapfun
!      which returns the values of the input function
!
!    k - the number of terms in the polynomial expansions
!    xs - the nodes of the k-point Clenshaw-Curtis quadrature returned by chebexps
!    u - the (k,k) matrix u which takes values to coefficients (as returned by
!      chebexps)
!   
!  Output parameters:
!
!    ier - an error return code;
!       ier =   0     indicates successful execution
!       ier =   4     means that the present maximum number of intervals
!                     was exceeded before an appropriate discretization scheme
!                     could be obtained
!       ier =   1024  means that the maximum number of recursion steps was
!                     exceeded 
!       ier =   4096  the number of intervals in the input discretization
!                     scheme exceeds the preset maximum 
!
!    nints - the number of subintervals in the new discretization scheme
!    ab - a new (2,nints) array each column of which specifies one of the
!      subintervals
!  
!

double precision, allocatable :: ab0(:,:),ab1(:,:),vals(:),coefs(:),ts(:)

ier     = 0
maxints = 1000000
nn      = k/2

allocate(ab0(2,maxints),ab1(2,maxints),vals(k),coefs(k),ts(k))

if (nints .gt. maxints) then
ier = 4096
return
endif


nints0          = nints
ab0(:,1:nints0) = ab

nints1   = 0

do while (nints0 .gt. 0) 

a0 = ab0(1,nints0)
b0 = ab0(2,nints0)
nints0 = nints0-1

if (b0-a0 .eq. 0) then
ier = 1024
return
endif

ts = xs*(b0-a0)/2 + (b0+a0)/2

do i=1,k
call fun(ts(i),vals(i))
end do

coefs = matmul(u,vals)

!
!  Measure the relative error in the trailing coefficients
!

dd2 = maxval(abs(coefs))+1.0d0
dd1 = sum(abs(coefs(k-nn+1:k)))/nn
dd  = dd1/dd2


if (dd .gt. eps) then

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

else

if (nints1+2 .gt. maxints) then
ier = 4
return
endif

nints1 = nints1 + 1
ab1(1,nints1) = a0
ab1(2,nints1) = b0
endif

end do


!
!  Copy out the resulting intervals in reserve order.
!

deallocate(ab)

nints = nints1
allocate(ab(2,nints))
do i=1,nints
ab(:,i) = ab1(:,nints-i+1)
end do

end subroutine




subroutine chebs(x,n,pols)
implicit double precision (a-h,o-z)

integer :: n 
double precision :: pols(n+1)

!
!  Evaluate the Chebyshev polynomials order 0 through n at a specified point
!  using the standard 3-term recurrence relation.
!
!  Input parameters:
!
!    x - point at which to evaluate the polynomials
!    n - the order of the polynomials to evaluate
!
!  Output parameters:
!
!    pols - this user-supplied and allocated array of length n+1 will
!      contain the values of the polynomials of order 0 through n upon
!      return
!

if (n .lt. 0) return

pols(1) = 1.0d0
if (n .eq. 0) return

pols(1) = 1.0d0
pols(2) = x
if (n .eq. 2) return

xx1 = 1.0d0
xx2 = x

do i=1,n-1
xx        = 2*x*xx2 - xx1
pols(i+2) = xx
xx1       = xx2
xx2       = xx
end do

end subroutine




subroutine chebders(x,n,pols,ders)
implicit double precision (a-h,o-z)

integer :: n 
double precision :: pols(n+1),ders(n+1)

!
!  Evaluate the Chebyshev polynomials order 0 through n and their derivatives
!  at a specified point using the standard 3-term recurrence relation.
!
!  Input parameters:
!
!    x - point at which to evaluate the polynomials
!    n - the order of the polynomials to evaluate
!
!  Output parameters:
!
!    pols - this user-supplied and allocated array of length n+1 will
!      contain the values of the polynomials of order 0 through n upon
!      return
!   ders - this user-supplied array of length n+1 will contain the values of
!      the derivative of orders 0 through n upon return
!

if (abs(x-1) .lt. 1.0d-16) then
pols = 1
do i=0,n
ders(i+1) = i**2
end do
return
endif

if (abs(x+1) .lt. 1.0d-16) then
do i=0,n/2

pols(2*i+1) = 1
ders(2*i+1) = -(2*i)**2

pols(2*i+2) = -1
ders(2*i+2) = (2*i+1)**2

end do
return
endif

if (n .lt. 0) return

pols(1) = 1.0d0
ders(1) = 0.0d0

if (n .eq. 0) return

pols(1) = 1.0d0
pols(2) = x
ders(1) = 0
ders(2) = 1

if (n .eq. 2) return

xx1 = 1.0d0
xx2 = x

do i=1,n-1
xx        = 2*x*xx2 - xx1
pols(i+2) = xx
xx1       = xx2
xx2       = xx
end do

!
!  Compute the derivatives
!

do i=2,n-1


if (abs(x-1) .lt.  cheby_eps00*10) then
ders(i+1) = 1.0d0
else if(abs(x+1) .lt. cheby_eps00*10) then
ders(i+1) = (-1.0d0)**(i-1) * i**2
else
ders(i+1) = i*(x*pols(i+1)-pols(i+2))/(1-x**2)
endif

end do


end subroutine




subroutine chebeval(a,b,n,xs,vals,x,val)
implicit double precision (a-h,o-z)

integer :: n
double precision ::  xs(n),vals(n),x,val

!
!  Use the barycentric formula to evaluate a function given its values at the n-point
!  Chebyshev grid on an interval [a,b].
!
!  Input parameters:
!
!    (a,b) - the interval on which the function is given
!    n - the number of nodes in the Chebyshev grid
!    xs - an array specifying the n Chevyshev node on the interval [-1,1]
!    vals - the values of the function on the n Chebyshev nodes on the
!      interval [-1,1]
!    x - the point in the interval (a,b) at which the function is to be
!      evaluated
!     
!  Output parameters:
!
!   val - the approximate value of the function at the point x
!

xx   = (2*x - (b+a) ) /(b-a)

sum1=0
sum2=0

dd1 = 1.0d0

do i=1,n
dd=1.0d0
if (i .eq. 1 .OR. i .eq. n) dd = 0.5d0

diff = xx-xs(i)

!
!  Handle the case in which the target node coincide with one of
!  of the Chebyshev nodes.
!


if(abs(diff) .le. cheby_eps00) then
val = vals(i)
return
endif

!
!  Otherwise, construct the sums.
!

dd   = (dd1*dd)/diff
dd1  = - dd1
sum1 = sum1+dd*vals(i)
sum2 = sum2+dd
dd   = - dd
end do

val = sum1/sum2


end subroutine


subroutine chebeval2(a,b,n,coefs,x,val)
implicit double precision (a-h,o-z)

integer, intent(in)             :: n
double precision, intent(in)    :: a,b,x,coefs(n)
double precision, intent(out)   :: val

!
!  Use the Clenshaw algorithm in order to evaluate a Chebyshev expansion on the 
!  interval [a,b].
!
!  Input parameters:
!    (a,b) - the interval on which the expansion is given
!    n - the number of terms in the Chebyshev expansion
!    coefs - an array of length n specifying the expansion coefficients
!    x - the point at which to evaluate the expansion
!
!  Output parameters:
!    val - the value of the expansion at the point x
!

xx = (x - (b+a)/2.0d0) * 2.0d0/(b-a)

b2 = coefs(n)
b1 = 2*xx*b2+coefs(n-1)

do i=n-2,2,-1
b0  = coefs(i)+2*xx*b1-b2
b2 = b1
b1 = b0
end do

val = b1 * xx + (coefs(1)-b2)

end subroutine


subroutine chebevalder(a,b,n,coefs,x,val,der)
implicit double precision (a-h,o-z)

integer, intent(in)             :: n
double precision, intent(in)    :: a,b,x,coefs(n)
double precision, intent(out)   :: val

!
!  Evaluate a Chebyshev expansion and its derivative at a point x in the interval 
!  [a,b].
!
!  Input parameters:
!    (a,b) - the interval on which the expansion is given
!    n - the number of terms in the expansion
!    coefs - an array of length n specifying the expansion coefficients
!    x - the point at which to evaluate the expansion
!
!  Output parameters:
!    val - the value of the expansion at the point x
!    der - the value of the derivative of the expansion at the point x
!

double precision :: pols(n),ders(n)

call chebders(x,n-1,pols,ders)

val = 0
der = 0
do i=1,n
val = val + coefs(i)*pols(i)
der = der + coefs(i)*pols(i)
end do

end subroutine


subroutine chebpw_eval(nints,ab,k,xscheb,vals,x,val)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nints,k
double precision, intent(in)  :: xscheb(k),ab(2,nints),vals(k,nints)
double precision, intent(out) :: val

!
!  Evaluate a function represented via its values at the nodes of the k-point
!  Chebyshev grids on a collection of subintervals of [a,b].
!
!  Input parameters:
!
!    (nints,ab) - arrays specifying the collection of subintervals of [a,b]
!    k - the number of terms in the Chebyshev expansions
!    xscheb - the nodes of the k-point Clenshaw-Curtis quadrature on [-1,1]
!    vals - a (k,nints) array the jth column of which gives the values of the
!      function at the nodes of the k-point Chebyshev grid in the jth
!      subinterval
!    x - the point in [a,b] at which to evaluate the function
!
!  Output parameters:
!    val - the value of the function at the point x
! 


!
!  Conduct several iterations of a binary search for the interval.
!

niters = 8
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
c     = ab(1,int)
if (x .gt. c) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do


!
!  Conduct a brute force check from here.
!

do int = intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do


!
!  Call chebeval to evaluate the expansion and the save the index
!  of the interval containing x.
!

a = ab(1,int)
b = ab(2,int)


! call chebeval(a,b,k,xscheb,vals(1,int),x,val)

xx   = (2*x - (b+a) ) /(b-a)

sum1=0
sum2=0

dd1 = 1.0d0

do i=1,k
dd=1.0d0
if (i .eq. 1 .OR. i .eq. k) dd = 0.5d0

diff = xx-xscheb(i)

!
!  Handle the case in which the target node coincide with one of
!  of the Chebyshev nodes.
!

if(abs(diff) .le. cheby_eps00 ) then
val = vals(i,int)
return
endif

!
!  Otherwise, construct the sums.
!

dd   = (dd1*dd)/diff
dd1  = - dd1
sum1 = sum1+dd*vals(i,int)
sum2 = sum2+dd
dd   = - dd
end do

val = sum1/sum2


end subroutine


subroutine chebpw_eval0(nints,ab,k,xscheb,vals,x,val,int0)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nints,k
double precision, intent(in)  :: xscheb(k),ab(2,nints),vals(k,nints)
double precision, intent(out) :: val

!
!  Evaluate a function represented via its values at the nodes of the k-point
!  Chebyshev grids on a collection of subintervals of [a,b].  In this version
!  of the routine, the user specifies an initial guess for the index of the
!  interval containing the point x.
!
!  Input parameters:
!
!    (nints,ab) - the subintervals
!    k - the number of Chebyshev nodes in each subnterval
!    xscheb - the nodes of the k-point Clenshaw-Curtis quadrature on [-1,1]
!    vals - a (k,nints) array the jth column of which gives the values of the
!      function at the nodes of the k-point Chebyshev grid in the jth
!      subinterval
!    x - the point in [a,b] at which to evaluate the function
!    int0 - a guess for the initial interval containing the point x
!
!  Output parameters:
!    val - the value of the function at the point x
! 

a = ab(1,int0)
b = ab(2,int0)
if (a .lt. x .AND. x .lt. b) then
int = int0
goto 1000
endif

!
!  Conduct several iterations of a binary search for the interval.
!

niters = 6
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
c     = ab(1,int)
if (x .gt. c) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do


!
!  Conduct a brute force check from here.
!

do int = intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do


!
!  Call chebeval to evaluate the expansion and the save the index
!  of the interval containing x.
!

a = ab(1,int)
b = ab(2,int)
int0 = int

! call chebeval(a,b,k,xscheb,vals(1,int),x,val)

1000 continue

xx   = (2*x - (b+a) ) /(b-a)

sum1=0
sum2=0

dd1 = 1.0d0

do i=1,k
dd=1.0d0
if (i .eq. 1 .OR. i .eq. k) dd = 0.5d0

diff = xx-xscheb(i)

!
!  Handle the case in which the target node coincide with one of
!  of the Chebyshev nodes.
!

if(abs(diff) .le. cheby_eps00 ) then
val = vals(i,int)
return
endif

!
!  Otherwise, construct the sums.
!

dd   = (dd1*dd)/diff
dd1  = - dd1
sum1 = sum1+dd*vals(i,int)
sum2 = sum2+dd
dd   = - dd
end do

val = sum1/sum2


end subroutine



subroutine chebpw_eval2(nints,ab,k,coefs,x,val)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nints,k
double precision, intent(in)  :: ab(2,nints),coefs(k,nints)
double precision, intent(out) :: val

!
!  Evaluate a function represented via piecewise chebyshev expansions on
!  a collection of subintervals.
!
!  Input parameters:
!    (nints,ab) - the 
!    k - an integer specifying the order of the Chebyshev expansions; on
!    
!    coefs - a (k,nints) array whose jth column specified the coefficients
!     of the function's Chebyshev expansion on the jth subinterval
!    x - the point at which to evaluate
!
!
!  Output parameters:
!    val - the value of the function at the point x
! 


double precision :: pols(k)
!
!  Conduct several iterations of a binary search for the interval.
!

niters = 6
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
c     = ab(1,int)
if (x .gt. c) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do


!
!  Conduct a brute force check from here.
!

do int = intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do

a = ab(1,int)
b = ab(2,int)

!
!  Evaluate the Chebyshev expansion
!


xx = (x - (b+a)/2.0d0) * 2.0d0/(b-a)
call chebs(xx,k-1,pols)

val = 0
do i=1,k
val = val + coefs(i,int)*pols(i)
end do

return

! call chebeval2(a,b,k,coefs(1,int),x,val)


xx = (x - (b+a)/2.0d0) * 2.0d0/(b-a)
xx2 = 2*xx

b2 = coefs(k,int)
b1 = xx2*b2+coefs(k-1,int)

do i=k-2,2,-1
b0  = coefs(i,int)+xx2*b1-b2
b2 = b1
b1 = b0
end do

val = b1 * xx + (coefs(1,int)-b2)

return
end subroutine


subroutine chebpw_eval20(nints,ab,k,coefs,x,val,int0)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nints,k
double precision, intent(in)  :: ab(2,nints),coefs(k,nints)
double precision, intent(out) :: val

!
!  Evaluate a piecewise Chebyshev expansion 
!
!     THIS VERSION ONLY EVALUATES 1/2 OF THE EXPANSION ... WHICH MIGHT
!     
!
!  Input parameters:
!    (nints,ab) - the 
!    k - the numbre of 
!    coefs - a (k,nints) array which specifies the coefficients of the
!      expansion
!    x - the point at which to evaluate
!
!
!  Output parameters:
!    val - the value of the function at the point x
! 


double precision :: pols(k)

a = ab(1,int0)
b = ab(2,int0)
if (a .lt. x .AND. x .lt. b) then
int = int0
goto 1000
endif

!
!  Conduct several iterations of a binary search for the interval.
!

niters = 6
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
c     = ab(1,int)
if (x .gt. c) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do


!
!  Conduct a brute force check from here.
!

do int = intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do

a = ab(1,int)
b = ab(2,int)
int0 = int
!
!  Evaluate the Chebyshev expansion
!

1000 continue

! xx = (x - (b+a)/2.0d0) * 2.0d0/(b-a)
! call chebs(xx,k-1,pols)

! val = 0
! do i=1,k
! val = val + coefs(i,int)*pols(i)
! end do

! return

! call chebeval2(a,b,k,coefs(1,int),x,val)


xx = (x - (b+a)/2.0d0) * 2.0d0/(b-a)
xx2 = 2*xx

kk = k/2+1

b2 = coefs(k,int)
b1 = xx2*b2+coefs(kk-1,int)


do i=kk-2,2,-1
b0  = coefs(i,int)+xx2*b1-b2
b2 = b1
b1 = b0
end do

val = b1 * xx + (coefs(1,int)-b2)

return
end subroutine


subroutine chebpw_evalder(nints,ab,k,coefs,x,val,der)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nints,k
double precision, intent(in)  :: ab(2,nints),coefs(k,nints)
double precision, intent(out) :: val

!
!  Evaluate a piecewise Chebyshev expansion and its derivative at an arbitrary
!  point on the interval on which it is given.
!
!  Input parameters:
!    (nints,ab) - the collection of subintervals
!    k - the number of terms in each of the Chebyshev expansions
!    coefs - a (k,nints) array specifying the coefficients in the expansion
!    x - the point at which to evaluate the expansion
!
!
!  Output parameters:
!    val - the value of the expansion at the point x
!    der - the value of the derivative of the expansion at the point x
! 


double precision :: pols(k+10),ders(k+10)

!
!  Conduct several iterations of a binary search for the interval.
!

niters = 8
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
c     = ab(1,int)
if (x .gt. c) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do


!
!  Conduct a brute force check from here.
!

do int = intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do

a = ab(1,int)
b = ab(2,int)

!
!  Evaluate the Chebyshev expansion
!

xx = (x - (b+a)/2 ) * 2/(b-a)
call chebders(xx,k-1,pols,ders)

! print *,"----------"
! print *,a
! print *,x
! print *,b
! print *,xx
! print *,"----------"

val = 0
der = 0
do i=1,k
val = val + coefs(i,int)*pols(i)
der = der + coefs(i,int)*ders(i)
end do

der = der*2/(b-a)

return
end subroutine



subroutine chebpw_evalder0(nints,ab,k,coefs,x,val,der,int0)
implicit double precision (a-h,o-z)

integer, intent(in)           :: nints,k
double precision, intent(in)  :: ab(2,nints),coefs(k,nints)
double precision, intent(out) :: val

!
!  Evaluate a piecewise Chebyshev expansion and its derivative at an arbitrary
!  point on the interval on which it is given.
!
!  Input parameters:
!    (nints,ab) - the collection of subintervals
!    k - the number of terms in each of the Chebyshev expansions
!    coefs - a (k,nints) array specifying the coefficients in the expansion
!    x - the point at which to evaluate the expansion
!
!
!  Output parameters:
!    val - the value of the expansion at the point x
!    der - the value of the derivative of the expansion at the point x
! 


double precision :: pols(k+10),ders(k+10)




a = ab(1,int0)
b = ab(2,int0)
if (a .lt. x .AND. x .lt. b) then
int = int0
goto 1000
endif
!
!  Conduct several iterations of a binary search for the interval.
!

niters = 6
intl   = 1
intr   = nints

do iter=1,niters
int   = (intl+intr)/2
c     = ab(1,int)
if (x .gt. c) then
intl = int
else
if (int .gt. 1) intr = int-1
endif
end do


!
!  Conduct a brute force check from here.
!

do int = intl,intr-1
b = ab(2,int)
if (x .le. b) exit
end do

int0 = int
a    = ab(1,int)
b    = ab(2,int)

!
!  Evaluate the Chebyshev expansion
!

1000 continue

xx = (x - (b+a)/2 ) * 2/(b-a)
call chebders(xx,k-1,pols,ders)

val = 0
der = 0
do i=1,k
val = val + coefs(i,int)*pols(i)
der = der + coefs(i,int)*ders(i)
end do

der = der*2/(b-a)

return
end subroutine


subroutine chebpw_plot(title,iplot,nints,ab,k,xs,vals)
implicit double precision (a-h,o-z)

integer, intent(in)          :: iplot,nints,k
double precision, intent(in) :: ab(2,nints),xs(k),vals(k,nints)
character*1 title(1)

!
!  Produce a GNUPLOT file called "gnuplot.???" where ??? is a specified
!  integer which contains commnds for plotting a function specified
!  by its values at the nodes of a discretization scheme.
!  
!  Input parameters:
!    title = an asterick-terminated character string which specifies the
!      title for the plot
!    iplot = the index of the GNUPLOT ouput file
!    (nints,ab) - the subintervals of the discretization scheme used to
!      represent the input function
!    k - the length of the Chebyshev grid on each interval
!    xs - the nodes of the k-point Chebyshev grid on the interval [-1,1]   
!    vals - a (k,nints) array specifying the values of the input function
!
!  Output parameters:
!
!    N/A  
!
!

double precision, allocatable :: ts(:,:)

allocate(ts(k,nints))

do int = 1,nints
a0 = ab(1,int)
b0 = ab(2,int)
do i   = 1,k
ts(i,int) = xs(i)*(b0-a0)/2 + (b0+a0)/2
end do
end do

call gnuplot_points(title,iplot,k*nints,ts,vals)

end subroutine

end module
