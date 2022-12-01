!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!
!  Construct Gauss-Laguerre quadrature formulas.  
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


module kummer_laguerre_routines

use utils
use chebyshev
use odesolve
use kummer

double precision :: dnu, dalpha

contains



subroutine gamma_ratio(a,n,val)
implicit double precision (a-h,o-z)

!
!  Approximate the quantity  Gamma ( a + 1+n) /  (Gamma(1+n) 
!  via an asymptotic formula.
!

dn = n 

val = (1+(a+a**2)/(2.q0*dn)+(3.4722222222222223q-3-  &
2.770432502787727q-3*(sqrt(6.283185307179586q0)+  &
6.q0*a*sqrt(6.283185307179586q0)+  &
6.q0*a**2*sqrt(6.283185307179586q0))+  &
1.3852162513938635q-3*(sqrt(6.283185307179586q0)-  &
1.2q1*a*sqrt(6.283185307179586q0)-  &
2.4q1*a**2*sqrt(6.283185307179586q0)+  &
2.4q1*a**3*sqrt(6.283185307179586q0)+  &
3.5999999999999996q1*a**4*sqrt(6.283185307179586q0)))/dn**2+  &
(2.681327160493827q-3+  &
1.154346876161553q-4*(sqrt(6.283185307179586q0)+  &
6.q0*a*sqrt(6.283185307179586q0)+  &
6.q0*a**2*sqrt(6.283185307179586q0))-  &
1.154346876161553q-4*(sqrt(6.283185307179586q0)-  &
1.2q1*a*sqrt(6.283185307179586q0)-  &
2.4q1*a**2*sqrt(6.283185307179586q0)+  &
2.4q1*a**3*sqrt(6.283185307179586q0)+  &
3.5999999999999996q1*a**4*sqrt(6.283185307179586q0))+  &
7.6956458410770185q-6*(-1.3900000000000001q2*sqrt(6.283185307179586q0) &
-2.7q2*a*sqrt(6.283185307179586q0)+  &
1.7100000000000002q3*a**2*sqrt(6.283185307179586q0)+  &
1.44q3*a**3*sqrt(6.283185307179586q0)-  &
2.7q3*a**4*sqrt(6.283185307179586q0)-  &
1.08q3*a**5*sqrt(6.283185307179586q0)+  &
1.08q3*a**6*sqrt(6.283185307179586q0)))/dn**3+  &
(-2.2947209362139915q-4+  &
8.914123099247549q-5*(sqrt(6.283185307179586q0)+  &
6.q0*a*sqrt(6.283185307179586q0)+  &
6.q0*a**2*sqrt(6.283185307179586q0))+  &
4.809778650673138q-6*(sqrt(6.283185307179586q0)-  &
1.2q1*a*sqrt(6.283185307179586q0)-  &
2.4q1*a**2*sqrt(6.283185307179586q0)+  &
2.4q1*a**3*sqrt(6.283185307179586q0)+  &
3.5999999999999996q1*a**4*sqrt(6.283185307179586q0))-  &
6.413038200897516q-7*(-1.3900000000000001q2*sqrt(6.283185307179586q0)  &
-2.7q2*a*sqrt(6.283185307179586q0)+  &
1.7100000000000002q3*a**2*sqrt(6.283185307179586q0)+  &
1.44q3*a**3*sqrt(6.283185307179586q0)-  &
2.7q3*a**4*sqrt(6.283185307179586q0)-  &
1.08q3*a**5*sqrt(6.283185307179586q0)+  &
1.08q3*a**6*sqrt(6.283185307179586q0))+  &
1.603259550224379q-7*(-5.709999999999999q2*sqrt(6.283185307179586q0)  &
+1.6680000000000001q4*a*sqrt(6.283185307179586q0)+  &
1.2864q4*a**2*sqrt(6.283185307179586q0)-  &
7.271999999999999q4*a**3*sqrt(6.283185307179586q0)-  &
2.268q4*a**4*sqrt(6.283185307179586q0)+  &
7.862400000000001q4*a**5*sqrt(6.283185307179586q0)-  &
2.5919999999999996q4*a**7*sqrt(6.283185307179586q0)+  &
6.48q3*a**8*sqrt(6.283185307179586q0)))/dn**4+  &
(-7.840392217200666q-4-  &
7.628843359817671q-6*(sqrt(6.283185307179586q0)+  &
6.q0*a*sqrt(6.283185307179586q0)+  &
6.q0*a**2*sqrt(6.283185307179586q0))+  &
3.714217958019812q-6*(sqrt(6.283185307179586q0)-  &
1.2q1*a*sqrt(6.283185307179586q0)-  &
2.4q1*a**2*sqrt(6.283185307179586q0)+  &
2.4q1*a**3*sqrt(6.283185307179586q0)+  &
3.5999999999999996q1*a**4*sqrt(6.283185307179586q0))+  &
2.6720992503739653q-8*(-1.3900000000000001q2*sqrt(6.283185307179586q0)&
-2.7q2*a*sqrt(6.283185307179586q0)+  &
1.7100000000000002q3*a**2*sqrt(6.283185307179586q0)+  &
1.44q3*a**3*sqrt(6.283185307179586q0)-  &
2.7q3*a**4*sqrt(6.283185307179586q0)-  &
1.08q3*a**5*sqrt(6.283185307179586q0)+  &
1.08q3*a**6*sqrt(6.283185307179586q0))-  &
1.3360496251869827q-8*(-5.709999999999999q2*sqrt(6.283185307179586q0) &
+1.6680000000000001q4*a*sqrt(6.283185307179586q0)+  &
1.2864q4*a**2*sqrt(6.283185307179586q0)-  &
7.271999999999999q4*a**3*sqrt(6.283185307179586q0)-  &
2.268q4*a**4*sqrt(6.283185307179586q0)+  &
7.862400000000001q4*a**5*sqrt(6.283185307179586q0)-  &
2.5919999999999996q4*a**7*sqrt(6.283185307179586q0)+  &
6.48q3*a**8*sqrt(6.283185307179586q0))+  &
1.9086423216956894q-9*(1.63879q5*sqrt(6.283185307179586q0)+  &
1.67874q5*a*sqrt(6.283185307179586q0)-  &
2.475942q6*a**2*sqrt(6.283185307179586q0)-  &
7.93632q5*a**3*sqrt(6.283185307179586q0)+  &
5.615063999999999q6*a**4*sqrt(6.283185307179586q0)+  &
1.11888q5*a**5*sqrt(6.283185307179586q0)-  &
4.170096q6*a**6*sqrt(6.283185307179586q0)+  &
9.43488q5*a**7*sqrt(6.283185307179586q0)+  &
9.525599999999999q5*a**8*sqrt(6.283185307179586q0)-  &
4.536q5*a**9*sqrt(6.283185307179586q0)+  &
5.4432q4*a**10*sqrt(6.283185307179586q0)))/dn**5+  &
(6.972813758365857q-5-  &
2.606553291976399q-5*(sqrt(6.283185307179586q0)+  &
6.q0*a*sqrt(6.283185307179586q0)+  &
6.q0*a**2*sqrt(6.283185307179586q0))-  &
3.1786847332573624q-7*(sqrt(6.283185307179586q0)-  &
1.2q1*a*sqrt(6.283185307179586q0)-  &
2.4q1*a**2*sqrt(6.283185307179586q0)+  &
2.4q1*a**3*sqrt(6.283185307179586q0)+  &
3.5999999999999996q1*a**4*sqrt(6.283185307179586q0))+  &
2.0634544211221173q-8*(-1.3900000000000001q2*sqrt(6.283185307179586q0)&
-2.7q2*a*sqrt(6.283185307179586q0)+  &
1.7100000000000002q3*a**2*sqrt(6.283185307179586q0)+  &
1.44q3*a**3*sqrt(6.283185307179586q0)-  &
2.7q3*a**4*sqrt(6.283185307179586q0)-  &
1.08q3*a**5*sqrt(6.283185307179586q0)+  &
1.08q3*a**6*sqrt(6.283185307179586q0))+  &
5.566873438279094q-10*(-5.709999999999999q2*sqrt(6.283185307179586q0)&
+1.6680000000000001q4*a*sqrt(6.283185307179586q0)+  &
1.2864q4*a**2*sqrt(6.283185307179586q0)-  &
7.271999999999999q4*a**3*sqrt(6.283185307179586q0)-  &
2.268q4*a**4*sqrt(6.283185307179586q0)+  &
7.862400000000001q4*a**5*sqrt(6.283185307179586q0)-  &
2.5919999999999996q4*a**7*sqrt(6.283185307179586q0)+  &
6.48q3*a**8*sqrt(6.283185307179586q0))-  &
1.590535268079741q-10*(1.63879q5*sqrt(6.283185307179586q0)+  &
1.67874q5*a*sqrt(6.283185307179586q0)-  &
2.475942q6*a**2*sqrt(6.283185307179586q0)-  &
7.93632q5*a**3*sqrt(6.283185307179586q0)+  &
5.615063999999999q6*a**4*sqrt(6.283185307179586q0)+  &
1.11888q5*a**5*sqrt(6.283185307179586q0)-  &
4.170096q6*a**6*sqrt(6.283185307179586q0)+  &
9.43488q5*a**7*sqrt(6.283185307179586q0)+  &
9.525599999999999q5*a**8*sqrt(6.283185307179586q0)-  &
4.536q5*a**9*sqrt(6.283185307179586q0)+  &
5.4432q4*a**10*sqrt(6.283185307179586q0))+  &
5.30178422693247q-12*(5.246819q6*sqrt(6.283185307179586q0)-  &
2.6548398q8*a*sqrt(6.283185307179586q0)-  &
1.0647972q8*a**2*sqrt(6.283185307179586q0)+  &
1.35715356q9*a**3*sqrt(6.283185307179586q0)+  &
9.858618q7*a**4*sqrt(6.283185307179586q0)-  &
1.87642224q9*a**5*sqrt(6.283185307179586q0)+  &
3.0669408q8*a**6*sqrt(6.283185307179586q0)+  &
9.708336q8*a**7*sqrt(6.283185307179586q0)-  &
3.7871064q8*a**8*sqrt(6.283185307179586q0)-  &
1.3372128q8*a**9*sqrt(6.283185307179586q0)+  &
1.0777536q8*a**10*sqrt(6.283185307179586q0)-  &
2.286144q7*a**11*sqrt(6.283185307179586q0)+  &
1.63296q6*a**12*sqrt(6.283185307179586q0)))/dn**6+  &
(5.921664373536939q-4+  &
2.3181251846474664q-6*(sqrt(6.283185307179586q0)+  &
6.q0*a*sqrt(6.283185307179586q0)+  &
6.q0*a**2*sqrt(6.283185307179586q0))-  &
1.086063871656833q-6*(sqrt(6.283185307179586q0)-  &
1.2q1*a*sqrt(6.283185307179586q0)-  &
2.4q1*a**2*sqrt(6.283185307179586q0)+  &
2.4q1*a**3*sqrt(6.283185307179586q0)+  &
3.5999999999999996q1*a**4*sqrt(6.283185307179586q0))-  &
1.765935962920757q-9*(-1.3900000000000001q2*sqrt(6.283185307179586q0) &
-2.7q2*a*sqrt(6.283185307179586q0)+  &
1.7100000000000002q3*a**2*sqrt(6.283185307179586q0)+  &
1.44q3*a**3*sqrt(6.283185307179586q0)-  &
2.7q3*a**4*sqrt(6.283185307179586q0)-  &
1.08q3*a**5*sqrt(6.283185307179586q0)+  &
1.08q3*a**6*sqrt(6.283185307179586q0))+  &
4.298863377337745q-10*(-5.709999999999999q2*sqrt(6.283185307179586q0) &
+1.6680000000000001q4*a*sqrt(6.283185307179586q0)+  &
1.2864q4*a**2*sqrt(6.283185307179586q0)-  &
7.271999999999999q4*a**3*sqrt(6.283185307179586q0)-  &
2.268q4*a**4*sqrt(6.283185307179586q0)+  &
7.862400000000001q4*a**5*sqrt(6.283185307179586q0)-  &
2.5919999999999996q4*a**7*sqrt(6.283185307179586q0)+  &
6.48q3*a**8*sqrt(6.283185307179586q0))+  &
6.627230283665588q-12*(1.63879q5*sqrt(6.283185307179586q0)+  &
1.67874q5*a*sqrt(6.283185307179586q0)-  &
2.475942q6*a**2*sqrt(6.283185307179586q0)-  &
7.93632q5*a**3*sqrt(6.283185307179586q0)+  &
5.615063999999999q6*a**4*sqrt(6.283185307179586q0)+  &
1.11888q5*a**5*sqrt(6.283185307179586q0)-  &
4.170096q6*a**6*sqrt(6.283185307179586q0)+  &
9.43488q5*a**7*sqrt(6.283185307179586q0)+  &
9.525599999999999q5*a**8*sqrt(6.283185307179586q0)-  &
4.536q5*a**9*sqrt(6.283185307179586q0)+  &
5.4432q4*a**10*sqrt(6.283185307179586q0))-  &
4.418153522443725q-13*(5.246819q6*sqrt(6.283185307179586q0)-  &
2.6548398q8*a*sqrt(6.283185307179586q0)-  &
1.0647972q8*a**2*sqrt(6.283185307179586q0)+  &
1.35715356q9*a**3*sqrt(6.283185307179586q0)+  &
9.858618q7*a**4*sqrt(6.283185307179586q0)-  &
1.87642224q9*a**5*sqrt(6.283185307179586q0)+  &
3.0669408q8*a**6*sqrt(6.283185307179586q0)+  &
9.708336q8*a**7*sqrt(6.283185307179586q0)-  &
3.7871064q8*a**8*sqrt(6.283185307179586q0)-  &
1.3372128q8*a**9*sqrt(6.283185307179586q0)+  &
1.0777536q8*a**10*sqrt(6.283185307179586q0)-  &
2.286144q7*a**11*sqrt(6.283185307179586q0)+  &
1.63296q6*a**12*sqrt(6.283185307179586q0))+  &
4.418153522443725q-13*(-5.34703531q8*sqrt(6.283185307179586q0)  &
-3.46290054q8*a*sqrt(6.283185307179586q0)+  &
8.792452254q9*a**2*sqrt(6.283185307179586q0)+  &
1.28061792q9*a**3*sqrt(6.283185307179586q0)-  &
2.27124729q10*a**4*sqrt(6.283185307179586q0)+  &
1.955830968q9*a**5*sqrt(6.283185307179586q0)+  &
2.0837817q10*a**6*sqrt(6.283185307179586q0)-  &
6.10841088q9*a**7*sqrt(6.283185307179586q0)-  &
7.54933608q9*a**8*sqrt(6.283185307179586q0)+  &
4.07165616q9*a**9*sqrt(6.283185307179586q0)+  &
4.2810768q8*a**10*sqrt(6.283185307179586q0)-  &
7.9252991999999995q8*a**11*sqrt(6.283185307179586q0)+  &
2.3351328q8*a**12*sqrt(6.283185307179586q0)-  &
2.9393279999999997q7*a**13*sqrt(6.283185307179586q0)+  &
1.39968q6*a**14*sqrt(6.283185307179586q0)))/dn**7)/(1/dn)**(1.q0*a)


end subroutine




subroutine laguerre_taylor(a,t,n,val,der)
implicit double precision (a-h,o-z)

!
!  Evaluate  L_n^a (t) Exp[-t/2] * t^(a/2) for a small value of t.
!
dn = n

val = t**(a/2.q0)*(1+(-5.q-1-(1.q0*dn)/(1+a))*t+(1.25q-1+  &
dn/(2.q0*(1+a))+((-1+dn)*dn)/(2.q0*(1+a)*(2+a)))*t**2+  &
(-2.083333333333333q-2-dn/(8.q0*(1+a))-((-1+dn)*dn)/(4.q0*(1  &
+a)*(2+a))-((-2+dn)*(-1+dn)*dn)/(6.q0*(1+a)*(2+a)*(3+  &
a)))*t**3)

der = t**(a/2.q0)*(-((2+a)*(1+a+2.q0*dn))/(4.q0*(1+a))+a/(2.q0*t)+  &
((4+a)*(2+3.q0*a+a**2+4.q0*dn+4.q0*a*dn+  &
4.q0*dn**2)*t)/(1.6q1*(1+a)*(2+a))-((6+a)*(6+1.1q1*a+  &
6.q0*a**2+a**3+1.6q1*dn+1.7999999999999998q1*a*dn+  &
6.q0*a**2*dn+1.2q1*dn**2+1.2q1*a*dn**2+  &
8.q0*dn**3)*t**2)/(9.6q1*(1+a)*(2+a)*(3+a))+((192+4.24q2*a+  &
3.3000000000000003q2*a**2+1.1500000000000001q2*a**3+  &
1.7999999999999998q1*a**4+a**5+5.12q2*dn+8.96q2*a*dn+  &
4.88q2*a**2*dn+1.12q2*a**3*dn+8.q0*a**4*dn+6.4q2*dn**2+  &
6.5600000000000005q2*a*dn**2+2.64q2*a**2*dn**2+  &
2.4q1*a**3*dn**2+2.56q2*dn**3+2.88q2*a*dn**3+  &
3.2q1*a**2*dn**3+1.28q2*dn**4+  &
1.6q1*a*dn**4)*t**3)/(7.68q2*(1+a)*(2+a)*(3+a)*(4+a)))

call gamma_ratio(a,n,dd)
val = val *dd / gamma(a+1.0d0)
der = der *dd / gamma(a+1.0d0)

end subroutine



subroutine laguerreq(t,val)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: t
double precision, intent(out) :: val


val = exp(t)/2 -0.25d0*(dalpha-exp(t))**2+exp(t)*dnu

end subroutine


subroutine laguerreq2(t,val)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: t
double precision, intent(out) :: val


val = 2.0d0 + 2*dalpha + 4*dnu +1.0d0/(4*t**2) - t**2 - dalpha**2/t**2

end subroutine


end module


program kummer_laguerre

use utils
use odesolve
use kummer
use kummer_laguerre_routines

implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),chebintl(:,:),chebintr(:,:), &
   ucheb(:,:),vcheb(:,:)

double precision, allocatable :: ab(:,:),alpha(:,:),alphap(:,:),alphapp(:,:)
double precision, allocatable :: alphainv(:,:),alphainvp(:,:),abinv(:,:)

double precision, allocatable :: ab2(:,:),alpha2(:,:),alphap2(:,:),alphapp2(:,:)
double precision, allocatable :: alphainv2(:,:),alphainvp2(:,:),abinv2(:,:)

double precision, allocatable :: nodes(:),weights(:),nodes0(:),weights0(:)

character*32 arg,filename


pi = acos(-1.0d0)
call mach_zero(eps0)

!
!  Set the wavenumber for the problem.
!

dnu     = 10**7
dalpha  = sqrt(pi)/4
norder  = dnu

call prin2("dalpha = ",dalpha)
call prin2("dnu = ",dnu)

!
!  Fetch the Chebyshev quadrature and related matrices.
!

k = 24
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)

call elapsed(t1)
call elapsed(tt1)

!
!  Set upper and lower bounds for the roots of L_n.
!

call prinl("in kummer_laguerre, norder = ",norder)

if (eps0 .lt. 1.0d-20) then
eps    = 1.0d-20
else
eps    = 1.0d-13
endif

!
!  Construct the first phase function, it inverse, and find the coefficients c_1 and c_2z
!

a      = -20.0d0
b      =  0.0d0
ifleft = 1

call kummer_adap(eps,a,b,laguerreq,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)

call kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alpha,alphap,alphapp)

call kummer_phase_inverse(nints,ab,k,xscheb,chebintl,ucheb,alpha,alphap, &
   nintsinv,abinv,alphainv,alphainvp)

call laguerre_taylor(dalpha,exp(a),norder,val,der)


ya  = val 
ypa = der*exp(a)

call kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,ya,ypa,c1,c2)

call prind("after kummer_coefs, a  = ",a)
call prind("after kummer_coefs, ya  = ",ya)
call prind("after kummer_coefs, ypa  = ",ypa)
call prind("after kummer_coefs, c1  = ",c1)
call prind("after kummer_coefs, c2  = ",c2)



!
!  Find the last root and compute the value of y' at that point
!

call kummer_zeros_count(nints,ab,k,xscheb,alpha,c1,c2,nroots)


xx = pi*nroots - c2
call chebpw_eval(nintsinv,abinv,k,xscheb,alphainv,xx,u0)
call chebpw_eval(nints,ab,k,xscheb,alpha,u0,aval)
call chebpw_eval(nints,ab,k,xscheb,alphap,u0,apval)

root = exp(u0)
der  = c1 * (-1)**nroots * sqrt(apval)
val  = c1 * sin(aval+c2)/sqrt(apval)

!
! ... and the second phase function, slightly overlapping
!


a2     = exp(u0/2)

b2     = sqrt((2*dnu**2 + dalpha*dnu - dnu + 2*dalpha + 2 + 2 *(dnu-1) * &
         sqrt(dnu**2 + (dnu+1)*(dalpha+1))) / (dnu+2))


call kummer_adap(eps,a2,b2,laguerreq2,k,xscheb,chebintl,chebintr,ucheb, &
   nints2,ab2,alphap2,alphapp2)

call kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints2,ab2,alpha2,alphap2,alphapp2)

call kummer_phase_inverse(nints2,ab2,k,xscheb,chebintl,ucheb,alpha2,alphap2, &
   nintsinv2,abinv2,alphainv2,alphainvp2)

ya  = 0.0d0
ypa = 2* exp(-0.25d0*u0)*der

call kummer_coefs(ifleft,nints2,ab2,k,xscheb,alphap2,alphapp2,ya,ypa,d1,d2)

call prind("after kummer_coefs, d1  = ",d1)
call prind("after kummer_coefs, d2  = ",d2)

call kummer_zeros_count(nints2,ab2,k,xscheb,alpha2,d1,d2,nroots2)
call prini("after kummer_adap, nints = ",nints)
call prini("after kummer_adap, nints2 = ",nints2)

call prinl("number of roots  = ",nroots + nroots2)

call elapsed(tt2)

t_phase =tt2-tt1


call gamma_ratio(dalpha,norder,dconst)
call prind("in kummer_lageurre, dconst = ",dconst)

allocate(nodes(norder),weights(norder))
allocate(nodes0(norder),weights0(norder))

! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! !  Time the computation of the roots
! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

call elapsed(tt1)
int0 = 1
int1 = 1

do i=1,nroots
xx = pi*i - c2
call chebpw_eval0(nintsinv,abinv,k,xscheb,alphainv,xx,root,int0)
call chebpw_eval0(nints,ab,k,xscheb,alphap,root,apval,int1)

x   = exp(root)
wht = exp(-x)*x**(1+dalpha)/(c1**2 * apval)

end do



int0 = 1
int1 = 1

do i=1,nroots2

xx = pi*(i+1) - d2

call chebpw_eval0(nintsinv2,abinv2,k,xscheb,alphainv2,xx,root,int0)
call chebpw_eval0(nints2,ab2,k,xscheb,alphap2,root,apval,int1)

x   = root**2
wht = 4*exp(-x)*x**(0.50d0+dalpha)/(d1**2 * apval)

end do

call elapsed(tt2)
t_quad = tt2-tt1

call elapsed(t2)
t_total = t2-t1


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  Store the formula and normalize the weights
!  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


int0 = 1
int1 = 1

dsum = 0

do i=1,nroots

xx = pi*i - c2

call chebpw_eval0(nintsinv,abinv,k,xscheb,alphainv,xx,root,int0)
call chebpw_eval0(nints,ab,k,xscheb,alphap,root,apval,int1)

x   = exp(root)
wht = dconst*exp(-x)*x**(1+dalpha)/(c1**2 * apval)

nodes(i)    = x
weights(i) = wht

dsum = dsum + wht
end do


int0 = 1
int1 = 1

do i=1,nroots2

xx = pi*(i+1) - d2

call chebpw_eval0(nintsinv2,abinv2,k,xscheb,alphainv2,xx,root,int0)
call chebpw_eval0(nints2,ab2,k,xscheb,alphap2,root,apval,int1)

x   = root**2
wht = 4*dconst*exp(-x)*x**(0.5d0+dalpha)/(d1**2 * apval)

nodes(i+nroots)    = x
weights(i+nroots)  = wht

end do

dsum = sum(weights)



!
!  Normalize weights
!
weights = weights*gamma(1.0d0+dalpha)/dsum


!
!  Perform a crude error check by approximating an integral
!

sum1 =0
do i=1,norder

x   = nodes(i)   
wht = weights(i) 

sum1 = sum1 + wht*cos(x)
end do

sum0 = 2**(-0.5 - dalpha/2.)*Cos(((1 + dalpha)*Pi)/4.)*Gamma(1 + dalpha)

call prin2("error in integral = ",sum1-sum0)
call prina("")


call prina("")
call prini("norder = ",norder)
call prin2("phase time = ",t_phase)
call prin2("quadrature time = ",t_quad)
call prini("phase function expansion size = ",(k-1) * (nints + nints2)+1)
call prin2("total time = ",t_total)
call prina("")



end program
