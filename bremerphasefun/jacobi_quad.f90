module kummer_jacobi_routines

use utils
use chebyshev

double precision :: dnu, da, db, dconst

contains


subroutine gammaratio1(val,da0)
implicit double precision (a-h,o-z)

!
!  Approximate the value of 
!
!         gamma ( 1 + dnu + da0 )
!         ----------------------------
!         gamma ( 1 + dnu) gamma (1+da0 )
!      
!  with around 10^(-16) accuracy in the event that dnu is large (>10^3) 
!  and -1 < da0 < 1.
! 
!

val =    (1/Gamma(1 + da0) + (48*da0 + 20*da0**2 - 180*da0**3 - 25*da0**4 + 192*da0**5 - & 
         10*da0**6- 60*da0**7 + 15*da0**8)/(5760.*dnu**4*Gamma(1 + da0)) + (2*da0**2 +   &
         da0**3 - 3*da0**4 - da0**5 + da0**6)/(48.*dnu**3*Gamma(1 + da0)) + (-2*da0 -    &
         3*da0**2 + 2*da0**3 + 3*da0**4)/(24.*dnu**2*Gamma(1 + da0)) + (da0 + da0**2)    &
         /(2.*dnu*Gamma(1 + da0)))/(1/dnu)**da0

end subroutine



subroutine gammaratio2(val)
implicit double precision (a-h,o-z)

!
!  Approximate the value of 
!
!                      gamma ( dnu+da+1 ) gamma(dnu+db+a)
!   2^(da+db+1)      --------------------------------------
!                     gamma( 1 + dnu)  gamma(dnu+da+db+1)
!      
!  with around 10^(-16) accuracy in the event that dnu is large (>10^3) 
!  and -1 < da < 1 and -1 < db < 1.
!



val = 0
nn  = 10

do m=1,nn

d1 = -da
d2 = -db
d3 = -da-db-dnu

do i=1,m-1
d1 = d1 * (-da+i)
d2 = d2 * (-db+i)
d3 = d3 * (-da-db-dnu+i)
end do

dterm = d1*d2/(d3*gamma(m+1.0d0))

val= val + dterm

end do

val = (val+1) * 2.0d0**(da+db+1.0d0)


end subroutine



subroutine jacobitaylor(t,val,der,da0,db0)
implicit double precision (a-h,o-z)

!
!  Approximate the function
!
!       (da0,db0)
!      P         (cos(t)) r(t)
!       dnu
!
!  and its derivative, where
!
!        r(t) = cos(t/2)^ ( (db0-da0)/2 )  sin(t/2)^ ( (da0-db0)/2)  sin(t)^((1+da0+db0)/2)
!
!  when t is close to 0.
!

call gammaratio1(val0,da0)


val = t**da0*(2**((-da0 + db0)/2.)*Sqrt(t)*val0 - (2**(-3 - da0/2. + db0/2.)*               &
     (2 + 3*da0 + da0**2 + 3*db0 + 3*da0*db0 + 6*dnu + 6*da0*dnu + 6*db0*dnu + 6*dnu**2)*   &
     t**2.5*val0)/(3.*(1 + da0)) + (2**(-7 - da0/2. + db0/2.)*(8 + 48*da0 + 68*da0**2 +     &
      33*da0**3 + 5*da0**4 + 60*db0 + 150*da0*db0 + 120*da0**2*db0 + 30*da0**3*db0 +        &
      90*db0**2 + 135*da0*db0**2 + 45*da0**2*db0**2 + 120*dnu + 300*da0*dnu + 240*da0**2*   &
      dnu + 60*da0**3*dnu + 300*db0*dnu + 540*da0*db0*dnu + 240*da0**2*db0*dnu +            &
      180*db0**2*dnu + 180*da0*db0**2*dnu + 300*dnu**2 + 540*da0*dnu**2 +                   &
              240*da0**2*dnu**2 + 540*db0*dnu**2 + 540*da0*db0*dnu**2 + 180*db0**2*dnu**2 + &
      360*dnu**3 + 360*da0*dnu**3 + 360*db0*dnu**3 + 180*dnu**4)*t**4.5*val0)/(45.*         &
      (1 + da0)*(2 + da0)))

der =  t**da0*((2**(-1 - da0/2. + db0/2.)*(1 + 2*da0)*val0)/Sqrt(t) - (2**(-4 - da0/2. +      &
       db0/2.)*(5 + 2*da0)*(2 + 3*da0 + da0**2 + 3*db0 + 3*da0*db0 + 6*dnu + 6*da0*dnu +      &
       6*db0*dnu + 6*dnu**2)*t**1.5*val0)/(3.*(1 + da0)) + t**3.5*((2**(-4 - da0/2. +         & 
       db0/2.)*dnu*(1 + da0 + db0 + dnu)*(2*da0 + da0**2 + 3*da0*db0 + 6*dnu + 6*da0*dnu +    &
       6*db0*dnu + 6*dnu**2)*val0)/(3.*(1 + da0)*(2 + da0)) - (2**(-6 - da0/2. + db0/2.)*     &
       (da0 - db0)*(da0 + da0**2 + 3*db0 + 3*da0*db0 + 6*dnu + 6*da0*dnu + 6*db0*dnu +        &
       6*dnu**2)*val0)/(3.*(1 + da0)) + (2**(-8 - da0/2. + db0/2.)*(da0 - db0)*               &
       (72 + 184*da0 + 160*da0**2 + 53*da0**3 + 5*da0**4 + 180*db0 + 330*da0*db0 +            &
       180*da0**2*db0 + 30*da0**3*db0 + 90*db0**2 + 135*da0*db0**2 + 45*da0**2*db0**2 +       &
       360*dnu + 660*da0*dnu + 360*da0**2*dnu + 60*da0**3*dnu + 540*db0*dnu +                 &
       660*da0*db0*dnu + 240*da0**2*db0*dnu + 180*db0**2*dnu + 180*da0*db0**2*dnu +           &
       540*dnu**2 + 660*da0*dnu**2 + 240*da0**2*dnu**2 + 540*db0*dnu**2 + 540*da0*db0*        &
       dnu**2 + 180*db0**2*dnu**2 + 360*dnu**3 + 360*da0*dnu**3 + 360*db0*dnu**3 +            &
       180*dnu**4)*val0)/(45.*(1 + da0)*(2 + da0)) + (2**(-8 - da0/2. + db0/2.)*              &
       (1 + da0 + db0)*(72 + 304*da0 + 340*da0**2 + 113*da0**3 + 5*da0**4 + 540*db0 +         &
       870*da0*db0 + 360*da0**2*db0 + 30*da0**3*db0 + 90*db0**2 + 135*da0*db0**2 +            &
       45*da0**2*db0**2 + 1080*dnu + 1740*da0*dnu + 720*da0**2*dnu + 60*da0**3*dnu +          &
       1260*db0*dnu + 1020*da0*db0*dnu + 240*da0**2*db0*dnu + 180*db0**2*dnu +                &
       180*da0*db0**2*dnu + 1260*dnu**2 + 1020*da0*dnu**2 + 240*da0**2*dnu**2 +               &
       540*db0*dnu**2 + 540*da0*db0*dnu**2 + 180*db0**2*dnu**2 + 360*dnu**3 + 360*da0*dnu**3+ &
                 360*db0*dnu**3 + 180*dnu**4)*val0)/(45.*(1 + da0)*(2 + da0))))


! Lower order approximations, which loose a bit too much accuracy for negative da0
! and large dnu.

! val =  t**da0*(2**((-da0 + db0)/2.)*Sqrt(t)*val0 - (2**(-3 - da0/2. + db0/2.)*                 &
!        (2 + 3*da0 + da0**2 + 3*db0 + 3*da0*db0 + 6*dnu + 6*da0*dnu + 6*db0*dnu + 6*dnu**2)     &
!        *t**2.5*val0)/(3.*(1 + da0)))


! der =  t**da0*((2**(-1 - da0/2. + db0/2.)*(1 + 2*da0)*val0)/Sqrt(t) - (2**(-4 - da0/2. +       &
!        db0/2.)*(5 + 2*da0)*(2 + 3*da0 + da0**2 + 3*db0 + 3*da0*db0 + 6*dnu + 6*da0*dnu +       &
!        6*db0*dnu + 6*dnu**2)*t**1.5*val0)/(3.*(1 + da0)))


end subroutine




subroutine evalr(t,val,der,da0,db0)
implicit double precision (a-h,o-z)

!
!  Evaluate the function r(t) and its derivative.
!

if (t .lt. .001d0) then

val = t**da0*(-(2**(-3-da0/2.q0+db0/2.q0)*(2+da0+  &
3.q0*db0)*t**2.5q0)/3.q0+(2**(-7-da0/2.q0+db0/2.q0)*(4+  &
1.7999999999999998q1*da0+5.q0*da0**2+3.q1*db0+3.q1*da0*db0+  &
4.5q1*db0**2)*t**4.5q0)/4.5q1+2**((-1.q0*da0+  &
db0)/2.q0)*sqrt(t))

der = t**da0*((2**(-1-da0/2.q0+db0/2.q0)*(1+2.q0*da0))/Sqrt(t)-  &
(2**(-4-da0/2.q0+db0/2.q0)*(5+2.q0*da0)*(2+da0+  &
3.q0*db0)*t**1.5q0)/3.q0+(2**(-8-da0/2.q0+db0/2.q0)*(9+  &
2.q0*da0)*(4+1.7999999999999998q1*da0+5.q0*da0**2+3.q1*db0+  &
3.q1*da0*db0+4.5q1*db0**2)*t**3.5q0)/4.5q1)

else

val = cos(t/2)**((db0-da0)/2) * sin(t/2)**( (da0-db0)/2)  * sin(t)**((da0+db0+1)/2)


der = (Cos(t/2.)**((-2 - da0 + db0)/2.)*(da0 - db0 + &
        (1 + da0 + db0)*Cos(t))*Sin(t/2.)**((-2 + da0 - db0)/2.)*Sin(t)**((1 + da0 + db0)/2.))/4.0d0

endif

end subroutine



subroutine jacobipt(norder,nroots,nints,nintsinv,abinv,ab,k,xscheb,alphainv,alphap,a1,a2, &
  nints2,nints2inv,abinv2,ab2,alphainv2,alphap2,b1,b2,i,root,wht)
implicit double precision (a-h,o-z)

double precision, intent(in)  :: abinv(2,nints), ab(2,nints), xscheb(k)
double precision, intent(in)  :: abinv2(2,nints2), ab2(2,nints2)
double precision, intent(in)  :: alphainv(k,nints),alphap(k,nints)
double precision, intent(in)  :: alphainv2(k,nints2),alphap2(k,nints2)

double precision, intent(out) :: root,wht
integer, intent(in)           :: norder,nroots

double precision              :: pi


data pi        / 3.14159265358979323846264338327950288d0 / 
data dscale    / 100.499170807130528801066368660787514d0 /
data twooverpi / 0.636619772367581343075535053490057448d0/



if (i .le. nroots) then
j = i
xx = pi *j-b2

call chebpw_eval(nints2inv,abinv2,k,xscheb,alphainv2,xx,root)
call chebpw_eval(nints2,ab2,k,xscheb,alphap2,root,apval)

call evalr(root,rval,rder,db,da)
wht = dconst/(b1**2*apval) * rval**2

else

j = norder-i+1


xx = pi *j-a2


call chebpw_eval(nintsinv,abinv,k,xscheb,alphainv,xx,root)
call chebpw_eval(nints,ab,k,xscheb,alphap,root,apval)


call evalr(root,rval,rder,da,db)

wht = dconst/(a1**2*apval) * rval**2


endif

if (i .le. nroots) then
root = -cos(root)
else
root = cos(root)
endif

end subroutine





subroutine jacobiq0(t,val,da0,db0)
implicit double precision (a-h,o-z)
double precision, intent(in)   :: t,da0,db0
double precision, intent(out)  :: val


val =  dnu*(1 + da0 + db0 + dnu) + ((1 + da0 + db0 + (da0 - db0)*Cos(t))*1.0d0/sin(t)**2)/2. - &
       ((da0 - db0 + (1 + da0 + db0)*Cos(t))**2*1.0d0/sin(t)**2)/4.0d0

if (abs(t) .lt. 1.0d-3) then
val =   (2 + 3*da0 + da0**2 + 3*db0 + 3*da0*db0 + 6*dnu + 6*da0*dnu + 6*db0*dnu + &
        6*dnu**2)/6. + (0.25 - da0**2)/t**2 +  ((4 - da0**2 - 15*db0**2)*t**2)/240.
endif




end subroutine


subroutine jacobiq(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)   :: t
double precision, intent(out)  :: val

call jacobiq0(t,val,da,db)

end subroutine



subroutine jacobiq2(t,val)
implicit double precision (a-h,o-z)
double precision, intent(in)   :: t
double precision, intent(out)  :: val

call jacobiq0(t,val,db,da)

end subroutine



end module



program kummer_jacobi

use utils
use kummer
use kummer_jacobi_routines

implicit double precision (a-h,o-z)

double precision, allocatable :: xscheb(:),whtscheb(:),chebintl(:,:),chebintr(:,:), &
   ucheb(:,:),vcheb(:,:)
double precision, allocatable :: ab(:,:),qs(:,:),xs(:,:),rs(:,:),rders(:,:)
double precision, allocatable :: rs2(:,:),rders2(:,:),ab2(:,:)

double precision, allocatable :: alpha(:,:),alphap(:,:),alphapp(:,:)
double precision, allocatable :: alpha2(:,:),alphap2(:,:),alphapp2(:,:)

double precision, allocatable :: abinv(:,:),alphainv(:,:),alphainvp(:,:),xsinv(:,:)
double precision, allocatable :: abinv2(:,:),alphainv2(:,:),alphainvp2(:,:),xsinv2(:,:)

double precision, allocatable :: xslege(:),whtslege(:),coefs(:)

double precision, allocatable :: nodes(:),weights(:)

call elapsed(t1)

pi = acos(-1.0d0)
call mach_zero(eps0)


call elapsed(tt1)
!
!  Fetch the Clenshaw-Curtis quadrature and related matrices.
!  

k = 20
call chebexps(k,xscheb,whtscheb,ucheb,vcheb,chebintl,chebintr)

!
! Record the time required to construct the phase function
!


!
!  Set the wavenumber for the problem.
!

norder      = 10**5
dnu         = norder

da          =  0.25d0
db          =  0.3d0

eps    = 1.0d-13
a      = 1.0d-15
b      = pi/2

if (eps0 .lt. 1.0d-20) then
a = 1.0d-20
endif

ifleft = 1

!
!  Construct the first phase function
!


call kummer_adap(eps,a,b,jacobiq,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alphap,alphapp)
call kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints,ab,alpha,alphap,alphapp)


!
!  ... and the second ...
! 

call kummer_adap(eps,a,b,jacobiq2,k,xscheb,chebintl,chebintr,ucheb, &
   nints2,ab2,alphap2,alphapp2)

call kummer_phase(ifleft,k,xscheb,chebintl,chebintr,ucheb, &
   nints2,ab2,alpha2,alphap2,alphapp2)


call elapsed(tt2)

t_phase = tt2-tt1
!
!  Report on the number of intervals being used to represent the phase function
!

n  = nints  * k
n2 = nints2 * k

call prini("after kummer_adap, nints  = ",nints)
call prini("after kummer_adap, nints2 = ",nints2)
call prini("after kummer_logform, n + n2 = ",n+n2)


!
!  Compute the coefficients a_1 and a_2 such that the solution is 
!  represented in the form
!
!         a_1 sin( \alpha(t) + a_2)
!
!

if (eps0 .gt. 1.0d-20) then

a0 = 1.0d0/dnu * 1.0d-4
if (a0 .lt. 1.0d-15) a0 = 1.0d-15

call jacobitaylor(a0,ya,ypa,da,db)
call kummer_coefs2(nints,ab,k,xscheb,alpha,alphap,alphapp,a0,ya,ypa,a1,a2)

call jacobitaylor(a0,ya2,ypa2,db,da)
call kummer_coefs2(nints2,ab2,k,xscheb,alpha2,alphap2,alphapp2,a0,ya2,ypa2,b1,b2)

else

call jacobitaylor(a,ya,ypa,da,db)
call kummer_coefs(ifleft,nints,ab,k,xscheb,alphap,alphapp,ya,ypa,a1,a2)

call jacobitaylor(a,ya2,ypa2,db,da)
call kummer_coefs(ifleft,nints2,ab2,k,xscheb,alphap2,alphapp2,ya2,ypa2,b1,b2)

endif



!
!  Calculate the constant in the weight formula.
!


call prina("")

call gammaratio2(dconst)
call prind("in kummer_jacobi, dconst = ",dconst)

call prina("")
call prind("dnu=",dnu)
call prind("a=",a)
call prind("ya = ",ya)
call prind("ypa = ",ypa)
call prind("after kummer_coefs, a1 = ",a1)
call prind("after kummer_coefs, a2 = ",a2)

call prina("")
call prind("ya2 = ",ya)
call prind("ypa2 = ",ypa)
call prind("after kummer_coefs, b1 = ",b1)
call prind("after kummer_coefs, b2 = ",b2)


!
!  Compute the inverse phase function 
!


allocate(abinv(2,nints),alphainv(k,nints),alphainvp(k,nints),xsinv(k,nints))
allocate(abinv2(2,nints2),alphainv2(k,nints2),alphainvp2(k,nints2),xsinv2(k,nints2))

call kummer_phase_inverse(nints,ab,k,xscheb,chebintl,ucheb,alpha,alphap, &
    nintsinv,abinv,alphainv,alphainvp)

call kummer_phase_inverse(nints2,ab2,k,xscheb,chebintl,ucheb,alpha2,alphap2, &
    nints2inv,abinv2,alphainv2,alphainvp2)

call kummer_zeros_count(nints,ab,k,xscheb,alpha,a1,a2,nroots)


allocate(nodes(norder),weights(norder))

!
!  Compute the quadrature rule
!

call elapsed(tt1)
dsum = 0

do i = 1,norder
call jacobipt(norder,nroots,nints,nintsinv,abinv,ab,k,xscheb,alphainv,alphap,a1,a2, &
  nints2,nints2inv,abinv2,ab2,alphainv2,alphap2,b1,b2,i,x,wht)
end do

call elapsed(tt2)
t_quad = tt2-tt1

call elapsed(t2)


call prina("")
call prini("norder = ",norder)
call prin2("log10(norder) = ",log10(norder+0.0d0))
call prin2("phase time = ",t_phase)
call prin2("quadrature time = ",t_quad)
call prin2("total time = ",t2-t1)

end program
