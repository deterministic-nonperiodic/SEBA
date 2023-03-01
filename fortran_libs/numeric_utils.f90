! manipulacion de errores ...
!==========================================================================
  subroutine model_error( message )
!==========================================================================

 implicit none

 character (len = *), intent (in) :: message

 call model_message('Execution aborted with status', 'Traceback:', 1.d0, 'I2')
 call model_message( message, '.', 0.d0, 'I2')
 stop

  return
 end subroutine model_error
!==========================================================================

!==========================================================================
  subroutine model_message( main_message, optn_message, values, vfmt)
!==========================================================================

 implicit none

 character (len= *), intent(in) :: main_message
 character (len= *), intent(in) :: optn_message
 character (len= *), intent(in) :: vfmt

 double precision,   intent(in) :: values
 character (len= 500)           :: str_fmt
 integer                        :: ios

 str_fmt = '("'// trim(adjustl(main_message)) //'",'// trim(vfmt) //',"'// &
                  trim(adjustl(optn_message)) //'")'

 if ( vfmt(1:1) == 'I' ) then
   write( unit=*, fmt=trim(adjustl(str_fmt)), iostat=ios) int( values )
 else
   write( unit=*, fmt=trim(adjustl(str_fmt)), iostat=ios) values
 endif

 if ( ios /= 0 ) stop 'write error in unit '

return
end subroutine model_message
!==========================================================================

!==========================================================================
  subroutine gradient(dpdx, var, dx, nx, nt, order)
!==========================================================================

 implicit none

 ! input vars ...
 integer,          intent(in)  :: order
 integer,          intent(in)  :: nx, nt
 double precision, intent(in)  :: dx
 double precision, intent(in)  :: var (nx, nt)

 ! local
 integer                       :: k, i, nxo, nxf

 ! output vars ...
 double precision, intent(out) :: dpdx (nx, nt)

  dpdx = 0.0

  nxo = 2
  nxf = nx - (nxo - 1)

  ! loop over samples
  do k = 1, nt
    call compactderv(dpdx(:, k), var(:, k), dx, nx, nxo, nxf, order)
  enddo

 return
 end subroutine gradient
!==========================================================================

!==========================================================================
  subroutine linederv(ads, var, ds, ns, nso, nsf, order)
!==========================================================================
!
! module for computing high order centered finite differences formulas for
! aproximating first derivatives amoung a line.
!
!                       !  ds  !
!
! *------*------ ... ---*------*------*------*------*--- ... ------*------*
! 1     nso            j-2    j-1     j     j+1    j+2            nsf     ns
!
! options: ord = 1-2, 3-4, 5-6
!
! uses:
!
! advection (x) = u * linederv( u(x) , dx, nx, nxo,nxf, 6 )
!
 implicit none

! inputs :
 integer         , intent(in ) :: order
 integer         , intent(in ) :: ns, nso, nsf
 double precision, intent(in ) :: ds
 double precision, intent(in ) :: var (ns)

! local :
 integer                       :: i
 double precision              :: cflux2, cflux4
 double precision              :: bflux2

! output :
 double precision, intent(out) :: ads (ns)

 ads = 0.0

 ! All stencils are centered differences formulas (2nd, 4th and 6th order)

 ! compute fluxes ...
 select case(order)

  case(2)

    do i = nso, nsf
      ads(i) = cflux2( var(i-1), var(i+1) )
    enddo

    ! foward/backward schemes O( ds ^ 2 )
    ads(nso-1) =   bflux2(var(nso-1), var(nso), var(nso+1))
    ads(nsf+1) = - bflux2(var(nsf+1), var(nsf), var(nsf-1))

  case(4)

    do i = nso + 1, nsf - 1
      ads(i) = cflux4( var(i-2), var(i-1), var(i+1), var(i+2) )
    enddo

    ads(nso) = cflux2( var(nso-1), var(nso+1) )
    ads(nsf) = cflux2( var(nsf-1), var(nsf+1) )

    ads(nso-1) =   bflux2(var(nso-1), var(nso), var(nso+1))
    ads(nsf+1) = - bflux2(var(nsf+1), var(nsf), var(nsf-1))

  case default

    call model_error('wrong flux operator: order = 2 or 4')

 end select

 ads = ads / ds

 return
 end subroutine linederv
!==========================================================================

!==========================================================================
  subroutine compactderv(ads, var, ds, ns, is, ie, order)
!==========================================================================
!
! module for computing high order centered compact finite differences
! formulas for aproximating first derivatives amoung a line.
! the system of equations is solved by thomas reduction algorithm.
!
!                       !  ds  !
!
! *------*------ ... ---*------*------*------*------*--- ... -----*------*
! 1     is             i-2    i-1     i     i+1    i+2           ie     ns
!
! options: ord = 1-2, 3-4, 5-6
!
! uses:
!
! du/dx = compactderv( u(x), dx, nx, nxo, nxf, order )
!
 implicit none
! inputs :
 integer         , intent(in ) :: order
 integer         , intent(in ) :: ns, is, ie
 double precision, intent(in ) :: ds
 double precision, intent(in ) :: var (ns  )

! local :
 double precision              :: rhs (ns  )
 double precision              :: a, b, c, d, ds4
 logical                       :: flag
 integer                       :: i

! output :
 double precision, intent(out) :: ads (ns)

  rhs = 0.0
  ads = 0.0
  flag = .true.

  ds4 = 4.0 * ds

  ! selecting coefficients for compact scheme depending on order
  scheme_order: select case (order)

    case(4) ! fourth order compact scheme o(dx^4)

      a = 1.0; b = 4.0; c = 1.0; d = 6.0

      ! store rhs of the system:
      call linederv(rhs, var, ds, ns, is, ie, 2)
      rhs(is:ie) = d * rhs(is:ie)

    case(5, 6)
      ! 5- modified sixth order compact scheme o(dx^6). lele, (1992).

      ! define coefficients
      if (order == 5) then
        a = 2.5; b = 7.0; c = 2.5; d = 11.0
      else
        a = 3.0; b = 9.0; c = 3.0; d = 14.0
      endif

      ! compute ( u(i+1) - u(i-1) ) / 2 ds
      call linederv(rhs, var, ds, ns, is, ie, 2)

      ! compute ( u(i+2) - u(i-2) ) / 4 ds
      do i = is+1, ie-1
          rhs(i) = d * rhs(i) + ( var(i+2) - var(i-2) ) / ds4
      enddo
      rhs(is) = (d + 1.0) * rhs(is)
      rhs(ie) = (d + 1.0) * rhs(ie)

    case default

      ! no need to solve the system,
      ! the algorithm is consistent with a=0, b=1, c=0 anyway
      ! (single diagonal matrix can be explicitly solved)
      flag = .false.

      ! second order scheme  o(dx^2)
      a = 0.0; b = 1.0; c = 0.0; d = 1.0
      call linederv(ads, var, ds, ns, is, ie, 2)

  end select scheme_order

  if (flag) then ! (only for order /= 2)

    ! boundaries are necesary 2nd order (test 4th)
    ads(is-1) = rhs(is-1)
    ads(ie+1) = rhs(ie+1)

    ! firt and last rows of the system:
    rhs(is) = rhs(is) - a * ads(is-1)
    rhs(ie) = rhs(ie) - c * ads(ie+1)

    ! solving the tridiagonal system with thomas algorithm O(n):
    call cons_thomas(rhs(is:ie), a, b, c, ns-2)

    ads(is:ie) = rhs(is:ie)

  endif

  return
 end subroutine compactderv
!==========================================================================

!==========================================================================
  subroutine cons_thomas(v, a, b, c, ns)
!==========================================================================
!         Thomas tridiagonal solver with constant coefficients
!--------------------------------------------------------------------------
   implicit none
   ! inputs...
   integer,          intent(in   ) :: ns
   double precision, intent(in   ) :: a, b, c

   !local temporal arrays..
   double precision                :: q (2:ns)
   double precision                :: r
   integer                         :: k

   ! input-output array ...
   double precision, intent(inout) :: v (ns  )

  r = b
  v(1) = v(1) / b

  ! Foward substitution
  do k = 2, ns
      q(k) = c / r
      r = b - a * q(k)
      v(k) = ( v(k) - a * v(k-1) ) / r
  enddo

  ! Backward substitution...
  do k = ns-1, 1, -1
      v(k) = v(k) - q(k+1) * v(k+1)
  enddo

  return
 end subroutine cons_thomas
!==========================================================================


!==========================================================================
  subroutine gen_thomas(v, a, b, c, nn)
!==========================================================================
!          Thomas tridiagonal solver with generic coefficients
!--------------------------------------------------------------------------

 implicit none
 ! inputs...
 integer,          intent(in   ) :: nn

 double precision, intent(in   ) :: a (nn-1)
 double precision, intent(in   ) :: b (nn  )
 double precision, intent(in   ) :: c (nn-1)
 !local temporal arrays..
 double precision                :: q (nn-1)
 double precision                :: rk
 integer                         :: k

 ! input-output array ...
 double precision, intent(inout) :: v (nn  )

 ! coeficientes matriciales ...
 !
 ! a [ 1, 3, 4, ..., nz-1 ]  (diagonal debajo de la principal)
 ! b [ 1, 2, 3, ..., nz   ]  (diagonal principal             )
 ! c [ 1, 3, 4, ..., nz-1 ]  (diagonal encima de la principal)
 !
 ! v  ... (miembro derecho del sistema de ecuaciones / solución)

  rk = b(1)
  v(1) = v(1) / rk
  ! foward substitution ...
  do k = 1, nn-1
      q(k) = c(k) / rk
      rk   = b(k+1) - a(k) * q(k)
      v(k+1) = ( v(k+1) - a(k) * v(k) ) / rk
  enddo

  ! backward substitution...
  do k = nn-1, 1, -1
      v(k) = v(k) - q(k) * v(k+1)
  enddo

 return
end subroutine gen_thomas
!==========================================================================


!==========================================================================
! define central differences operators for dq/dx (avoid repeating code) :
!==========================================================================

 double precision function cflux2(q_im1, q_ip1)
  double precision, intent(in) :: q_im1, q_ip1

  cflux2 = (q_ip1 - q_im1) / 2.0

  return
 end function cflux2

 double precision function cflux4(q_im2, q_im1, q_ip1, q_ip2)
  double precision, intent(in) :: q_im2, q_im1, q_ip1, q_ip2

  cflux4 = ( 8.d0 * (q_ip1 - q_im1) - (q_ip2 - q_im2) ) / 12.0

  return
 end function cflux4

!==========================================================================
! define foward-backward differences operators for dq/dx (at boundaries):
!==========================================================================
 ! points [ i, i+-1, i+-2 ]
 double precision function bflux2(q_icen, q_ipm1, q_ipm2)
  double precision, intent(in) :: q_icen, q_ipm1, q_ipm2

  ! foward difference (backward is: '-flux2' with reversed arguments )
  bflux2 = - ( 3.0 * q_icen - 4.0 * q_ipm1 + q_ipm2 ) / 2.0

  return
 end function bflux2

 ! points [ i, i+-1, i+-2, i+-3, i+-4 ]
 double precision function bflux4(q_icen, q_ipm1, q_ipm2, q_ipm3, q_ipm4)
  double precision, intent(in) :: q_icen, q_ipm1, q_ipm2, q_ipm3, q_ipm4
  double precision             :: coeffs(5)

  ! foward difference (backward is: '-flux4' with reversed arguments )
  ! bflux4 = - ( 25.0 * q_icen - 48.0 * q_ipm1 +                             &
  !              36.0 * q_ipm2 - 16.0 * q_ipm3 + 3.0 * q_ipm4 ) / 12.0

  coeffs = [-25.0, 48.0, -36.0, 16.0, -3.0] / 12.0
  bflux4 = dot_product([q_icen, q_ipm1, q_ipm2, q_ipm3, q_ipm4], coeffs )

  return
 end function bflux4

 ! points [ i-+1, i, i+-1, i+-2, i+-3 ]
 double precision function iflux4(q_imp1, q_icen, q_ipm1, q_ipm2, q_ipm3)
  double precision, intent(in) :: q_imp1, q_icen, q_ipm1, q_ipm2, q_ipm3
  double precision             :: coeffs(5)

  ! foward difference (backward is: '-flux4' with reversed arguments )
  ! iflux4 = - ( 3.0 * q_imp1 + 10.0 * q_icen -                              &
  !             18.0 * q_ipm1 +  6.0 * q_ipm2 - q_ipm3 ) / 12

  coeffs = [-3.0, -10.0, 18.0, -6.0, 1.0] / 12.0
  iflux4 = dot_product([q_imp1, q_icen, q_ipm1, q_ipm2, q_ipm3], coeffs )

  return
 end function iflux4

!==========================================================================

!==========================================================================
  subroutine adams_moulton(var, var0, func, ds, ns, nso, nsf)
!==========================================================================

 implicit none

 integer         , intent(in ) :: ns, nso, nsf
 double precision, intent(in ) :: ds

 double precision, intent(in ) :: var0
 double precision, intent(in ) :: func (ns)

 double precision              :: intvar
 integer                       :: k

 double precision, intent(out) :: var (ns)

 ! adams moulton-2 ...
 var(1) = var0
 var(nso) = var(nso-1) + ds * ( func(nso-1) + func(nso) ) / 2.0

 ! adams moulton-3 ...
 intvar = 5.0 * func(nso+1) + 8.0 * func(nso) - func(nso-1)
 var(nso+1) = var(nso) + ds * intvar / 12.0

! adams moulton-4 ... ( foward )
 do k = nso+1, nsf

   intvar = 9.0 * func(k+1) + 19.0 * func(k) - 5.0 * func(k-1) + func(k-2)
   var(k+1) = var(k) + ds * intvar / 24.0

 end do

 return
end subroutine adams_moulton
!==========================================================================

! vertical integration for any f(z) ( Simpson rule is used ) ...
!==========================================================================
  subroutine intdomvar(intvar, var, zc, nz, n)
!==========================================================================

 implicit none

 integer,          intent(in ) :: nz, n
 double precision, intent(in ) :: zc   (nz)
 double precision, intent(in ) :: var  (nz)

 double precision              :: vari (n )
 double precision              :: dzi, z, zend

 integer                       :: kn, nnods
 integer                       :: nnstp, kstr, kend

 double precision, intent(out) :: intvar
!***********************************************************************
!
!interpolacion polinomica(posible modificar la cantidad de nodos: nnods
!                         entre 3-7 recomendado, para valores menores se
!                         prefiere interpolacion lineal, para valores
!                         mayores el metodo de lagrange es inadecuado )
!
 nnods = 5
 nnstp = nnods - 1
 kend  = 1

 dzi = ( zc(nz)-zc(1 ) ) / ( n - 1)
 z   = dzi
 kn  = 2

 vari(1) = var(1 ) ! extremos deben coincidir
 vari(n) = var(nz) !

 do while ( kend < nz )

  kstr = kend
  kend = min(kstr + nnstp, nz)

  zend = zc(kend)

  do while ( z < zend .and. kn < n )

    call lagrange_intp(vari(kn), var(kstr:kend), z, zc(kstr:kend), nnods)

    z  = z  + dzi
    kn = kn + 1

  end do

 end do

!integracion por metodo de simpson ...

  intvar = ( vari(1) + vari(n) )

  intvar = dzi * ( intvar + 4.0 * sum(vari(1:n-1:2))   +                   &
                            2.0 * sum(vari(2:n-1:2)) ) / 3.0

 return
 end subroutine intdomvar
!==========================================================================

!==========================================================================
  subroutine lagrange_intp(datao, datai, posio, posii, nnods)
!==========================================================================
!
! interpolacion polinomica por el metodo de lagrange...
!
integer,          intent(in ) :: nnods
double precision, intent(in ) :: posio
double precision, intent(in ) :: posii(nnods)
double precision, intent(in ) :: datai(nnods)

double precision              :: lgrw
integer                       :: i, j

double precision, intent(out) :: datao

datao = 0.0

do i = 1, nnods

 lgrw = 1.0

 do j = 1, nnods

  if (j /= i) then
   lgrw = lgrw * (posio - posii(j)) / (posii(i) - posii(j))
  endif

 enddo

 datao = datao + datai(i) * lgrw

enddo

return
end subroutine lagrange_intp
!==========================================================================
