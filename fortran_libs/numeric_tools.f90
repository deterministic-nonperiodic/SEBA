! manipulacion de errores ...
!===================================================================================================
  subroutine model_error( message )
!===================================================================================================

 implicit none

 character (len = *), intent (in) :: message

 call model_message('Execution aborted with status', 'Traceback:', 1.d0, 'I2')
 call model_message( message, '.', 0.d0, 'I2')
 stop

  return
 end subroutine model_error
!===================================================================================================

!===================================================================================================
  subroutine model_message( main_message, optn_message, values, vfmt)
!===================================================================================================

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
!===================================================================================================

!===================================================================================================
  subroutine gradient(dpds, var, ds, ns, nt, order)
!===================================================================================================

 implicit none

 ! input vars ...
 integer,          intent(in)  :: order
 integer,          intent(in)  :: ns, nt
 double precision, intent(in)  :: ds
 double precision, intent(in)  :: var (nt, ns)

 ! local
 integer                       :: k, nso, nsf

 ! output vars ...
 double precision, intent(out) :: dpds (nt, ns)

  dpds = 0.0

  nso = 2
  nsf = ns - (nso - 1)

  ! loop over samples
  do k = 1, nt
    call compact_derivative(dpds(k, :), var(k, :), ds, ns, order)
  enddo

 return
 end subroutine gradient
!===================================================================================================

!===================================================================================================
  subroutine linederv(ads, var, ds, ns, nso, nsf, order)
!===================================================================================================
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
!===================================================================================================

!===================================================================================================
  subroutine compact_derivative(ads, var, ds, ns, order)
!===================================================================================================
!
! module for computing high order centered compact finite differences
! formulas for aproximating first derivatives amoung a line. The tridiagonal
! system of equations is solved by thomas reduction algorithm O(ns).
!
!                       !  ds  !
!
! *------*------ ... ---*------*------*------*------*--- ... -----*------*
! 1     is             i-2    i-1     i     i+1    i+2           ie     ns
!
! options for scheme of orders between 2 and 8
!
! Signature:
! ----------
! du/dx = compact_derivative(u(x), dx, nx, nxo, nxf, order)
!---------------------------------------------------------------------------------------------------
 implicit none
! inputs :
 integer         , intent(in ) :: order
 integer         , intent(in ) :: ns
 double precision, intent(in ) :: ds
 double precision, intent(in ) :: var (ns)
! local :
 double precision              :: rhs (ns)
 double precision              :: a, b, c, d
 double precision              :: ds2, ds3
 double precision              :: cflux2, cflux4, bflux4
 logical                       :: flag
 integer                       :: i, n, is, ie, bs, be, ne
! output derivative:
 double precision, intent(out) :: ads (ns)

  rhs = 0.0
  ads = 0.0
  flag = .true.

  ds2 = 2.0 * ds
  ds3 = 3.0 * ds

  is = int(order / 2) + 1
  ie = ns + 1 - is

  bs = is - 1
  be = ie + 1

  ne = ie - is + 1

  ! compute ( u(i+1) - u(i-1) ) / 2 ds for all schemes
  call linederv(rhs, var, ds, ns, 2, ns-1, 2)

  ! selecting coefficients for compact scheme depending on order
  scheme_order: select case (order)

    case(4) ! fourth order compact scheme o(dx^4)

      a = 1.0; b = 4.0; c = 1.0; d = 6.0

      ! store rhs of the system:
      rhs(is:ie) = d * rhs(is:ie)

    case(5, 6)
      ! 5- modified sixth order compact scheme o(dx^6). lele, (1992).

      ! define coefficients
      if (order == 5) then
        a = 2.5; b = 7.0; c = 2.5; d = 11.0
      else
        a = 3.0; b = 9.0; c = 3.0; d = 14.0
      endif

      ! compute d(u[i+1] - u[i-1])/(2ds) + b(u[i+2] - u[i-2])/(4ds)
      rhs(is:ie) = d * rhs(is:ie)

      do i = is, ie
          rhs(i) = rhs(i) + cflux2(var(i-2), var(i+2)) / ds2
      enddo

    case(8)
      ! modified family of sixth-order compact scheme o(dx^6). lele, (1992).

      ! define coefficients
      a = 3.0; b = 8.0; c = 3.0; d = 12.5

      ! compute d(u[i+1] - u[i-1])/(2ds) + b(u[i+2] - u[i-2])/(4ds) + c(u[i+3] - u[i-3])/(6ds)
      rhs(is:ie) = d * rhs(is:ie)

      do i = is, ie
          rhs(i) = rhs(i) + 1.6 * cflux2(var(i-2), var(i+2)) / ds2 &
          &               - 0.1 * cflux2(var(i-3), var(i+3)) / ds3
      enddo

    case default

      ! no need to solve the system,
      ! the algorithm is consistent with a=0, b=1, c=0 anyway
      ! (single diagonal matrix can be explicitly solved)
      flag = .false.

      ads = rhs

  end select scheme_order

  ! Solve system and compute derivatives at the boundaries
  if (flag) then ! (only for order /= 2)

    !---------------------------------------------------------------------------------
    ! Using fourth order explicit scheme at boundaries
    !---------------------------------------------------------------------------------

    ! Centered scheme at inner boundary points (3, bs), and (be, ns-3)
    do i = 3, bs ! has no effect for order < 4 since bs = 2
        n = be - 3 + i
        ads(i) = cflux4(var(i-2), var(i-1), var(i+1), var(i+2)) / ds
        ads(n) = cflux4(var(n-2), var(n-1), var(n+1), var(n+2)) / ds
    end do

    ! Forward-centered scheme at outer boundary points (1, 3), and (ns-2, ns)
    do i = 1, 2
        n = ns + 1 - i
        ads(i) =   bflux4(var(i), var(i+1), var(i+2), var(i+3), var(i+4)) / ds
        ads(n) = - bflux4(var(n), var(n-1), var(n-2), var(n-3), var(n-4)) / ds
    end do

    !------------------------------------------------------------------------------------
    ! Solving the tridiagonal system with constant coefficients using thomas solver O(n)
    !------------------------------------------------------------------------------------
    ! Substract known values of the derivatives at the first and last row of the system
    ! (calculated with 4th order scheme)
    rhs(is) = rhs(is) - a * ads(bs)
    rhs(ie) = rhs(ie) - c * ads(be)

    ! Solve tridiagonal system for the interior grid points
    call cons_thomas(rhs(is:ie), a, b, c, ne)

    ads(is:ie) = rhs(is:ie)

  endif

  return
 end subroutine compact_derivative
!===================================================================================================

!===================================================================================================
  subroutine cons_thomas(v, a, b, c, ns)
!===================================================================================================
!         Thomas tridiagonal solver with constant coefficients
!---------------------------------------------------------------------------------------------------
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
!===================================================================================================


!===================================================================================================
  subroutine gen_thomas(v, a, b, c, nn)
!===================================================================================================
!          Thomas tridiagonal solver with generic coefficients
!---------------------------------------------------------------------------------------------------

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

 ! matrix coefficients
 !
 ! a [ 1, 3, 4, ..., nz-1 ]  (lower diagonal)
 ! b [ 1, 2, 3, ..., nz   ]  (diagonal      )
 ! c [ 1, 3, 4, ..., nz-1 ]  (upper diagonal)
 !
 ! v  ... (rhs of system of equations / solution)

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
!===================================================================================================


!===================================================================================================
! define central differences operators for dq/dx (avoid repeating code) :
!===================================================================================================

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

!===================================================================================================
! define foward-backward differences operators for dq/dx (at boundaries):
!===================================================================================================
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
  bflux4 = dot_product([q_icen, q_ipm1, q_ipm2, q_ipm3, q_ipm4], coeffs)

  return
 end function bflux4

 ! points [ i, i+-1, i+-2, i+-3, i+-4 ]
 double precision function bflux6(q_icen, q_ipm1, q_ipm2, q_ipm3, q_ipm4)
  double precision, intent(in) :: q_icen, q_ipm1, q_ipm2, q_ipm3, q_ipm4
  double precision             :: coeffs(5)

  ! foward difference (backward is: '-flux4' with reversed arguments )
  ! bflux4 = - ( 25.0 * q_icen - 48.0 * q_ipm1 +                             &
  !              36.0 * q_ipm2 - 16.0 * q_ipm3 + 3.0 * q_ipm4 ) / 12.0
  coeffs = [-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0]
  bflux6 = dot_product([q_icen, q_ipm1, q_ipm2, q_ipm3, q_ipm4], coeffs)

  return
 end function bflux6

 ! points [ i-+1, i, i+-1, i+-2, i+-3 ]
 double precision function iflux4(q_imp1, q_icen, q_ipm1, q_ipm2, q_ipm3)
  double precision, intent(in) :: q_imp1, q_icen, q_ipm1, q_ipm2, q_ipm3
  double precision             :: coeffs(5)

  ! foward difference (backward is: '-flux4' with reversed arguments )
  ! iflux4 = - ( 3.0 * q_imp1 + 10.0 * q_icen -                              &
  !             18.0 * q_ipm1 +  6.0 * q_ipm2 - q_ipm3 ) / 12

  coeffs = [-3.0, -10.0, 18.0, -6.0, 1.0] / 12.0
  iflux4 = dot_product([q_imp1, q_icen, q_ipm1, q_ipm2, q_ipm3], coeffs)

  return
 end function iflux4
!===================================================================================================

!===================================================================================================
  subroutine adams_moulton(var, var0, func, ds, ns)
!===================================================================================================

 implicit none

 integer         , intent(in ) :: ns
 double precision, intent(in ) :: ds

 double precision, intent(in ) :: var0
 double precision, intent(in ) :: func (ns)

 double precision              :: intvar
 integer                       :: k

 double precision, intent(out) :: var (ns)

 ! adams moulton-2 ...
 var(1) = var0
 var(2) = var(1) + ds * ( func(1) + func(2) ) / 2.0

 ! adams moulton-3 ...
 intvar = 5.0 * func(3) + 8.0 * func(2) - func(1)
 var(3) = var(2) + ds * intvar / 12.0

! adams moulton-4 ... ( foward )
 do k = 3, ns - 1

   intvar = 9.0 * func(k+1) + 19.0 * func(k) - 5.0 * func(k-1) + func(k-2)
   var(k+1) = var(k) + ds * intvar / 24.0

 end do

 return
end subroutine adams_moulton
!===================================================================================================


!===================================================================================================
  subroutine adaptative_adams_moulton(var, var0, func, s, ns)
!===================================================================================================

 implicit none

 integer         , intent(in ) :: ns
 double precision, intent(in ) :: var0
 double precision, intent(in ) :: s    (ns)
 double precision, intent(in ) :: func (ns)

 double precision              :: ds   (ns)
 double precision              :: c1, c2, c3
 integer                       :: k

 double precision, intent(out) :: var (ns)

 var = 0.0

 ! calculate grid step
 ds(1:ns-1) = s(2:ns) - s(1:ns-1)
 ds(ns) = 0.0

 ! adams moulton-2 ...
 var(1) = var0
 var(2) = var(1) + 0.5 * ds(1) * (func(1) + func(2))

! adams moulton-4 ... ( foward )
 do k = 2, ns - 1

     c1 =  (ds(k) / (6.0 * (ds(k) + ds(k-1)))) * (2.0 * ds(k) + 3.0 * ds(k-1))
     c2 =  (ds(k) / (6.0 * ds(k-1))) * (ds(k) + 3.0 * ds(k-1))
     c3 = -(ds(k) ** 3) / (6.0 * ds(k-1) * (ds(k) + ds(k-1)))

     var(k+1) = var(k) + c1 * func(k+1) + c2 * func(k) + c3 * func(k-1)

 end do

 return
end subroutine adaptative_adams_moulton
!===================================================================================================


! vertical integration for any f(z) ( Simpson rule is used ) ...
!===================================================================================================
  subroutine intdomvar(intvar, var, zc, nz, n)
!===================================================================================================

 implicit none

 integer,          intent(in ) :: nz, n
 double precision, intent(in ) :: zc   (nz)
 double precision, intent(in ) :: var  (nz)

 double precision              :: vari (n )
 double precision              :: dzi, z, zend

 integer                       :: kn, nnods
 integer                       :: nnstp, kstr, kend

 double precision, intent(out) :: intvar
!===================================================================================================
!
!Polynomic interpolation (possible to modify the numnber of nodes: nnods between 3-7 is recommended,
!                         reduces to linear interpolation for 2 nodes. Avoid using nnods > 7 nodes)

 nnods = 5
 nnstp = nnods - 1
 kend  = 1

 dzi = (zc(nz) - zc(1)) / (n - 1)
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

 !Integrating using Simpson's rule on regularly spaced data...

 intvar = ( vari(1) + vari(n) )

 intvar = dzi * (intvar + 4.0 * sum(vari(1:n-1:2)) + 2.0 * sum(vari(2:n-1:2))) / 3.0

 return
 end subroutine intdomvar
!===================================================================================================

!===================================================================================================
  subroutine lagrange_intp(datao, datai, posio, posii, nnods)
!===================================================================================================
!
! Lagrange polynomic interpolation
!
integer,          intent(in ) :: nnods
double precision, intent(in ) :: posio
double precision, intent(in ) :: posii(nnods)
double precision, intent(in ) :: datai(nnods)

double precision              :: lgrw
integer                       :: i, j

double precision, intent(out) :: datao

datao = 0.0

! check if the data is within the interpolation range
!if (posio >= min(posii)).and.(posio <= max(posii)) then
!
!end if

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
!===================================================================================================

!===================================================================================================
  subroutine surface_temperature(sfct, sfcp, temp, pres, nt, ns, np)
!===================================================================================================

 implicit none

 integer,          intent(in ) :: nt, ns, np
 double precision, intent(in ) :: pres (np)
 double precision, intent(in ) :: sfcp (ns)
 double precision, intent(in ) :: temp (nt, ns, np)

 integer                       :: i, j, ks, ke, nn
 double precision              :: safe_temp
 logical                       :: mask (np)
 double precision, intent(out) :: sfct (nt, ns)

  sfct = 0.0
  safe_temp = 120.0 ! safe surface temperature (K)

  do i = 1, nt
    do j = 1, ns

      ! find the index ks closest to the first level pierced by the surface [pres(ke) <= sfcp(j)]
      ! such that temperature is in a valid range [temp(i, j, ks) > safe_temp].
      mask = pack(temp(i, j, :) > safe_temp, .not.isnan(temp(i, j, :)))

      ks = minloc(abs(pres - sfcp(j)), mask=mask, dim=1)
      ke = min(ks + 1, np)

      nn = ke + 1 - ks

      call lagrange_intp(sfct(i, j), temp(i, j, ks:ke), log(sfcp(j)), log(pres(ks:ke)), nn)

    end do
  end do

 return
 end subroutine surface_temperature
!===================================================================================================

!===================================================================================================
  subroutine geopotential(phi, pres, temp, sfch, sfcp, sfct, nt, ns, np)
!===================================================================================================

      implicit none

      integer, intent(in) :: nt, ns, np

      double precision, intent(in) :: pres (np)
      double precision, intent(in) :: sfch (ns)
      double precision, intent(in) :: sfcp (ns)
      ! surface temperatue is optional. If not given,
      ! it is approximated by linear interpolation.
      double precision, optional, intent(in) :: sfct (nt, ns)
      double precision, intent(in) :: temp (nt, ns, np)

      double precision :: sfc_temp (nt, ns)
      double precision :: lnp  (np)
      double precision :: lnps (ns)

      double precision :: Rd, g
      double precision :: phi_bc, tbar

      integer :: i, j, nc, kn

      double precision, intent(out) :: phi (nt, ns, np)

 Rd = 287.058 ! gas constant for dry air (J / kg / K)
 g = 9.806650 ! acceleration of gravity  (m / s**2)

 ! initializing geopotential array
 phi = 0.0

 ! local arrays for working on log-pressure coordinates
 lnp = log(pres)
 lnps = log(sfcp)

 ! Check if surface temperature is passed as argument (default .false.)
 if (present(sfct)) then
     sfc_temp = sfct
 else
     ! Approximate surface temperature by linear interpolation
     call surface_temperature(sfc_temp, sfcp, temp, pres, nt, ns, np)
 end if

 ! Start vertical integration of the hydrostatic equation: d(phi)/d(log p) = - Rd * T(p)
      do i = 1, nt ! loop over samples
          do j = 1, ns ! loop over spatial dimension

              ! find the first level above the surface (p <= sfcp)
              kn = minloc(abs(pres - sfcp(j)), mask = pres <= sfcp(j), dim = 1)

              ! number of levels above the surface
              nc = np + 1 - kn

              ! Using second order accurate mid-point method in log-pressure for the
              ! first integration step. Average temperature in the first two levels:
     tbar = (lnp(kn) * temp(i, j, kn) + lnps(j) * sfc_temp(i, j)) / (lnp(kn) + lnps(j))

     ! Lower boundary condition for the geopotential (first level above the surface)
     phi_bc = g * sfch(j) - Rd * (lnp(kn) - lnps(j)) * tbar

     ! Vertical integration for levels above the surface: d(phi)/d(log p) = - Rd * T(p)
     ! Integrating in log-pressure space using 4th-order Adams-Moulton linear multistep method.
     call adaptative_adams_moulton(phi(i, j, kn:np), phi_bc, -Rd*temp(i, j, kn:np), lnp(kn:np), nc)

   end do
 end do

 return
end subroutine geopotential
!===================================================================================================
