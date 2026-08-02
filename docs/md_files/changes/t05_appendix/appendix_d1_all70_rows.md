| Problem | Expression | $n$ | Range | Sampling | Source | GT |
|---|---|---|---|---|---|---|
| Nguyen-1 | `x^3 + x^2 + x` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-2 | `x^4 + x^3 + x^2 + x` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-3 | `x^5 + x^4 + x^3 + x^2 + x` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-4 | `x^6 + x^5 + x^4 + x^3 + x^2 + x` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-5 | `sin(x^2) * cos(x) - 1` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-6 | `sin(x) + sin(x + x^2)` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-7 | `log(x + 1) + log(x^2 + 1)` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-8 | `sqrt(x)` | 1 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-9 | `sin(x) + sin(y^2)` | 2 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-10 | `2 * sin(x) * cos(y)` | 2 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-11 | `x^y` | 2 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| Nguyen-12 | `x^4 - x^3 + 0.5*y^2 - y` | 2 | -- | uniform i.i.d. | \cite{uy2011} | yes |
| I.6.20a | `exp(-theta^2/2) / sqrt(2*pi)` | 1 | $[1,3]$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.12.1 | `mu * N_s` | 2 | $[1,5]^{2}$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.14.3 | `m * g * z` | 3 | $[1,5]^{3}$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.25.13 | `q / C` | 2 | $[1,3]^{2}$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.34.27 | `hbar * omega` | 2 | $[1,5]^{2}$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.39.10 | `1.5 * p_r * V` | 2 | $[1,5]^{2}$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.12.4 | `Ef = q1 * r / (4 * pi * epsilon * r^3)` | 3 | $[1,5]^{3}$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| II.3.24 | `flux = Pwr / (4 * pi * r^2)` | 2 | $[1,5]^{2}$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.10.7 | `m0 / sqrt(1 - v^2/c^2)` | 3 | $[1,5]$, $[1,2]$, $[3,10]$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.48.20 | `m*c^2 / sqrt(1 - (v/c)^2)` | 3 | $[1,5]$, $[3,10]$, $[1,2]$ | uniform i.i.d. | \cite{udrescu2020} | yes |
| I.15.10 | `m0*v / sqrt(1 - v^2/c^2)` | 3 | $[1,5]$, $[1,2]$, $[3,10]$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| I.30.3 | `sin(n*theta/2)^2 / sin(theta/2)^2` | 2 | $[1,5]^{2}$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| I.37.4 | `I1 + I2 + 2*sqrt(I1*I2)*cos(delta)` | 3 | $[1,5]^{3}$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| II.11.27 | `n*alpha/(1 - n*alpha/3) * epsilon * Ef` | 4 | $[0,1]$, $[0,1]$, $[1,2]$, $[1,2]$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| III.17.37 | `beta*(1 + alpha*cos(theta))` | 3 | $[1,5]^{3}$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| Pagie-1 | `1/(1 + x^(-4)) + 1/(1 + y^(-4))` | 2 | $[-5,5]^{2}$ | $26\times26$ grid, zero skipped | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| Korns-12 | `2.0 - 2.1*cos(9.8*x1)*sin(1.3*x5)` | 5 | $[-50,50]^{5}$ | uniform i.i.d., 2000/2000 | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| Vladislavleva-4 | `10 / (5 + (x1-3)^2 + (x2-3)^2 + (x3-3)^2 + (x4-3)^2 + (x5-3)^2)` | 5 | $[0.05,6.05]^{5}$ | uniform i.i.d., 1024/5000 | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| Vladislavleva-2 | `exp(-x)*x^3*(cos(x)*sin(x))*(cos(x)*sin(x)^2 - 1)` | 1 | $[0.05,10]$ | uniform train, grid test | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| Keijzer-6 | `log(x)` | 1 | $[1,120]$ | integer grid (extrapolation) | \cite{udrescu2020,pagie1997,korns2011,vladislavleva2009,keijzer2003} | yes |
| I.29.16 | `sqrt(x1^2 + x2^2 - 2*x1*x2*cos(theta1 - theta2))` | 4 | $[1,5]^{4}$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| I.50.26 | `x1*(cos(omega*t) + alpha*cos(omega*t)^2)` | 4 | $[1,3]^{4}$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| I.16.6 | `(u + v) / (1 + u*v/c^2)` | 3 | $[1,5]^{3}$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| II.11.28 | `1 + n*alpha / (1 - n*alpha/3)` | 2 | $[0,1]^{2}$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| III.14.14 | `I_0 * (exp(q*Volt/(kb*T)) - 1)` | 5 | $[1,5]$, $[1,2]$, $[1,2]$, $[1,2]$, $[1,2]$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| Vlad-7 | `(x1-3)*(x2-3) + 2*sin((x1-4)*(x2-4))` | 2 | $[0.05,6.05]^{2}$ | uniform i.i.d., 300/1200 | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| R2 | `(x^5 - 3*x^3 + 1) / (x^2 + 1)` | 1 | $[-1,1]$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| R3 | `(x^6 + x^5) / (x^4 + x^3 + x^2 + x + 1)` | 1 | $[-1,1]$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| Keijzer-11 | `x*y + sin((x-1)*(y-1))` | 2 | $[-3,3]^{2}$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| Liv-14 | `x1^3 + x1^2 + x1 + sin(x1) + sin(x2^2)` | 2 | $[-1,1]^{2}$ | uniform i.i.d. | \cite{udrescu2020,vladislavleva2009,keijzer2003,mundhenk2021} | yes |
| III.10.19 | `mom * sqrt(Bx^2 + By^2 + Bz^2)` | 4 | $[1,5]^{4}$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| II.11.3 | `q*Ef / (m*(omega_0^2 - omega^2))` | 5 | $[1,5]$, $[1,5]$, $[1,3]$, $[3,5]$, $[1,2]$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| I.13.12 | `G*m1*m2*(1/r2 - 1/r1)` | 5 | $[1,5]^{5}$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| I.44.4 | `n*kb*T*ln(V2/V1)` | 5 | $[1,5]^{5}$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| R1 | `(x+1)^3 / (x^2 - x + 1)` | 1 | $[-1,1]$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| Pagie-2 | `1/(1+x^(-4)) + 1/(1+y^(-4)) + 1/(1+z^(-4))` | 3 | $[-5,5]^{3}$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| Liv-4 | `ln(x+1) + ln(x^2+1) + ln(x)` | 1 | $[0.1,10]$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| Liv-19 | `ln(x^2+x) + ln(x^3+x)` | 1 | $[0.1,10]$ | uniform i.i.d. | \cite{udrescu2020,pagie1997,mundhenk2021} | yes |
| Strogatz-bacres1 | `20 - x - x*y/(1 + 0.5*x^2)` | 2 | $[3.4854,16.3302]$, $[9.79266,54.4452]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-bacres2 | `10 - x*y/(1 + 0.5*x^2)` | 2 | $[3.4854,16.3302]$, $[9.79266,54.4452]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-barmag1 | `0.5*sin(x - y) - sin(x)` | 2 | $[4.11849,5.98626]$, $[0.200012,6.64978]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-barmag2 | `0.5*sin(y - x) - sin(y)` | 2 | $[4.11849,5.98626]$, $[0.200012,6.64978]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-glider1 | `-0.05*x^2 - sin(y)` | 2 | $[0.162555,5.60327]$, $[-1.32283,28.7652]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-glider2 | `x - cos(y)/x` | 2 | $[0.162555,5.60327]$, $[-1.32283,28.7652]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-lv1 | `3*x - 2*x*y - x^2` | 2 | $[1.51029e-05,8]$, $[0.000164748,3]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-lv2 | `2*y - x*y - y^2` | 2 | $[1.51029e-05,8]$, $[0.000164748,3]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-predprey1 | `x*(4 - x - y/(1 + x))` | 2 | $[0.00525005,6.58096]$, $[2.23001,11.6491]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-predprey2 | `y*(x/(1 + x) - 0.075*y)` | 2 | $[0.00525005,6.58096]$, $[2.23001,11.6491]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-shearflow1 | `cot(y)*cos(x)  [written cos(y)*cos(x)/sin(y)]` | 2 | $[-4.3501,3.59983]$, $[-2.74699,2.14721]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-shearflow2 | `(cos(y)^2 + 0.1*sin(y)^2)*sin(x)` | 2 | $[-4.3501,3.59983]$, $[-2.74699,2.14721]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-vdp1 | `10*(y - (1/3)*(x^3 - x))` | 2 | $[-1.19935,1.93737]$, $[-0.197026,0.927609]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| Strogatz-vdp2 | `-(1/10)*x` | 2 | $[-1.19935,1.93737]$, $[-0.197026,0.927609]$ | published simulation, random split, 300/100 | \cite{lacava2016,lacava2021,strogatz2014} | yes |
| I.12.2 | `q1*q2*r/(4*pi*epsilon*r**3)` | 4 | $[1,5]^{4}$ | uniform i.i.d. | \cite{udrescu2020,lacava2021} | yes |
| II.34.29a | `q*h/(4*pi*m)` | 3 | $[1,5]^{3}$ | uniform i.i.d. | \cite{udrescu2020,lacava2021} | yes |
| II.34.29b | `g_*mom*B*Jz/(h/(2*pi))` | 5 | $[1,5]^{5}$ | uniform i.i.d. | \cite{udrescu2020,lacava2021} | yes |
| III.19.51 | `-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)` | 5 | $[1,5]^{5}$ | uniform i.i.d. | \cite{udrescu2020,lacava2021} | yes |
| III.4.32 | `1/(exp((h/(2*pi))*omega/(kb*T))-1)` | 4 | $[1,5]^{4}$ | uniform i.i.d. | \cite{udrescu2020,lacava2021} | yes |
| test_4 | `sqrt(2/m*(E_n-U-L**2/(2*m*r**2)))` | 5 | $[1,3]$, $[8,12]$, $[1,3]$, $[1,3]$, $[1,3]$ | uniform i.i.d. | \cite{udrescu2020,lacava2021} | yes |
