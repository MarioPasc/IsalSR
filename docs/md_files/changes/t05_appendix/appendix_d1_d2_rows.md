| Problem | Expression | $n$ | Range | Sampling | Source | GT |
|---|---|---|---|---|---|---|
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
