# LyaRTAnalytical

A Python module computing Lyman-alpha spectra with radiative transfer (RT) under simplified geometries, recoil, and velocity gradients. 

## Installation

Please follow the following codes for installation: 

```bash
git clone https://github.com/PengfeiLiAstro/LyaRTAnalytical.git
cd LyaRTAnalytical
pip install -e .
```
## Repository Structure

A quick overview of the package layout:
```
LyaRTAnalytical/
|-- lyart_spec/               # Core Python package
|   |-- __init__.py           # Exposes main functions
|   |-- lyaspec.py            # Main functions for evaluating Lya spectra. 
|   |-- utils.py              # Sub routines for evaluating Lya spectra
|-- lyart_spec_example.ipynb  # Jupyter-notebook. Usage examples. 
|-- .gitignore                # Untracked files
|-- LICENSE                   # MIT License
|-- pyproject.toml            # Package and dependency configuration
|-- README.md                 # Project documentation
```
## Usage
### Examples in jupyter-notebook
Check the jupyter-notebook [lyart_spec_example.ipynb](lyart_spec_example.ipynb) for usage examples. More details can be found in Li & Zheng (2026). 

### Function description
There are three main functions for evaluating Lya spectra with RT, `lya_spec_cls` using closed-form solutions/formulae, `lya_spec_ses` using series solutions, and `lya_spec_sim` using cached RT simulations. 

Here is a brief description of the input parameters that all three functions share. 

+ `tau0 (float)`: Optical depth
+ `taus2tau0 (float)`: The ratio of source optical depth to cloud optical depth, between 0.0 and 1.0. 
+ `xi (float)`: Initial frequency.  
+ `geometry (str)`: `"sla"`, `"cyl"`, `"sph"` for slab, cylindrical, and spherical geometry, respectively.  
+ `v2b (float)`: The ratio of cloud edge velocity to thermal velocity.  
+ `recoil_flag (bool)`: `True`: with recoil; `False`: without recoil.  
+ `T (float)`: Temperature of the gas cloud in K.

Here is a brief description of the additional input parameter for `lya_spec_cls`. 

+ `x (numpy.ndarray)`: A 1D array of frequency parameter.

Here is a brief description of the additional input parameters for `lya_spec_ses`. 

+ `x (numpy.ndarray)`: A 1D array of frequency parameter.

+ `Nses (int)`: The number of terms to be evaluated in the series. For Nses=1000, the series solution will be evaluated from n=  0 to n=999, 1000 terms in total. 

Here are the usage examples for each of the three functions. 

```python
# Example for lya_spec_cls.

import lyart_spec as las
import numpy as np

# Set the frequency parameter for which emergent Lya intensity is evaluated. 
x = np.linspace(-50, 50, 201)

xcls, ycls = las.lya_spec_cls(x = x, tau0 = 1e5, taus2tau0 = 0.0, xi = 0.0, geometry = "cyl", v2b = 0.0, recoil_flag = False, T = 10.0)
```

```python
# Example for lya_spec_ses.

import lyart_spec as las
import numpy as np

# Set the frequency parameter for which emergent Lya intensity is evaluated. 
x = np.linspace(-50, 50, 201)

xses, yses = las.lya_spec_ses(x = x, tau0 = 1e5, taus2tau0 = 0.0, xi = 0.0, geometry = "cyl", v2b = 0.0, recoil_flag = False, T = 10.0, Nses = 1000)
```

```python
# Example for lya_spec_sim.

import lyart_spec as las

xsim, ysim = las.lya_spec_sim(tau0 = 1e5, taus2tau0 = 0.0, xi = 0.0, geometry = "cyl", v2b = 0.0, recoil_flag = False, T = 10.0)
```

### Available parameters of simulated Lya spectra with RT

Only certain discrete values of the input parameters have corresponding RT simulations. When using `lya_spec_sim`, passing parameter values other than the available ones will raise an error. Here is a summary of the available input parameter values. 

**Simulation set `Sim_basic`**: varying `tau0`, `geometry`, and `recoil_flag`. 

```
Varying parameters:
tau0 = 1e4, 1e5, 1e6
geometry = "sla", "cyl", "sph"
recoil_flag = True, False

Fixed parameters:
taus2tau0 = 0.0
xi = 0.0
v2b = 0.0
T = 10.0
```

**Simulation set `Sim_taus2tau0`**: varying `taus2tau0`, `geometry`, and `recoil_flag`. 

```
Varying parameters: 
taus2tau0 = 0.3, 0.6, 0.9
geometry = "sla", "cyl", "sph"
recoil_flag = True, False

Fixed parameters: 
tau0 = 1e5
xi = 0.0
v2b = 0.0
T = 10.0
```

**Simulation set `Sim_xi`**: varying `xi`, `geometry`, and `recoil_flag`. 

```
Varying parameters: 
xi = 4.0, 8.0, 12.0
geometry = "sla", "cyl", "sph"
recoil_flag = True, False

Fixed parameters: 
tau0 = 1e5
taus2tau0 = 0.0
v2b = 0.0
T = 10.0
```

**Simulation set `Sim_vgrad`**: varying `tau0`, `taus2tau0`, `xi`, `geometry`, and `v2b`. 

```
Varying parameters: 
log10(tau0): range from 4 to 6 with interval of 0.2.
geometry = "sla", "cyl", "sph"
log10(v2b): range from -1 to 2 with interval of 0.2. 

Fixed parameters: 
taus2tau0 = 0.0
xi = 0.0
recoil_flag = False
T = 10.0
```

**Simulation set `Sim_vgrad_tt_xi`**: varying `tau0`, `taus2tau0`, `xi`, `geometry`, and `v2b`. 

```
Varying parameters: 
tau0 = 1e4, 1e5, 1e6
taus2tau0 = 0.3, 0.6, 0.9
xi = 4.0, 8.0, 12.0
geometry = "sla", "cyl", "sph"
log10(v2b) = -0.6, 0.0, 0.6, 1.2

Fixed parameters: 
recoil_flag = False
T = 10.0
```

**Simulation set `Sim_vgradT4`**: varying `tau0`, `geometry`, and `v2b`. (In previous simulation sets, temperature `T` is set at 10 K. In this set, the temperature `T` is set at 10000 K. ) 

```
Varying parameters: 
tau0 = 1e4*sqrt(1000), 1e5*sqrt(1000), 1e6*sqrt(1000)
geometry = "sla", "cyl", "sph"
log10(v2b) = -0.6, 0.0, 0.6, 1.2

Fixed parameters: 
taus2tau0 = 0.0
xi = 0.0
recoil_flag = False
T = 10000.0
```

### Additional usage caution

For function `lya_spec_cls` using closed-form solutions/formulae, 

+ The closed-form solution/formula for non-zero `v2b` is only valid for `taus2tau0=0` and `xi=0`. Switch to series solutions using function `lya_spec_ses` for non-zero `v2b`, `taus2tau0`, and `xi`. Notice that the series solution for non-zero `v2b` is not accurate for `abs(v2b)>1`. 
+ The closed-form solution/formula for `abs(v2b)>100` is not tested again Monte Carlo radiative transfer simulation. 
+ The closed-form solution/formula for `log10(av*tau0)` outside [2.2, 4.2] is not tested again Monte Carlo radiative transfer simulation. 
+ The recoil correction for non-zero `v2b` is not tested again Monte Carlo radiative transfer simulation. 

For function `lya_spec_ses` using series solutions, 

+ The series solution for `abs(v2b)>1` is no longer accurate. Switch to closed-form solutions/formulae using function `lya_spec_cls` for better accuracy. Notice that the closed-form solution/formula for non-zero `v2b` is only functional for `taus2tau0=0` and `xi=0`. 
+ The recoil correction for non-zero `v2b` is not tested again Monte Carlo radiative transfer simulation. 

For function `lya_spec_sim` using cached simulated Lya spectra, 

+ Only certain discrete values of the input parameters have corresponding RT simulations. When using `lya_spec_sim`, input parameters other than the available ones will raise an error. Check Section **Available parameters of simulated Lya spectra with RT** for available parameters. 

### Author & Email

Pengfei Li. Email: Pengfei.Li@utah.edu.

### Citations

Please cite Li & Zheng (2026) if you use the package. 

## License

This code is open-source and distributed under the [MIT License](LICENSE).

