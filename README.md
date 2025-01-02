December 29, 2024
Version: 1.0
Co-author: M. Mommert, M.C. Volk, C. Bauer

## LPINN

## How to install?

#### Requirements:

LPINN requires Python 3.10 installed with pip. It was developed and tested on Python 3.10.4 which can be downloaded at: https://www.python.org/downloads/release/python-3104/

#### Installation:

1) download the LPINN project to a desired location

2) install python 3.10.4 and pip on Windows or Linux

3) open your terminal, navigate into the proPTV folder location and set up a venv (virtual environment) if you want to use it, else jump to point 5.

  `python -m venv venv`

4) active the venv one Windows with 

  `cd venv/Scripts activate`
   
   and on Linux with:

  `source venv/bin/activate`

5) install the required python packages

  `pip install -r requirements.txt`
  
## How to use LPINN?

1) set up the config file, e.g. config_1.py

2) prepare dataset (instructions follow in a make_data.py soon), but until now test dataset available.

3) run in a terminal with active venv: python ./code/main.py config/config_1.py 

## How to cite?

When LPINN is useful for your scientific work, you may cite us as:

[1] M. Mommert, R. Barta, C. Bauer, M. C. Volk, and C. Wagner. Periodically activated physics-informed neural networks for assimilation tasks for three-dimensional Rayleigh-Bénard convection. Computers and Fluids, 2024. https://doi.org/10.1016/j.compfluid.2024.106419

[2] R. Barta, M.-C. Volk, M. Mommert, C. Bauer and C. Wagner. Comparing assimilation techniques for pressure and temperature fields in Rayleigh-Bénard convection. Springer notes on Numerical Fluid Mechanics and Multidisciplinary Design - New Results in Numerical and Experimental Fluid Mechanics XV, 2024.

and include the licence file in all copies with modifications or other code that uses parts of the LPINN framework.

## Contact

If you have a question or need help installing LPINN or fixing a bug you have found, please contact me: robin.barta@dlr.de

I am happy to help and look forward to meeting you.
