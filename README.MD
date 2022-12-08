###
# Disclaimer
###

This is a test environment and that does not provide any warranty, is intended for internal use, and is currently under development.
Use at your own risks.

###
# Setup in anaconda environment
###

Install [miniconda](https://conda.io/miniconda.html)

Update conda (might need to open prompt with admin rights)<br>
```
conda activate base
conda update -n base -c defaults conda
```

Create environment with python 3.8
```
conda create --name fes python=3.8
```

Go to target environment
```
conda activate fes
```

If exists, remove conda-forge channel to speed up installations
```python
# conda config --show channels # to show current available channels
# start by removing conda-forge to speed up the base installation
conda config --remove channels conda-forge
```

Install required libraries for our scripts
```
cd "Documents/Rehastim/libs" ; pip install sciencemode3_cffi-1.0.0-cp38-cp38-win_amd64.whl
conda install -c conda-forge spyder=5.4
conda install -c conda-forge qtpy termcolor python-kaleido beautifultable
conda install numpy matplotlib pandas scikit-learn plotly pyserial git scipy
```

##### Tips

- Some USB ports don't seem to work with the stimulator


