This repository hosts the implementation of the paper "Online Obstacle evasion for Space-Filling Curves". The strategy is implemented in Python 3.11.5. Usage of conda environment for setuping and running the code is highly recommended. Steps to setup the code is described initially, while the file structure and files are described towards the end

- Install Conda
1. Install Conda using the official website. [Conda Installation on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). Note : Use the instruction for your operating system.
2. Create a conda environment with Python 3.11.5
```
conda create -n SKC python=3.11.5
```
SKC is the name of the conda environment. Any user defined name can be used.

3. Activate the conda environment
```
conda activate SKC
```
4. Install Dependencies

    a. [igraph library](https://igraph.org/) for graph-related computations
    ```
    conda install -c conda-forge python-igraph=0.11.3
    ```
    b. [Matplotlib](https://matplotlib.org/) for visualization
    ```
     conda install -c conda-forge matplotlib=3.8.2
    ```
    c. [hilbertcurve](https://pypi.org/project/hilbertcurve/) for plotting Hilbert's curve
    ```
     conda install -c conda-forge hilbertcurve=2.0.5
    ```
    d. [Numba](https://numba.pydata.org/) for compile time computations to reduce the runtime. (Optional - user will have to comment the call to the library and @njits )
    ```
     conda install -c numba numba=0.59.0
    ```

5. Clone the repository

6. Activate the conda environment and open the cloned repository in it.
```
conda activate SKC
```
```
cd {clone location}/SKC
```

7. Run the code
```
python3 main.py
```

8. Deactivate the conda 
```
conda deactivate SKC
```

File description :

main.py : contains source code for the project

Illustrations folder contains illustrations used in the paper