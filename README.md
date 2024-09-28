# This repository hosts code to implement coverage path planning for unstructured environments. #

## Steps to setup and run and example
#### 1. Install Conda using the official website. [Conda Installation on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). Note : Use the instruction for your operating system.
#### 2. Create a conda environment with Python 3.11.5
    ```
    conda create -n <env_name> python=3.11.5
    ```
<env_name> is the name of the conda environment. Any user defined name can be used, here SKC is used for the sake of example.

#### 3. Activate the conda environment
    ```
    conda activate SKC
    ```

#### 4. Install Dependencies

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

#### 5. Clone the repository

#### 6. Activate the conda environment and open the cloned repository in it.
    ```
    conda activate SKC
    ```
    ```
    cd {clone location}/SKC
    ```

#### 7. Run the code
    ```
    python3 main.py
    ```

#### 8. Deactivate the conda 
    ```
    conda deactivate SKC
    ```

## Codemap

* main.py: Run examples using this script

* algorithms: This folder hosts source code for various coverage path planning strategies 

* docs: Hosts plots and illustrations

* sensors: Hosts sensor model used by coverage path planning strategies

* workspace: Hosts script to create grid workspace with obstacles

* environment.yml: Conda environment file
