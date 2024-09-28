This repository hosts code to implement coverage path planning for unstructured environments.

### Steps to setup and run and example
#### Install Conda using the official website. [Conda Installation on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html). Note : Use the instruction for your operating system.
#### Create a conda environment with Python 3.11.5
    ```
    conda create -n <env_name> python=3.11.5
    ```
<env_name> is the name of the conda environment. Any user defined name can be used, here SKC is used for the sake of example.

#### Activate the conda environment
    ```
    conda activate SKC
    ```
#### Install Dependencies

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
#### Clone the repository

#### Activate the conda environment and open the cloned repository in it.
    ```
    conda activate SKC
    ```
    ```
    cd {clone location}/SKC
    ```
#### Run the code
    ```
    python3 main.py
    ```

#### Deactivate the conda 
    ```
    conda deactivate SKC
    ```

### Codemap

#### main.py : Source code for the project

#### metric.py :  Calculating and Plotting Metric for the proposed rerouting strategy

#### Illustrations folder contains illustrations used in the paper
