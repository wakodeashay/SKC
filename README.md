# This repository hosts code to implement coverage path planning for unstructured environments using Hilbert's Space Filling curve. #

## Code setup 
#### 1. Install Conda using the official website. [Conda Installation on Linux](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

#### 2. Clone the repository
    ```
    git clone git@github.com:wakodeashay/SKC.git
    ```

#### 3. Activate the conda environment
    ```
    cd SKC
    conda create -f environment.yml
    ```

#### 4. Activate the conda environment a
    ```
    conda activate SKC
    ```

## Run example

#### 1. Run the code
    ```
    python3 main.py
    ```

#### 2. (Optional) Deactivate the conda 
    ```
    conda deactivate SKC
    ```

## Codemap

* main.py: Script to run examples

* algorithms: This folder hosts source code for various coverage path planning strategies 

* docs: Plots and Animation are saved here

* sensors: Hosts sensor model used by coverage path planning strategies

* workspace: Hosts script to create grid workspace with obstacles

* environment.yml: Conda environment file
