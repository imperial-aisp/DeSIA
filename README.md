# DeSIA Public Release

Source code repository for the paper "Automated Privacy Risk Estimation of Limited Fixed Aggregate Statistics" by Yifeng Mao*, Bozhidar Stevanoski*, and Yves-Alexandre de Montjoye (* denotes equal contribution).

We introduce a framework and a method for attribute inference attacks against fixed aggregate statistics. 
Our source code is organized as follows:

<ul>
<li> <code>main.py</code>: The main entry point to run DeSIA. </li>
<li> <code>src</code>: Source code folder </li>
<ul>
    <li>
        <code>algo</code>: Attack methods, currently containing only the code for our method, DeSIA
    </li>
    <li>
        <code>asset</code>: Code to obtain and process the datasets and queries.
    </li>
    <li>
        <code>utils</code>: Utility functions, such as functions for data loading and sampling.
    </li>
</ul>

<li> <code>requirements.txt</code>: The python packages required for running the source code. </li>
</ul>

## 1. Python environment
The code has dependencies on common Python libraries. To run it, first please create a Python environment and install the dependencies in the `requirements.txt` file as follows:

```
conda create -n desia python=3.8 pip
conda activate desia
pip install -r requirements.txt  # Make sure that the cuda version is compatible with the installed pytorch version!
```

## 2. Third-party solver
The deterministic module of DeSIA uses a solver for the constraint integer programming problem. 
In particular, we use a third-party solver, [Gurobi](https://www.gurobi.com/). 
We gratefully acknowledge Gurobi for providing a free academic licence [for students and researchers](https://www.gurobi.com/academia/academic-program-and-licenses/).
Please download your academic license (gurobi.lic) from [Gurobi Portal](https://portal.gurobi.com/iam/login/), and move it under your home directory (~/).

## 3. Data
Run the two notebooks under ./src/asset to obtain the dataset and queries.

## 4. Develop your own method
You can develop your method by imporving upon our source code. 
<code>BaseAttack</code> is the superclass to instantiate all our attack methods, which is defined in <code>./src/algo/desia/base.py</code>. The only thing you need to do is creating a new child-class of <code>BaseAttack</code> and instantiating <code>attack</code>, <code>evaluate</code> function of that child-class. 

## 5. Run experiments
Switch to ./scripts and run the scripts under that folder to obtain empirical results in the paper.

## Acknowledgements and Citation
Some of the functions for generating the queries and evaluating them are reused and upgraded from [RAP-Rank](https://github.com/terranceliu/rap-rank-reconstruction/), [DP-Query-Release](https://github.com/terranceliu/dp-query-release), [QuerySnout](https://github.com/computationalprivacy/querysnout) and [QueryCheetah](https://github.com/computationalprivacy/querycheetah). We thank the authors for making open-sourcing their code.
