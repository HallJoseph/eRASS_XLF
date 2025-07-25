# eRASS_XLF
Plot the eRASS XLF to explore excess of galaxy clusters at L ~ 2e43 erg/s observed by Koens 2013 in WARPS XLF

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Environment initialisation](#environment-initialisation)
  - [First time setup](#first-time-setup)
  - [Activating the environment](#activating-the-environment)
  - [Adding further packages](#adding-further-packages)
  - [Deactivating the environment](#deactivating-the-environment)

## Environment initialisation

<details>
<summary>Click to view environment setup instructions</summary>

### First time setup

First we need to create the conda environment using the conda environment.yml config file

```bash
conda env create -f environment.yml 
```

### Activating the environment

Once the environment has been created, you need to activate

```bash
conda activate eRASS_XLF
```

### Adding further packages

These can be installed to the activated environment using the terminal (emcee example below), and remember to update environment.yml and git commit for future reference.

```bash
conda install -c conda-forge emcee
```

### Deactivating the environment

```bash
conda deactivate 
```

</details>
