# PackDock: a Diffusion Based Side Chain Packing Model for Flexible Protein-Ligand Docking 

# How to Get this Running...

- Build environment - see `env.yaml` for my exact working environment. Roughly, I followed Asma's install:

```bash
conda create --name packdock python=3.8
conda activate packdock
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
conda install conda-forge::openbabel
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
```
Then added:
```bash
mamba install wandb
mamba install "pandas<2"
```
because the code breaks with pandas2+ due to the removal of the `append` method for dataframes. Also, make sure [these lines](https://github.com/Zhang-Runze/PackDock/blob/aa4ade6985ef242d9622375a9431e8a59d506673/utils/utils.py#L397) point to your gnina if running the gnina script. Also, download and unzip the model weights from zenodo (link in orginal instructions).

- Some comments on various broken things in the code.
    - The code tends to catch errors, then ignore them e.g. [here](https://github.com/Zhang-Runze/PackDock/blob/aa4ade6985ef242d9622375a9431e8a59d506673/docking_evaluate_gnina_apo.py#L358). It's best to remove the try/ except blocks which catch everything because this makes things much harder to debug.
    - [`randomize_position`](https://github.com/Zhang-Runze/PackDock/blob/aa4ade6985ef242d9622375a9431e8a59d506673/utils/sampling.py#L10) takes three positional arguments, but is called in the gnina script with 4. I've fixed this.
    - This line in [`set_time`](https://github.com/Zhang-Runze/PackDock/blob/aa4ade6985ef242d9622375a9431e8a59d506673/utils/diffusion_utils.py#L73) adds the key "ligand" to the complex graph, which breaks things later if it's not already there. I've fixed this.
    - Many other broken things.

- What I've tried and where I've got to.
    - Running the original `apo2holo_datasets`. Running the original `get_pocket.py` with the original input, because the aligned inputs are not provided, and empty files are created. To fix this, you need to align the proteins to `refined.pdb` - see example commented out in `get_pocket.py`. After aligning, the script runs a couple of ligands before breaking due to an unregocnised atom. I ignored this and modified `docking_evaluate_gnina_apo.py` to run with a single working ligand (run with `python docking_evaluate_gnina_apo.py --rmsd`). This runs for a while until I hit:
    ```Traceback (most recent call last):
    File "docking_evaluate_gnina_apo.py", line 398, in <module>
        packing_result_path = side_chain_packing(complex_graphs)
    File "docking_evaluate_gnina_apo.py", line 203, in side_chain_packing
        data_list, confidence = sampling(data_list=data_list, model=model,
    File "/home/campus.ncl.ac.uk/nfc78/software/devel/PackDock/utils/sampling.py", line 44, in sampling
        side_score = model(complex_graph_batch)
    File "/home/campus.ncl.ac.uk/nfc78/miniforge3/envs/packdock/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
    File "/home/campus.ncl.ac.uk/nfc78/miniforge3/envs/packdock/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
        return forward_call(*args, **kwargs)
    File "/home/campus.ncl.ac.uk/nfc78/software/devel/PackDock/models/all_atom_score_model.py", line 228, in forward
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data)
    File "/home/campus.ncl.ac.uk/nfc78/software/devel/PackDock/models/all_atom_score_model.py", line 325, in build_lig_conv_graph
        data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['t'])
    File "/home/campus.ncl.ac.uk/nfc78/miniforge3/envs/packdock/lib/python3.8/site-packages/torch_geometric/data/storage.py", line 96, in __getattr__
        raise AttributeError(
    AttributeError: 'NodeStorage' object has no attribute 'node_t'
    ```
    See current output in `results/1wfcA_3HA8A`

    - Running the MERS input (copied from original Polaris submission -see [here](https://github.com/michellab/polaris-poses-challenge-fegrow-a3fe/tree/main/mers-040225/input/full_run-MERS)). I modified `get_pocket.py` and the layout of the input data to match what the script expects, and this appeared to process fine. However, PackDock raises an error if for `HID/P/E` residues, so I changed these all to `HIS` before extracting the pocket. Also, PackDock seems to only take deprotonated input, so I converted the renamed pdb with `obabel protein_his_renamed.pdb -O protein.pdb -d` (otherwise the pocket pdb is deprotonated while the original is protonated, which produces errors to do with finding rotatable torsions). I then ran this with `python docking_evaluate_gnina_apo_mers.py --rmsd`. This produces the same error as above.

- Comment - it's helpful to run scripts with `python -i -m pdb script.py` as this will put you into the interactive debugg




This repo contains a PyTorch implementation for the paper  PackDock: a Diffusion Based Side Chain Packing Model for Flexible Protein-Ligand Docking 

If you have any question, feel free to open an issue or reach out to us: [zhangrunze@simm.ac.cn](zhangrunze@simm.ac.cn)✉️.

by [Runze Zhang](https://github.com/Zhang-Runze)
# PackDock Overview
![](https://github.com/Zhang-Runze/PackDock/blob/main/figs/Method%20Overview.jpg)


# Setup Environment

We recommend setting up the environment using [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

Clone the current repo

    git clone https://github.com/Zhang-Runze/PackDock.git
    
This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):

    conda create --name packdock python=3.8
    conda activate packdock
    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    python -m pip install openbabel PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
    
Then you need to install a ligand conformation sampling algorithm (such as [AutoDock-Vina](https://github.com/ccsb-scripps/AutoDock-Vina), [gnina](https://github.com/gnina/gnina), [Vina-GPU](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0), etc.).
It's worth noting that PackDock offers a highly general flexible docking strategy, capable of integrating any ligand conformation sampling algorithm develop yourself or you might choose to use. This implies that employing more  advanced ligand conformation sampling algorithms could potentially lead to unexpectedly impressive docking results.


# Data preprocess
First run:

    python get_pocket/get_pocket.py
    
Then proceed with further processing using [ADFR-Suite](https://ccsb.scripps.edu/adfr/downloads/) and OpenBabel.


# Model weights

The model weights are available on [zenodo](https://zenodo.org/records/10851699).


# Running PackPocket on your protein

Protein pocket packing:

    python -m evaluate_protein --run_name Protein pocket packing --inference_steps 20 --samples_per_complex 10 --batch_size 10 --actual_steps 20 --cache_path data/cacheTest_CASP14 --no_final_step_noise --data_dir data/CASP/CASP14/ --split_path data/splits/CASP14_list --save_visualisation

Ligand based protein pocket packing:

    python -m evaluate_ligand_based_protein --run_name Ligand based protein pocket packing --inference_steps 20 --samples_per_complex 10 --batch_size 10 --actual_steps 20 --cache_path data/cacheTest_PDBBind --no_final_step_noise --data_dir data/PDBBind_processed --split_path data/splits/timesplit_test_no_rec_overlap --save_visualisation

# Running PackDock on your complex

    python docking_evaluate_gnina_apo.py --rmsd

or

    python docking_evaluate_vina_apo.py --rmsd
    

# Retraining PackPocket
Download the data([BC40](https://zenodo.org/) or [PDBbind](https://zenodo.org/records/6408497)) and place it as described in the "Dataset" section above.

### Training a model yourself and using those weights
Train the PackPocket:

    python -m train_protein --run_name PackPocket_protein --test_sigma_intervals  --log_dir workdir --lr 1e-3 --batch_size 8 --ns 48 --nv 10 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 30.0 --atom_radius 5.0 --cross_distance_embed_dim 64 --distance_embed_dim 64 --sigma_embed_dim 64 --cross_max_distance 20 --num_dataloader_workers 36 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 100 --use_ema --scheduler_patience 30 --n_epochs 300 --all_atoms --num_worker 36 --no_torsion --data_dir data/bc40_pockets_processed/ --split_train data/splits/bc40_train_set --split_val data/splits/bc40_validation_set --split_test data/splits/bc40_test_set 

Train the ligand-based PackPocket:

    python -m train_ligand_based_protein --run_name PackPocket_ligand_based_protein --test_sigma_intervals  --log_dir workdir --lr 1e-3 --batch_size 8 --ns 48 --nv 10 --num_conv_layers 6 --dynamic_max_cross --scheduler plateau --scale_by_sigma --dropout 0.1 --remove_hs --c_alpha_max_neighbors 24 --receptor_radius 30.0 --atom_radius 5.0 --cross_distance_embed_dim 64 --distance_embed_dim 64 --sigma_embed_dim 64 --cross_max_distance 20 --num_dataloader_workers 1 --cudnn_benchmark --val_inference_freq 5 --num_inference_complexes 100 --use_ema --scheduler_patience 30 --n_epochs 300 --all_atoms --num_worker 36 --no_torsion --data_dir data/PDBBind_processed/ --split_train data/splits/timesplit_no_lig_overlap_train --split_val data/splits/timesplit_no_lig_overlap_val --split_test data/splits/timesplit_test_no_rec_overlap

The model weights are saved in the `workdir` directory.
