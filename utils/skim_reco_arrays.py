# system
import os
import sys
import argparse
import logging
import pickle
import multiprocessing as mp
from functools import partial
from collections import Counter

# externals 
import uproot
import pandas as pd
import awkward as ak
import numpy as np
import particle 
from particle import PDGID
from particle import Particle 
from particle.pdgid import is_meson

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', '--indir', type=str, default='/tigress/andresf/displaced_taus/gun_samples')
    add_arg('-o', '--outdir', type=str, default='../skims')
    add_arg('-v', '--verbose', type=bool, default=0)
    return parser.parse_args()

def get_arrays(path):
    directory = uproot.open(path)
    logging.info(f"Opening {path}")
    reco_trees = [key for key, value in directory.classnames().items()
                  if (value=='TTree' and 'recoT' in key)]
    logging.info(f"Available reco trees: {reco_trees}")
    gen_trees = [key for key, value in directory.classnames().items()
                 if (value=='TTree' and 'genT' in key)]
    logging.info(f"Available gen trees: {gen_trees}")
    reco_tree = directory[reco_trees[0]]
    gen_tree = directory[gen_trees[0]]
    logging.info(f"Selected {reco_trees[0]} and {gen_trees[0]}")
    reco_keys = ['rec_hits_ee_detid', 'rec_hits_ee_energy', 
                 'rec_hits_ee_ix', 'rec_hits_ee_iy', 'rec_hits_ee_time',
                 'rec_hits_ee_rho', 'rec_hits_ee_eta', 'rec_hits_ee_phi', 
                 'rec_hits_eb_detid', 'rec_hits_eb_energy', 
                 'rec_hits_eb_ieta', 'rec_hits_eb_iphi', 'rec_hits_eb_time',
                 'rec_hits_eb_rho', 'rec_hits_eb_eta', 'rec_hits_eb_phi', 
                 'rec_hits_es_detid', 'rec_hits_es_energy', 
                 'rec_hits_es_six', 'rec_hits_es_siy', 'rec_hits_es_time',
                 'rec_hits_es_rho', 'rec_hits_es_eta', 'rec_hits_es_phi', 
                 'rec_hits_hb_detid', 'rec_hits_hb_energy', 
                 'rec_hits_hb_ieta', 'rec_hits_hb_iphi', 'rec_hits_hb_time',
                 'rec_hits_hb_rho', 'rec_hits_hb_eta', 'rec_hits_hb_phi', 
                 'rec_hits_he_detid', 'rec_hits_he_energy',
                 'rec_hits_he_ieta', 'rec_hits_he_iphi', 'rec_hits_he_time',
                 'rec_hits_he_rho', 'rec_hits_he_eta', 'rec_hits_he_phi',
                 'rec_hits_hf_detid', 'rec_hits_hf_energy', 
                 'rec_hits_hf_ieta', 'rec_hits_hf_iphi', 'rec_hits_hf_time',
                 'rec_hits_hf_rho', 'rec_hits_hf_eta', 'rec_hits_hf_phi', 
                 'rec_hits_ho_detid', 'rec_hits_ho_energy', 
                 'rec_hits_ho_ieta', 'rec_hits_ho_iphi', 'rec_hits_ho_time',
                 'rec_hits_ho_rho', 'rec_hits_ho_eta', 'rec_hits_ho_phi',
                 'reco_PF_n_jets', 'reco_PF_jet_pt', 'reco_PF_jet_eta', 
                 'reco_PF_jet_phi']
    gen_keys = ['gen_ID', 'gen_pt', 'gen_eta', 'gen_phi', 'gen_mass', 
                'gen_energy', 'gen_vxy', 'gen_vz']
    sim_keys = ['sim_hits_ee_detid', 'sim_hits_ee_energy',
                'sim_hits_ee_ix', 'sim_hits_ee_iy', 'sim_hits_ee_time',
                'sim_hits_ee_rho', 'sim_hits_ee_eta', 'sim_hits_ee_phi', 
                'sim_hits_eb_detid', 'sim_hits_eb_energy', 
                'sim_hits_eb_ieta', 'sim_hits_eb_iphi', 'sim_hits_eb_time',
                'sim_hits_eb_rho', 'sim_hits_eb_eta', 'sim_hits_eb_phi', 
                'sim_hits_es_detid', 'sim_hits_es_energy',
                'sim_hits_es_six', 'sim_hits_es_siy', 'sim_hits_es_time',
                'sim_hits_es_rho', 'sim_hits_es_eta', 'sim_hits_es_phi',
                'sim_hits_hb_detid', 'sim_hits_hb_energy', 
                'sim_hits_hb_iphi', 'sim_hits_hb_ieta',
                'sim_hits_he_detid', 'sim_hits_he_energy',
                'sim_hits_he_iphi', 'sim_hits_he_ieta',
                'sim_hits_hf_detid', 'sim_hits_hf_energy',
                'sim_hits_hf_iphi', 'sim_hits_hf_ieta',
                'sim_hits_ho_detid', 'sim_hits_ho_energy',
                'sim_hits_ho_iphi', 'sim_hits_ho_ieta']
    
    reco_data = reco_tree.arrays(reco_keys)
    gen_data = gen_tree.arrays(sim_keys + gen_keys)
    return {'gen_arrays': gen_data, 
            'reco_arrays': reco_data}

def main():
    args = parse_args()
 
    # set up logging 
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')
    
    # find the input files
    all_files = os.listdir(args.indir)
    file_prefixes = [f.split('.')[0].split('_')[-1] for f in all_files
                     if 'ntuple' in f]
    file_paths = [os.path.join(args.indir, f) for f in all_files
                  if 'ntuple' in f]
    n_files = len(file_paths)
    
    # multi-threaded file loading 
    with mp.Pool(processes=n_files) as pool:
        dataset = pool.map(get_arrays, file_paths)
        dataset = {file_prefixes[i]: dataset[i] for i in range(n_files)}
    
    for name, data in dataset.items():
        pickle.dump(data, open(os.path.join(args.outdir, name)+'.p', 'wb'))


if __name__ == '__main__':
    main()
