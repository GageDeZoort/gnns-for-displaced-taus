# system
import os
import sys
import logging
import pickle
import argparse
import multiprocessing as mp
from functools import partial
from datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# externals 
import torch
import awkward as ak
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader, Data
import particle 
from particle import PDGID
from particle import Particle 
from particle.pdgid import is_meson
pd.options.mode.chained_assignment = None  # default='warn'

def parse_args(args):
    """ parse the command line
    """
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', '--infile', type=str, default='../skims/data.p')
    add_arg('-o', '--outdir', type=str, default='../graphs')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('-n', '--n-workers', type=int, default=1)
    add_arg('--n-events', type=int, default=100)
    add_arg('--eps', type=int, default=2)
    add_arg('--ecal-min', type=float, default=0.025)
    add_arg('--hcal-min', type=float, default=0.025)
    add_arg('--sim-min', type=float, default=0.025)
    add_arg('--tau-pt-min', type=float, default=0)
    add_arg('--task', type=int, default=0)
    add_arg('--save-output', type=bool, default=False)
    add_arg('--statfile', type=str, default='')
    return vars(parser.parse_args(args)) # return as dictionary

def calc_dphi(phi1, phi2):
    """ computes phi2-phi1 given in range [-pi,pi]
    """
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    """ computes pseudorapidity given polar angle theta
    """
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def zero_div(a,b):
    """ divide, potentially by zero
    """
    if (b<1e-9): return 0
    return a/b

def filter_gen_arrays(data, args={}):
    """ shape gen arrays into convenient dictionary,
        add relevant quantities
    """
    # grab arguments parsed from command line
    
    # mask to select the tau in the event
    tau_mask = (abs(data['gen_ID'])==15) 
            
    # mask to select visible and invisible systems
    pdgID = data['gen_ID']
    vis_mask = ((abs(pdgID)==111) | # pi0
                (abs(pdgID)==211) | # pi+
                (abs(pdgID)==311) | # K0
                (abs(pdgID)==321) | # K+
                (abs(pdgID)==130) | # K0S
                (abs(pdgID)==310))  # K0L
    inv_mask = ((abs(pdgID)==12) |  # nu_e
                (abs(pdgID)==14) |  # nu_mu
                (abs(pdgID)==16))   # nu_tau

    tau = {'pt': data['gen_pt'][tau_mask][:,0],
           'eta': data['gen_eta'][tau_mask][:,0],
           'phi': data['gen_phi'][tau_mask][:,0],
           'mass': data['gen_mass'][tau_mask][:,0],
           'energy': data['gen_energy'][tau_mask][:,0],
           'vxy': data['gen_vxy'][tau_mask][:,0],
           'vz': data['gen_vz'][tau_mask][:,0]}

    vis = {'ID': data['gen_ID'][vis_mask],
           'pt': data['gen_pt'][vis_mask],
           'eta': data['gen_eta'][vis_mask],
           'phi': data['gen_phi'][vis_mask],
           'mass': data['gen_mass'][vis_mask],
           'energy': data['gen_energy'][vis_mask],
           'vxy': data['gen_vxy'][vis_mask],
           'vz': data['gen_vz'][vis_mask]}
    
    inv = {'pt': data['gen_pt'][inv_mask],
           'eta': data['gen_eta'][inv_mask],
           'phi': data['gen_phi'][inv_mask],
           'mass': data['gen_mass'][inv_mask],
           'energy': data['gen_energy'][inv_mask],
           'vxy': data['gen_vxy'][inv_mask],
           'vz': data['gen_vz'][inv_mask]}
    
    # filter sim hits below min energy in each subdetector 
    ee_mask = (data['sim_hits_ee_energy'] > args['sim_min'])
    eb_mask = (data['sim_hits_eb_energy'] > args['sim_min'])
    es_mask = (data['sim_hits_es_energy'] > args['sim_min'])
    hb_mask = (data['sim_hits_hb_energy'] > args['sim_min'])
    he_mask = (data['sim_hits_he_energy'] > args['sim_min'])
    ho_mask = (data['sim_hits_ho_energy'] > args['sim_min'])
    hf_mask = (data['sim_hits_hf_energy'] > args['sim_min'])

    # group simhit quantities by subdetector
    sim_hits =  {'ee': {'detid': data['sim_hits_ee_detid'][ee_mask], 
                        'energy': data['sim_hits_ee_energy'][ee_mask],
                        'ix': data['sim_hits_ee_ix'][ee_mask],
                        'iy': data['sim_hits_ee_iy'][ee_mask],
                    },
                 'eb': {'detid': data['sim_hits_eb_detid'][eb_mask],
                        'energy': data['sim_hits_eb_energy'][eb_mask],
                        'ieta': data['sim_hits_eb_ieta'][eb_mask],
                        'iphi': data['sim_hits_eb_iphi'][eb_mask],
                    },
                 'es': {'detid': data['sim_hits_es_detid'][es_mask],
                        'energy': data['sim_hits_es_energy'][es_mask],
                        'six': data['sim_hits_es_six'][es_mask],
                        'siy': data['sim_hits_es_siy'][es_mask],
                    },
                 'hb': {'detid': data['sim_hits_hb_detid'][hb_mask],
                        'energy': data['sim_hits_hb_energy'][hb_mask],
                        'ieta': data['sim_hits_hb_ieta'][hb_mask],
                        'iphi': data['sim_hits_hb_iphi'][hb_mask],
                    },
                 'he': {'detid': data['sim_hits_he_detid'][he_mask],
                        'energy': data['sim_hits_he_energy'][he_mask],
                        'ieta': data['sim_hits_he_ieta'][he_mask],
                        'iphi': data['sim_hits_he_iphi'][he_mask],
                    },
                 'ho': {'detid': data['sim_hits_ho_detid'][ho_mask],
                        'energy': data['sim_hits_ho_energy'][ho_mask],
                        'ieta': data['sim_hits_ho_ieta'][ho_mask],
                        'iphi': data['sim_hits_ho_iphi'][ho_mask],
                    },
                 'hf': {'detid': data['sim_hits_hf_detid'][hf_mask],
                        'energy': data['sim_hits_hf_energy'][hf_mask],
                        'ieta': data['sim_hits_hf_ieta'][hf_mask],
                        'iphi': data['sim_hits_hf_iphi'][hf_mask],
                    },
                }
       
    return {'tau': tau, 'vis': vis, 'inv': inv, 'sim_hits': sim_hits}

def filter_reco_arrays(data, args={}):
    """ shape reco arrays into convenient dictionary,
        add additional quantities
    """
    
    ecal_min = args['ecal_min']
    hcal_min = args['hcal_min']
    
    # filter rec_hits below a given energy in each subdetector
    ee_mask = (data['rec_hits_ee_energy'] > ecal_min)
    eb_mask = (data['rec_hits_eb_energy'] > ecal_min)
    es_mask = (data['rec_hits_es_energy'] > ecal_min)
    hb_mask = (data['rec_hits_hb_energy'] > hcal_min)
    he_mask = (data['rec_hits_he_energy'] > hcal_min)
    hf_mask = (data['rec_hits_hf_energy'] > hcal_min)
    ho_mask = (data['rec_hits_ho_energy'] > hcal_min)

    rec_hits = {'ee': {'detid': data['rec_hits_ee_detid'][ee_mask], 
                       'energy': data['rec_hits_ee_energy'][ee_mask],
                       'ix': data['rec_hits_ee_ix'][ee_mask],
                       'iy': data['rec_hits_ee_iy'][ee_mask],
                       'rho': data['rec_hits_ee_rho'][ee_mask],
                       'eta': data['rec_hits_ee_eta'][ee_mask],
                       'phi': data['rec_hits_ee_phi'][ee_mask],
                      },
                'eb': {'detid': data['rec_hits_eb_detid'][eb_mask],
                       'energy': data['rec_hits_eb_energy'][eb_mask],
                       'ieta': data['rec_hits_eb_ieta'][eb_mask],
                       'iphi': data['rec_hits_eb_iphi'][eb_mask],
                       'rho': data['rec_hits_eb_rho'][eb_mask],
                       'eta': data['rec_hits_eb_eta'][eb_mask],
                       'phi': data['rec_hits_eb_phi'][eb_mask],
                      },
                'es': {'detid': data['rec_hits_es_detid'][es_mask],
                       'energy': data['rec_hits_es_energy'][es_mask],
                       'six': data['rec_hits_es_six'][es_mask],
                       'siy': data['rec_hits_es_siy'][es_mask],
                       'rho': data['rec_hits_es_rho'][es_mask],
                       'eta': data['rec_hits_es_eta'][es_mask],
                       'phi': data['rec_hits_es_phi'][es_mask],
                      },
                'hb': {'detid': data['rec_hits_hb_detid'][hb_mask],
                       'energy': data['rec_hits_hb_energy'][hb_mask],
                       'ieta': data['rec_hits_hb_ieta'][hb_mask],
                       'iphi': data['rec_hits_hb_iphi'][hb_mask],
                       'rho': data['rec_hits_hb_rho'][hb_mask],
                       'eta': data['rec_hits_hb_eta'][hb_mask],
                       'phi': data['rec_hits_hb_phi'][hb_mask],
                        },
                'he': {'detid': data['rec_hits_he_detid'][he_mask],
                       'energy': data['rec_hits_he_energy'][he_mask],
                       'ieta': data['rec_hits_he_ieta'][he_mask],
                       'iphi': data['rec_hits_he_iphi'][he_mask],
                       'rho': data['rec_hits_he_rho'][he_mask],
                       'eta': data['rec_hits_he_eta'][he_mask],
                       'phi': data['rec_hits_he_phi'][he_mask],
                        },
                'hf': {'detid': data['rec_hits_hf_detid'][hf_mask],
                       'energy': data['rec_hits_hf_energy'][hf_mask],
                       'ieta': data['rec_hits_hf_ieta'][hf_mask],
                       'iphi': data['rec_hits_hf_iphi'][hf_mask],
                       'rho': data['rec_hits_hf_rho'][hf_mask],
                       'eta': data['rec_hits_hf_eta'][hf_mask],
                       'phi': data['rec_hits_hf_phi'][hf_mask],
                      },
                'ho': {'detid': data['rec_hits_ho_detid'][ho_mask],
                       'energy': data['rec_hits_ho_energy'][ho_mask],
                       'ieta': data['rec_hits_ho_ieta'][ho_mask],
                       'iphi': data['rec_hits_ho_iphi'][ho_mask],
                       'rho': data['rec_hits_ho_rho'][ho_mask],
                       'eta': data['rec_hits_ho_eta'][ho_mask],
                       'phi': data['rec_hits_ho_phi'][ho_mask],
                      },
           }
    
    jets = {'n': data['reco_PF_n_jets'],
            'pt': data['reco_PF_jet_pt'],
            'eta': data['reco_PF_jet_eta'],
            'phi': data['reco_PF_jet_phi']}
    
    # derived rec_hit quantities
    for sd in rec_hits.keys():
        rec_hits[sd]['x'] = rec_hits[sd]['rho'] * np.cos(rec_hits[sd]['phi'])
        rec_hits[sd]['y'] = rec_hits[sd]['rho'] * np.sin(rec_hits[sd]['phi'])
        rec_hits[sd]['z'] = (rec_hits[sd]['rho'] / 
                             np.tan(2*np.arctan(np.exp(-1*rec_hits[sd]['eta']))))
        
    return {'rec_hits': rec_hits, 'jets': jets}

def debug_event(i, sim, rec, vis, zoom_rec=False):
    sim_detid = sim['eb']['detid'][i]
    sim_ieta = sim['eb']['ieta'][i]
    sim_iphi = sim['eb']['iphi'][i]
    sim_energy = sim['eb']['energy'][i]
    rec_detid = rec['eb']['detid'][i]
    rec_ieta = rec['eb']['ieta'][i]
    rec_iphi = rec['eb']['iphi'][i]
    rec_energy = rec['eb']['energy'][i]
    vis_ID = vis['ID'][i]
    vis_eta = vis['eta'][i]
    vis_phi = vis['phi'][i]
    vis_ieta = [0]*len(vis_eta)
    vis_iphi = [0]*len(vis_phi)
    eb_ieta = np.linspace(-1.479, 1.479, 171)
    eb_iphi = np.linspace(-np.pi, np.pi, 361)
    for v in range(len(vis_eta)):
        for idx, (low, high) in enumerate(list(zip(eb_ieta[:-1], eb_ieta[1:]))):
            if (vis_eta[v] < high) and (vis_eta[v] >= low):
                print('eta:', vis_eta[v], 'in bin', idx-85)
                vis_ieta[v] = idx-85
        for idx, (low, high) in enumerate(list(zip(eb_iphi[:-1], eb_iphi[1:]))):
            if (vis_phi[v] < high) and (vis_phi[v] >= low):
                print('phi:', vis_phi[v], 'in bin', idx)
                vis_iphi[v] = idx
                      
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6), dpi=100)
    plot_cluster(axs[0], sim_ieta, sim_iphi, np.exp(sim_energy))
    plot_cluster(axs[1], rec_ieta, rec_iphi, np.exp(rec_energy))
    xlim = [axs[0].get_xlim()[0]-30, axs[0].get_xlim()[1]+30]
    ylim = [axs[0].get_ylim()[0]-30, axs[0].get_ylim()[1]+30]
    for v in range(len(vis_ieta)):
        print(vis_ieta[v], vis_iphi[v])
        axs[1].add_patch(Ellipse(xy=(vis_ieta[v], vis_iphi[v]), 
                                 width=3, height=3, color="cyan", 
                                 fill=False, lw=1, zorder=10))
    if zoom_rec:
        axs[1].set_xlim(xlim)
        axs[1].set_ylim(ylim)
    plt.show()

def phi_neighbors(iphi_1, iphi_2, eps=2):
    """ decide if hits are neighbors in binned phi space
        while accounting for the "wrap-around" (branch cut) at 2pi
    """
    diff = abs(iphi_1 - iphi_2)
    return ((diff<(eps+1)) | ((360-diff)<(eps+1))) 

def truth_match_hits(e, sim, rec, args={}):
    # extract args
    eps = args['eps']
    sim_min = args['sim_min']
    
    # store summary statistics for subsequent analysis
    stats = {'total_sim_hits':  0,
             'matched_sim_hits': 0,
             'total_rec_hits': 0,
             'matched_rec_hits': 0}
    
    # loop over subdetectors, build dataframe in each
    df_sim_list, df_rec_list = [], []
    for s, sd in enumerate(sim.keys()): 
        #logging.info(f"Subdetector map: {s}-->{sd}")

        # grab binned coordinates for the subdetector
        ix = 'ix' if 'ix' in sim[sd].keys() else 'ieta'
        if 'six' in sim[sd].keys(): ix = 'six'
        iy = 'iy' if 'iy' in sim[sd].keys() else 'iphi'
        if 'siy' in sim[sd].keys(): iy = 'siy'
        
        # build dataframe for rec hits
        n_rec = len(rec[sd]['detid'][e])
        df_rec = pd.DataFrame({'x': rec[sd]['x'][e], 'ix': rec[sd][ix][e],
                               'y': rec[sd]['y'][e], 'iy': rec[sd][iy][e],
                               'z': rec[sd]['z'][e], 'ID': rec[sd]['detid'][e],
                               'energy': rec[sd]['energy'][e],
                               'temp': np.ones(n_rec),
                              })
        df_rec = df_rec.drop_duplicates('ID', keep='first')
        stats['total_rec_hits'] += len(df_rec)
        
        # build dataframe for sim hits
        n_sim = len(sim[sd]['detid'][e])
        df_sim = pd.DataFrame({'ix': sim[sd][ix][e], 'iy': sim[sd][iy][e],
                               'ID': sim[sd]['detid'][e],
                               'energy': sim[sd]['energy'][e],
                               'temp': np.ones(n_sim),
                              })
        
        # note that duplicate sim hits have different energies
        # to handle them, group by ID and sum energy:
        same_ID_energies = df_sim[['ID', 'energy']].groupby(['ID'])
        energy_sums = same_ID_energies.agg('sum')
        energy_map = {energy_sums.index.values[i]: energy_sums.energy.values[i]
                      for i in range(len(energy_sums))}
        
        df_sim = df_sim.drop_duplicates('ID', keep='first')
        df_sim = df_sim[df_sim.energy > sim_min]
        df_sim['energy'] = df_sim['ID'].map(energy_map)
        stats['total_sim_hits'] += len(df_sim)
        
        # dummy xyz coordinates to merge with rec hits
        df_sim['x'] = 1
        df_sim['y'] = 1
        df_sim['z'] = 1
        
        # form all possible sim-rec hit pairs 
        df = df_rec.reset_index().merge(df_sim.reset_index(), 
                                        on='temp', suffixes=('_rec', '_sim'))
        df['subdetector'] = sd
        df['subdetector_label'] = s
        
        # matching definition: within 2 ix/iy units from a simhit
        df['matched'] = ((abs(df['ix_rec'] - df['ix_sim']) < (eps+1)) &
                         (phi_neighbors(df['iy_rec'], df['iy_sim'], eps=eps)))
        
        # keep relevant columns, add to whole-event dataframe
        rec_cols = ['ID_rec', 'ix_rec', 'iy_rec',
                    'x_rec', 'y_rec', 'z_rec',
                    'energy_rec', 'subdetector_label', 'matched']
        sim_cols = ['ID_sim', 'ix_sim', 'iy_sim', 
                    'x_sim', 'y_sim', 'z_sim',
                    'energy_sim', 'subdetector_label', 'matched']

        #df_rec = df[rec_cols]
        #print(df_rec.groupby(['ID_rec'])['matched'].transform('sum'))
        #rec_matches = df_rec[['ID_rec', 'matched']].groupby('ID_rec')
        #rec_matches = rec_matches.agg('matched')
        
        #df_sim = df[['ix_sim', 'iy_sim', 'energy_sim', 'subdetector_label',
        #             'matched']]
        
        #df_list.append(df.drop_duplicates('ID_rec', keep='first'))
        
        # determine which hits were matched
        mask = (df['matched']==True)
        if (np.sum(mask)==0): continue
        df_rec = df[mask][rec_cols]
        df_rec = df_rec.drop_duplicates('ID_rec', keep='first')
        df_sim = df[mask][sim_cols]
        df_sim = df_sim.drop_duplicates('ID_sim', keep='first')

        stats['matched_sim_hits'] += len(df_sim)
        stats['matched_rec_hits'] += len(df_rec)
        
        df_sim_list.append(df_sim)
        df_rec_list.append(df_rec)

    stats['sim_match_fraction'] = zero_div(stats['matched_sim_hits'],
                                           stats['total_sim_hits'])
    stats['rec_match_fraction'] = zero_div(stats['matched_rec_hits'],
                                           stats['total_rec_hits'])

    if (len(df_sim_list)==0): sim = pd.DataFrame(columns=sim_cols)
    else: sim = pd.concat(df_sim_list)
    if (len(df_rec_list)==0): rec = pd.DataFrame(columns=rec_cols)
    else: rec = pd.concat(df_rec_list)

    sim.columns = sim.columns.str.rstrip('_sim')
    rec.columns = rec.columns.str.rstrip('_rec')
    event = {'sim': sim, 'rec': rec}
    
    return event, pd.DataFrame(stats, index=[0])


def select_hits(rec, args):

    # task 0: return truth matched rechits 
    if (args['task']==0):
        x = rec[['x', 'y', 'z', 'energy', 'subdetector_label']]
        x['energy_ecal'] = x['energy']
        x['energy_hcal'] = x['energy']
        x.loc[x.subdetector_label>2, 'energy_ecal'] = 0 
        x.loc[x.subdetector_label<3, 'energy_hcal'] = 0
        x = x[['x', 'y', 'z', 'energy_ecal', 'energy_hcal', 'subdetector_label']]
    return x.to_numpy()

def select_target(sim, args):
    if (args['task']==0):
        y = args['decay_mode']

    return y

def process_event(e, args={},
                  gen=ak.Array([]), reco=ak.Array([])):
    
    event, stats = truth_match_hits(e, gen['sim_hits'], 
                                    reco['rec_hits'], args=args) 

    # add additional statistics
    stats['tau_pt'] = gen['tau']['pt'][e]
    stats['vis_pt'] = ak.sum(gen['vis']['pt'][e])
    stats['vis_energy'] = ak.sum(gen['vis']['energy'][e])

    x = select_hits(event['rec'], args)
    y = select_target(event['sim'], args)
    #edge_index, edge_attr = build_edges(x)
    
    keep_cols = ['x', 'y', 'z', 'ix', 'iy', 'energy']
    data = Data(x=x, y=y, 
                sim=event['sim'][keep_cols].to_numpy())

    outdir = os.path.join(args['outdir'], f"task_{args['task']}")
    outdir = os.path.join(outdir, args['name'])
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
        logging.info(f"Created new directory: {outdir}")
    outfile = os.path.join(outdir, f"event{e}.pt")
    if args['save_output']: 
        print('saving', outfile)
        torch.save(data, outfile)

    logging.debug(f'Event {e} returned:\n {stats}')
    if not (e%10): logging.info(f'Processed event {e}') 
    return {'stats': stats}

def main(args):
    # parse the command line
    args = parse_args(args)

    # setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args['verbose'] else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')

    # determine graph construction task
    task_strs = {0: '(Task 0) Storing only rechits matched to simhits.',
                 1: '(Task 1) Storing only rechits matched to simhits.',
                 2: '(Task 2) Storing rechits matched to PF jets.',
                 3: '(Task 3) Storing rechits clustered by <strategy>.',
                 } 
    logging.info(task_strs[args['task']])
    
    # load skim file 
    infile = args['infile']
    infile_name = infile.split('/')[-1].split('.')[0]
    with open(infile, 'rb') as handle:
        data = pickle.load(handle)
        logging.info(f"Loaded {infile}")

    decay_modes = {'OneProngNoPi0': 0,
                   'OneProngOnePi0': 1,
                   'OneProngTwoPi0': 2,
                   'ThreeProngsNoPi0': 3,
                   'ThreeProngsOnePi0': 4}
    decay_mode = decay_modes[infile_name]
    args['decay_mode'] = decay_mode
    args['name'] = infile_name

    # filter out arrays of interest
    gen = filter_gen_arrays(data['gen_arrays'], args=args)
    reco = filter_reco_arrays(data['reco_arrays'], args=args)
    
    jets = reco['jets']
     
    # process taus with a worker pool
    nevts = len(gen['tau']['pt'])
    if args['n_events'] < nevts: nevts = args['n_events']
    evtids = np.arange(nevts)

    with mp.Pool(processes=args['n_workers']) as pool:
        process_func = partial(process_event, args=args,
                               gen=gen, reco=reco)
        output = pool.map(process_func, evtids)

    logging.info('All done!')

    # if save output, dump stat files
    stats = pd.concat([out['stats'] for out in output])
    outdir = f"stats/task_{args['task']}"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        logging.info(f"Created new directory: {outdir}")

    if args['statfile']=='':
        outfile = infile.split('.p')[0].split('/')[-1] + '.csv'
    else:
        outfile = args['statfile']
    
    outfile = os.path.join(outdir, outfile)
    logging.info(f'Saving summary stats to {outfile}')
    stats.to_csv(outfile, index=False)
    
if __name__ == '__main__':
    main(sys.argv[1:])
