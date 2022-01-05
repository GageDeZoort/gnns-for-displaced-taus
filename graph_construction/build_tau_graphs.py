# system
import os
import sys
import logging
import pickle
import argparse
import multiprocessing as mp
from functools import partial

# externals 
import awkward as ak
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader, Data
import particle 
from particle import PDGID
from particle import Particle 
from particle.pdgid import is_meson

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('-i', '--infile', type=str, default='../skims/data.p')
    add_arg('-o', '--outdir', type=str, default='../tau_graphs')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('-n', '--n-workers', type=int, default=1)
    add_arg('--n-events', type=int, default=10**5)
    return parser.parse_args()

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    """Computes pseudorapidity
       (https://en.wikipedia.org/wiki/Pseudorapidity)
    """
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def zero_div(a,b):
    if b==0: return 0
    return a/b

def filter_gen_arrays(data, pt_min=10, ecal_min=0.05, hcal_min=0.05):
    tau_mask = (abs(data['gen_ID'])==15)
    pt_filter = ak.flatten(data['gen_pt'][tau_mask] > pt_min)
    data = data[pt_filter]
    tau_mask = tau_mask[pt_filter]
    pdgID = data['gen_ID']
    
    # masks for visible and invisible systems
    lepton_mask = ((abs(pdgID)==11) | # ele 
                   (abs(pdgID)==13))  # mu
    hadron_mask = ((abs(pdgID)==111) | # pi0
                   (abs(pdgID)==211) | # pi+
                   (abs(pdgID)==311) | # K0
                   (abs(pdgID)==321) | # K+
                   (abs(pdgID)==130) | # K0S
                   (abs(pdgID)==310))  # K0L
    vis_mask = hadron_mask
    neutrino_mask = ((abs(pdgID)==12) | # nu_e
                     (abs(pdgID)==14) | # nu_mu
                     (abs(pdgID)==16))  # nu_tau

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
    inv = {'pt': data['gen_pt'][neutrino_mask],
           'eta': data['gen_eta'][neutrino_mask],
           'phi': data['gen_phi'][neutrino_mask],
           'mass': data['gen_mass'][neutrino_mask],
           'energy': data['gen_energy'][neutrino_mask],
           'vxy': data['gen_vxy'][neutrino_mask],
           'vz': data['gen_vz'][neutrino_mask]}
    
    # raw sim_hit quantities
    ee_mask = (data['sim_hits_ee_energy'] > ecal_min)
    eb_mask = (data['sim_hits_eb_energy'] > ecal_min)
    es_mask = (data['sim_hits_es_energy'] > ecal_min)
    #hcal_mask = (data['sim_hits_hcal_energy'] > hcal_min)
    sim_hits =  {'ee': {'detid': data['sim_hits_ee_detid'][ee_mask], 
                        'energy': data['sim_hits_ee_energy'][ee_mask],
                        'ix': data['sim_hits_ee_ix'][ee_mask],
                        'iy': data['sim_hits_ee_iy'][ee_mask],
                        'rho': data['sim_hits_ee_rho'][ee_mask],
                        'eta': data['sim_hits_ee_eta'][ee_mask],
                        'phi': data['sim_hits_ee_phi'][ee_mask],
                        'time': data['sim_hits_ee_time'][ee_mask]
                       },
                 'eb': {'detid': data['sim_hits_eb_detid'][eb_mask],
                        'energy': data['sim_hits_eb_energy'][eb_mask],
                        'ieta': data['sim_hits_eb_ieta'][eb_mask],
                        'iphi': data['sim_hits_eb_iphi'][eb_mask],
                        'rho': data['sim_hits_eb_rho'][eb_mask],
                        'eta': data['sim_hits_eb_eta'][eb_mask],
                        'phi': data['sim_hits_eb_phi'][eb_mask],
                        'time': data['sim_hits_eb_time'][eb_mask]
                       },
                 'es': {'detid': data['sim_hits_es_detid'][es_mask],
                        'energy': data['sim_hits_es_energy'][es_mask],
                        'six': data['sim_hits_es_six'][es_mask],
                        'siy': data['sim_hits_es_siy'][es_mask],
                        'rho': data['sim_hits_es_rho'][es_mask],
                        'eta': data['sim_hits_es_eta'][es_mask],
                        'phi': data['sim_hits_es_phi'][es_mask],
                        'time': data['sim_hits_es_time'][es_mask]
                       }
                }
    
    # derived sim_hit quantities
    for sd in sim_hits.keys():
        sim_hits[sd]['x'] = sim_hits[sd]['rho'] * np.cos(sim_hits[sd]['phi'])
        sim_hits[sd]['y'] = sim_hits[sd]['rho'] * np.sin(sim_hits[sd]['phi'])
        sim_hits[sd]['z'] = (sim_hits[sd]['rho'] / 
                             np.tan(2*np.arctan(np.exp(-1*sim_hits[sd]['eta']))))
   
    return {'tau': tau, 'vis': vis, 'inv': inv, 'sim_hits': sim_hits}

def filter_reco_arrays(data, ecal_min=0.05, hcal_min = 0.05):
    ee_mask = (data['rec_hits_ee_energy'] > ecal_min)
    eb_mask = (data['rec_hits_eb_energy'] > ecal_min)
    es_mask = (data['rec_hits_es_energy'] > ecal_min)
    hbhe_mask = (data['rec_hits_hbhe_energy'] > hcal_min)
    hf_mask = (data['rec_hits_hf_energy'] > hcal_min)
    ho_mask = (data['rec_hits_ho_energy'] > hcal_min)
    rec_hits = {'ee': {'detid': data['rec_hits_ee_detid'][ee_mask], 
                       'energy': data['rec_hits_ee_energy'][ee_mask],
                       'ix': data['rec_hits_ee_ix'][ee_mask],
                       'iy': data['rec_hits_ee_iy'][ee_mask],
                       'rho': data['rec_hits_ee_rho'][ee_mask],
                       'eta': data['rec_hits_ee_eta'][ee_mask],
                       'phi': data['rec_hits_ee_phi'][ee_mask],
                       'time': data['rec_hits_ee_time'][ee_mask]
                      },
                'eb': {'detid': data['rec_hits_eb_detid'][eb_mask],
                       'energy': data['rec_hits_eb_energy'][eb_mask],
                       'ieta': data['rec_hits_eb_ieta'][eb_mask],
                       'iphi': data['rec_hits_eb_iphi'][eb_mask],
                       'rho': data['rec_hits_eb_rho'][eb_mask],
                       'eta': data['rec_hits_eb_eta'][eb_mask],
                       'phi': data['rec_hits_eb_phi'][eb_mask],
                       'time': data['rec_hits_eb_time'][eb_mask]
                      },
                'es': {'detid': data['rec_hits_es_detid'][es_mask],
                       'energy': data['rec_hits_es_energy'][es_mask],
                       'six': data['rec_hits_es_six'][es_mask],
                       'siy': data['rec_hits_es_siy'][es_mask],
                       'rho': data['rec_hits_es_rho'][es_mask],
                       'eta': data['rec_hits_es_eta'][es_mask],
                       'phi': data['rec_hits_es_phi'][es_mask],
                       'time': data['rec_hits_es_time'][es_mask]
                      },
                'hbhe': {'detid': data['rec_hits_hbhe_detid'][hbhe_mask],
                         'energy': data['rec_hits_hbhe_energy'][hbhe_mask],
                         'ieta': data['rec_hits_hbhe_ieta'][hbhe_mask],
                         'iphi': data['rec_hits_hbhe_iphi'][hbhe_mask],
                         'rho': data['rec_hits_hbhe_rho'][hbhe_mask],
                         'eta': data['rec_hits_hbhe_eta'][hbhe_mask],
                         'phi': data['rec_hits_hbhe_phi'][hbhe_mask],
                         'time': data['rec_hits_hbhe_time'][hbhe_mask]

                        },
                'hf': {'detid': data['rec_hits_hf_detid'][hf_mask],
                       'energy': data['rec_hits_hf_energy'][hf_mask],
                       'ieta': data['rec_hits_hf_ieta'][hf_mask],
                       'iphi': data['rec_hits_hf_iphi'][hf_mask],
                       'rho': data['rec_hits_hf_rho'][hf_mask],
                       'eta': data['rec_hits_hf_eta'][hf_mask],
                       'phi': data['rec_hits_hf_phi'][hf_mask],
                       'time': data['rec_hits_hf_time'][hf_mask]

                      },
                'ho': {'detid': data['rec_hits_ho_detid'][ho_mask],
                       'energy': data['rec_hits_ho_energy'][ho_mask],
                       'ieta': data['rec_hits_ho_ieta'][ho_mask],
                       'iphi': data['rec_hits_ho_iphi'][ho_mask],
                       'rho': data['rec_hits_ho_rho'][ho_mask],
                       'eta': data['rec_hits_ho_eta'][ho_mask],
                       'phi': data['rec_hits_ho_phi'][ho_mask],
                       'time': data['rec_hits_ho_time'][ho_mask]

                      },
           }
    
    # derived rec_hit quantities
    for sd in rec_hits.keys():
        rec_hits[sd]['x'] = rec_hits[sd]['rho'] * np.cos(rec_hits[sd]['phi'])
        rec_hits[sd]['y'] = rec_hits[sd]['rho'] * np.sin(rec_hits[sd]['phi'])
        rec_hits[sd]['z'] = (rec_hits[sd]['rho'] / 
                             np.tan(2*np.arctan(np.exp(-1*rec_hits[sd]['eta']))))
        
    return {'rec_hits': rec_hits}

def debug_event(i, sim, rec, vis, zoom_rec=False):
    print('simhits in ee:', len(sim['ee']['detid'][i]))
    print('simhits in eb:', len(sim['eb']['detid'][i]))
    print('simhits in es:', len(sim['es']['detid'][i]))
    print('rechits in ee:', len(rec['ee']['detid'][i]))
    print('rechits in eb:', len(rec['eb']['detid'][i]))
    print('rechits in es:', len(rec['es']['detid'][i]))

    sim_detid = sim['eb']['detid'][i]
    sim_ieta = sim['eb']['ieta'][i]
    sim_iphi = sim['eb']['iphi'][i]
    sim_energy = sim['eb']['energy'][i]
    rec_detid = rec['eb']['detid'][i]
    rec_ieta = rec['eb']['ieta'][i]
    rec_iphi = rec['eb']['iphi'][i]
    rec_energy = rec['eb']['energy'][i]
    vis_ID = vis['ID'][i]
    print(vis_ID)
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

def truth_match_hits(e, sim, rec):
    stats = {'total_sim_hits':  0,
             'matched_sim_hits': 0,
             'total_rec_hits': 0,
             'matched_rec_hits': 0}
    
    # loop over subdetectors, build dataframe in each
    df_list = []
    for sd in sim.keys(): 
        
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
        df_sim = pd.DataFrame({'x': sim[sd]['x'][e], 'ix': sim[sd][ix][e],
                               'y': sim[sd]['y'][e], 'iy': sim[sd][iy][e],
                               'z': sim[sd]['z'][e], 'ID': sim[sd]['detid'][e],
                               'energy': sim[sd]['energy'][e],
                               'temp': np.ones(n_sim),
                              })
        df_sim = df_sim.drop_duplicates('ID', keep='first')
        stats['total_sim_hits'] += len(df_sim)
        
        # form all possible sim-rec hit pairs 
        df = df_rec.reset_index().merge(df_sim.reset_index(), 
                                        on='temp', suffixes=('_rec', '_sim'))
        df['subdetector'] = sd
        
        # matching definition: within 2 ix/iy units from a simhit
        df['matched'] = ((abs(df['ix_rec'] - df['ix_sim']) < 3) &
                         (abs(df['iy_rec'] - df['iy_sim']) < 3))
        
        # keep relevant columns, add to whole-event dataframe
        keep_cols = ['ix_rec', 'iy_rec', 'x_rec', 'y_rec', 'z_rec',
                     'energy_rec', 'subdetector', 'matched']
        df_list.append(df.drop_duplicates('ID_rec', keep='first')[keep_cols])
        
        # determine which hits were matched
        mask = (df['matched']==True)
        if (np.sum(mask)==0): continue
        stats['matched_sim_hits'] += len(np.unique(df[mask].ID_sim))
        stats['matched_rec_hits'] += len(np.unique(df[mask].ID_rec))
    
    stats['sim_match_fraction'] = zero_div(stats['matched_sim_hits'],
                                           stats['total_sim_hits'])
    stats['rec_match_fraction'] = zero_div(stats['matched_rec_hits'],
                                           stats['total_rec_hits'])

    event = pd.concat(df_list)
    
    return event, pd.DataFrame(stats, index=[0])

def process_event(e, args={}, 
                  gen=ak.Array([]), reco=ak.Array([])):
    
    
    event, stats = truth_match_hits(e, gen['sim_hits'], reco['rec_hits']) 
    logging.debug(f'Event {e} returned:\n {stats}')
    if not (e%10): logging.info(f'Processed event {e}') 
    return stats


def main():
    # parse the command line
    args = parse_args()

    # setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')
    
    # load skim file 
    with open(args.infile, 'rb') as handle:
        data = pickle.load(handle)
        logging.info(f'Loaded {args.infile}')

    # filter out arrays of interest
    gen = filter_gen_arrays(data['gen_arrays'])
    reco = filter_reco_arrays(data['reco_arrays'])
     
    # process taus with a worker pool
    nevts = len(gen['tau']['pt'])
    if args.n_events < nevts: nevts = args.n_events
    evtids = np.arange(nevts)
    print(evtids)
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, args=args,
                               gen=gen, reco=reco)
        output = pool.map(process_func, evtids)

    # analyze output statistics
    # total_sim_hits  matched_sim_hits sim_match_fraction  
    logging.info('All done!')
    stats = pd.concat(output)
    outfile = args.infile.split('.p')[0].split('/')[-1] + '.csv'
    outfile = 'stats/'+outfile
    logging.info(f'Saving summary stats to {outfile}')
    stats.to_csv(outfile, index=False)
    
if __name__ == '__main__':
    main()
