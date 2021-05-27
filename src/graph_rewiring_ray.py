import argparse
import os
import time
from functools import partial
import json
import numpy as np
import torch
import torch.nn.functional as F
from GNN_early import GNNEarly
from GNN import GNN
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.ax import AxSearch
from ray.tune import Analysis
from utils import get_sem, mean_confidence_interval
from run_GNN import get_optimizer, test, test_OGB, train
from torch import nn
from data import get_dataset, set_train_val_test_split
from graph_rewiring import get_two_hop, apply_gdc, GDC, dirichlet_energy, make_symmetric, KNN, apply_beltrami
from graph_rewiring_eval import train_GRAND

"""
python3 ray_tune.py --dataset ogbn-arxiv --lr 0.005 --add_source --function transformer --attention_dim 16 --hidden_dim 128 --heads 4 --input_dropout 0 --decay 0 --adjoint --adjoint_method rk4 --method rk4 --time 5.08 --epoch 500 --num_samples 1 --name ogbn-arxiv-test --gpus 1 --grace_period 50 

"""



def set_rewiring_space(opt):
    # DIGL args
    # opt['rewiring'] = None #'gdc'  # tune.choice(['gdc', None])
    opt['rewiring'] = 'pos_enc_knn'
    if opt['rewiring']:
      opt['attention_rewiring'] = False #tune.choice([True, False])
      opt['reweight_attention'] = False #tune.sample_from(lambda spec: tune.choice([True, False])
                                  # if spec.config.rewiring else False) #tune.choice([True, False])
      opt['gdc_sparsification'] = tune.choice(['topk','threshold'])

      opt['pos_dist_quantile'] = tune.sample_from(lambda spec: tune.choice([0.001,0.004,0.008])
                        if spec.config.gdc_sparsification == 'threshold' else None )

      ks = [8, 16, 32, 64]
      # opt['gdc_k'] = tune.choice(ks)
      opt['gdc_k'] = tune.sample_from(lambda spec: tune.choice(ks)
                        if spec.config.gdc_sparsification == 'topk' else None )

      opt['exact'] = True
      opt['gdc_threshold'] = 0.01
      opt['ppr_alpha'] = 0.05 # tune.uniform(0.01, 0.2)
      opt['rewire_KNN_sym'] = False# tune.choice([True, False])
      opt['pos_enc_orientation'] = None #tune.choice(["row", "col"])

    # experiment args
    opt['block'] = 'attention'
    opt['function'] = 'laplacian'
    # opt['use_lcc'] = True <- this is actually opt['not_lcc'] = False but is default for all except arxiv

    opt['beltrami'] = True  # tune.choice([True, False])
    # bel_choice = tune.choice(["exp_kernel", "cosine_sim", "pearson", "scaled_dot"])  # "scaled_dot"
    # non_bel_choice = tune.choice(["cosine_sim", "pearson", "scaled_dot"])  # "scaled_dot"
    # opt['attention_type'] = tune.sample_from(lambda spec: bel_choice if spec.config.beltrami else non_bel_choice)
    opt['attention_type'] = tune.choice(["cosine_sim", "scaled_dot"])

    # edge_sampling_space is in:
    # ['pos_distance','z_distance']) if ['attention_type'] == exp_kernel_z or exp_kernel_pos as have removed queries / keys
    # ['pos_distance_QK','z_distance_QK']) for exp_kernel
    # ['z_distance_QK']) for any other attention type, plus requires symmetric_attention as don't learn the pos QKp(p) just QK(z)

    # opt['symmetric_attention'] = True #symmetric attention required for distance in QK space
    # exp_kernel_choice = tune.choice(['pos_distance_QK','z_distance_QK'])
    # non_exp_kernel_choice = 'z_distance_QK'
    # opt['edge_sampling_space'] = tune.sample_from(lambda spec: exp_kernel_choice if spec.config.beltrami else non_exp_kernel_choice)

    opt['feat_hidden_dim'] = tune.choice([32, 64, 128])#, 128])
    opt['pos_enc_hidden_dim'] = tune.choice([16, 32])#, 64])
    opt['hidden_dim'] = tune.sample_from(lambda spec: spec.config.feat_hidden_dim + spec.config.pos_enc_hidden_dim
                        if spec.config.beltrami else tune.choice([32, 64, 128]))

    opt['pos_enc_type'] = tune.choice(['DW256', 'DW128', 'DW64']) #'GDC' #tune.choice(['HYP02', 'HYP04', 'HYP08', 'HYP16'])
    opt['pos_enc_orientation'] = 'row' #tune.choice(["row", "col"])
    opt['square_plus'] = True #tune.choice([True, False])

    # opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(6, 8))  # hidden dim of X in dX/dt

    opt['rewire_KNN'] = False
    if opt['rewire_KNN']:
        opt['rewire_KNN_T'] = tune.choice(["T0","TN"])
        opt['rewire_KNN_epoch'] = tune.choice([2,10,20,50])
        opt['rewire_KNN_k'] = tune.choice([16, 32, 64, 128, 256])
        opt['rewire_KNN_sym'] = tune.choice([True, False])
        opt['edge_sampling_T'] = None
        opt['edge_sampling_epoch'] = None
        opt['edge_sampling_add'] = None
        opt['edge_sampling_rmv'] = None
        opt['edge_sampling_sym'] = None
        opt['edge_sampling_space'] = None

    opt['edge_sampling'] = False
    if opt['edge_sampling']:
        opt['rewire_KNN_T'] = None
        opt['rewire_KNN_epoch'] = None
        opt['rewire_KNN_k'] = None
        opt['rewire_KNN_sym'] = None
        opt['edge_sampling_T'] = 'TN' #tune.choice(["T0","TN"])
        opt['edge_sampling_epoch'] = 10 #tune.choice([2,10,20,50])
        opt['edge_sampling_add'] = 0.08 #tune.choice([0.04, 0.08, 0.16, 0.32])
        opt['edge_sampling_rmv'] = 0.08 #tune.choice([0.04, 0.08, 0.16, 0.32])
        opt['edge_sampling_sym'] = False #tune.choice([True, False])
        opt['edge_sampling_space'] = 'pos_distance' #tune.choice(['pos_distance','z_distance'])

    opt['edge_sampling_online'] = False
    if opt['edge_sampling_online']:
      opt['edge_sampling_add_type'] = tune.choice(['importance','random'])
      opt['edge_sampling_space'] =  tune.choice(['attention','pos_distance','z_distance'])#,'pos_distance_QK','z_distance_QK'])
      opt['symmetric_attention'] = tune.sample_from(lambda spec: True if spec.config.edge_sampling_space
                                              in ['pos_distance_QK','z_distance_QK'] else False)
      opt['edge_sampling_online_reps'] = None#tune.choice([2,3,4])
      opt['edge_sampling_sym'] = None#tune.choice([True, False])
      opt['edge_sampling_add'] = None#tune.choice([0.04, 0.08, 0.16, 0.32, 0.64]) # tune.choice([0.04, 0.08, 0.16, 0.32])
      opt['edge_sampling_rmv'] = None#tune.choice([0.0, 0.04, 0.08])  # tune.choice([0.04, 0.08, 0.16, 0.32])
      opt['edge_sampling_rmv'] = None  # tune.choice([0.04, 0.08, 0.16, 0.32])

    opt['fa_layer'] = False
    if opt['edge_sampling']:
      opt['fa_layer_time']  = tune.choice([1, 2, 3])# 1.0
      opt['fa_layer_method'] = 'rk4'
      opt['fa_layer_step_size']  = tune.choice([0.5, 1])# 1.0
      opt['fa_layer_edge_sampling_rmv'] = tune.choice([0, 0.25 ,0.5, 0.75])
      # opt["time"] = tune.uniform(0.25, 5.0)  # tune.uniform(2.0, 30.0)  # terminal time of the ODE integrator;

    return opt


def set_cora_search_space(opt):
  opt["decay"] = tune.loguniform(0.01, 0.2)  # weight decay l2 reg
  if opt['regularise']:
    opt["kinetic_energy"] = tune.loguniform(0.001, 10.0)
    opt["directional_penalty"] = tune.loguniform(0.001, 10.0)

  opt["lr"] = tune.uniform(0.01, 0.2)
  opt["input_dropout"] = tune.uniform(0.2, 0.8)  # encoder dropout
  # opt["input_dropout"] = 0.5
  opt["optimizer"] = tune.choice(["adam", "adamax"])
  opt["dropout"] = tune.uniform(0, 0.2)  # output dropout
  opt["time"] = tune.uniform(10.0, 30.0)  # tune.uniform(2.0, 30.0)  # terminal time of the ODE integrator;

  if opt["block"] in {'attention', 'mixed', 'hard_attention'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))  #
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  # hidden dim for attention
    opt['attention_norm_idx'] = tune.choice([0, 1])
    # opt['attention_norm_idx'] = 0
    # opt["leaky_relu_slope"] = tune.uniform(0, 0.7)
    opt["self_loop_weight"] = tune.choice([0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  # if opt['self_loop_weight'] > 0.0:
  #     opt['exact'] = True  # for GDC, need exact if selp loop weight >0
  opt['exact'] = tune.sample_from(lambda spec: True if spec.config.self_loop_weight > 0.0 else False)

  opt["tol_scale"] = tune.loguniform(1, 1000)  # num you multiply the default rtol and atol by
  if opt["adjoint"]:
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun"])  # , "rk4"])
    opt["tol_scale_adjoint"] = tune.loguniform(100, 10000)

  opt['add_source'] = tune.choice([True, False])
  # opt['att_samp_pct'] = tune.uniform(0.3, 1)
  opt['batch_norm'] = tune.choice([True, False])
  opt['use_mlp'] = False
  # opt['use_mlp'] = tune.choice([True, False])

  return opt


def set_citeseer_search_space(opt):
  # opt["decay"] = 0.1
  opt['decay'] = tune.loguniform(2e-3, 1e-1)
  if opt['regularise']:
    opt["kinetic_energy"] = tune.loguniform(0.001, 10.0)
    opt["directional_penalty"] = tune.loguniform(0.001, 10.0)

  opt["lr"] = tune.uniform(1e-3, 1e-2)
  opt["input_dropout"] = tune.uniform(0.4, 0.7)
  opt["dropout"] = tune.uniform(0.1, 0.7)
  opt["time"] = tune.uniform(0.5, 12.0)
  opt["optimizer"] = tune.choice(["rmsprop", "adam", "adamax"])
  #

  if opt["block"] in {'attention', 'mixed', 'hard_attention'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(1, 4))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 8))
    opt['attention_norm_idx'] = 1  # tune.choice([0, 1])
    # opt["leaky_relu_slope"] = tune.uniform(0, 0.7)
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)  # 1 seems to work pretty well

  opt["tol_scale"] = tune.loguniform(1, 1e4)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun"])  # , "rk4"])

    opt['add_source'] = tune.choice([True, False])
    # opt['att_samp_pct'] = tune.uniform(0.3, 1)
    opt['batch_norm'] = tune.choice([True, False])
    # opt['use_mlp'] = tune.choice([True, False])
  return opt


def set_pubmed_search_space(opt):
  opt['adjoint'] = True
  opt["decay"] = tune.loguniform(1e-4, 1e-2)
  if opt['regularise']:
    opt["kinetic_energy"] = tune.loguniform(0.01, 1.0)
    opt["directional_penalty"] = tune.loguniform(0.01, 1.0)

  opt["hidden_dim"] = 128  # tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))
  opt["lr"] = tune.uniform(0.01, 0.05)
  opt["input_dropout"] = 0.5  # tune.uniform(0.2, 0.5)
  opt["dropout"] = tune.uniform(0, 0.6)
  opt["time"] = tune.uniform(5.0, 30.0)
  opt["optimizer"] = tune.choice(["rmsprop", "adam", "adamax"])

  if opt["block"] in {'attention', 'mixed', 'hard_attention'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 3))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 7))
    opt['attention_norm_idx'] = tune.choice([0, 1])
    # opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
    # opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
    #   [0, 1])  # whether or not to use self-loops
    opt["self_loop_weight"] = 1
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1, 1e5)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(10, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])
  else:
    raise Exception("Can't train on PubMed without the adjoint method.")

  opt['add_source'] = tune.choice([True, False])
  # opt['att_samp_pct'] = tune.uniform(0.3, 1)
  opt['batch_norm'] = tune.choice([True, False])
  # opt['batch_norm'] = True
  opt['use_mlp'] = False

  return opt


def set_photo_search_space(opt):
  opt['adjoint'] = True
  opt["decay"] = tune.loguniform(0.0001, 1e-2)
  if opt['regularise']:
    opt["kinetic_energy"] = tune.loguniform(0.01, 5.0)
    opt["directional_penalty"] = tune.loguniform(0.001, 10.0)

  opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 7))
  opt["lr"] = tune.loguniform(5e-4, 0.1)
  opt["input_dropout"] = tune.uniform(0.4, 0.8)
  opt["dropout"] = tune.uniform(0, 0.5)
  opt["time"] = tune.uniform(0.5, 12.0)
  # opt["optimizer"] = tune.choice(["adam", "adamax", "rmsprop"])
  opt["optimizer"] = "adam"

  if opt["block"] in {'attention', 'mixed', 'hard_attention'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 3))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 7))
    opt['attention_norm_idx'] = tune.choice([0, 1])
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(100, 1e6)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(100, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])

  if opt['rewiring'] == 'gdc':
    # opt['gdc_sparsification'] = tune.choice(['topk', 'threshold'])
    opt['gdc_sparsification'] = 'threshold'
    opt['exact'] = False
    # opt['gdc_method'] = tune.choice(['ppr', 'heat'])
    opt['gdc_method'] = 'ppr'
    # opt['avg_degree'] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  #  bug currently in pyg
    opt['gdc_threshold'] = tune.loguniform(0.0001, 0.0005)
    # opt['gdc_threshold'] = None
    opt['ppr_alpha'] = tune.uniform(0.1, 0.25)
    # opt['heat_time'] = tune.uniform(1, 5)

  opt['add_source'] = tune.choice([True, False])
  opt['add_source'] = False
  # opt['att_samp_pct'] = tune.uniform(0.3, 1)
  # opt['batch_norm'] = tune.choice([True, False])
  opt['batch_norm'] = True
  opt['use_mlp'] = tune.choice([True, False])

  return opt


def set_computers_search_space(opt):
  opt['adjoint'] = True
  opt["decay"] = tune.loguniform(2e-3, 1e-2)
  if opt['regularise']:
    opt["kinetic_energy"] = tune.loguniform(0.01, 10.0)
    opt["directional_penalty"] = tune.loguniform(0.001, 10.0)

  opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))
  opt["lr"] = tune.loguniform(5e-5, 5e-3)
  opt["input_dropout"] = tune.uniform(0.4, 0.8)
  opt["dropout"] = tune.uniform(0, 0.8)
  opt["self_loop_weight"] = tune.choice([0, 1])
  opt["time"] = tune.uniform(0.5, 10.0)
  opt["optimizer"] = tune.choice(["adam", "adamax", "rmsprop"])

  if opt["block"] in {'attention', 'mixed', 'hard_attention'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 8))
    opt['attention_norm_idx'] = 1  # tune.choice([0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1e1, 1e4)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])

  if opt['rewiring'] == 'gdc':
    # opt['gdc_sparsification'] = tune.choice(['topk', 'threshold'])
    opt['gdc_sparsification'] = 'threshold'
    opt['exact'] = False
    # opt['gdc_method'] = tune.choice(['ppr', 'heat'])
    opt['gdc_method'] = 'ppr'
    # opt['avg_degree'] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  #  bug currently in pyg
    opt['gdc_threshold'] = tune.loguniform(0.00001, 0.01)
    # opt['gdc_threshold'] = None
    opt['ppr_alpha'] = tune.uniform(0.01, 0.2)
    # opt['heat_time'] = tune.uniform(1, 5)
  return opt


def set_coauthors_search_space(opt):
  opt['adjoint'] = True
  opt["decay"] = tune.loguniform(1e-3, 2e-2)
  if opt['regularise']:
    opt["kinetic_energy"] = tune.loguniform(0.01, 10.0)
    opt["directional_penalty"] = tune.loguniform(0.01, 10.0)

  opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 6))
  opt["lr"] = tune.loguniform(1e-5, 0.1)
  opt["input_dropout"] = tune.uniform(0.4, 0.8)
  opt["dropout"] = tune.uniform(0, 0.8)
  opt["self_loop_weight"] = tune.choice([0, 1])
  opt["time"] = tune.uniform(0.5, 10.0)
  opt["optimizer"] = tune.choice(["adam", "adamax", "rmsprop"])

  if opt["block"] in {'attention', 'mixed', 'hard_attention'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 4))
    opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 8))
    opt['attention_norm_idx'] = tune.choice([0, 1])
    opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
    opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
      [0, 1])  # whether or not to use self-loops
  else:
    opt["self_loop_weight"] = tune.uniform(0, 3)

  opt["tol_scale"] = tune.loguniform(1e1, 1e4)

  if opt["adjoint"]:
    opt["tol_scale_adjoint"] = tune.loguniform(1, 1e5)
    opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])

  if opt['rewiring'] == 'gdc':
    # opt['gdc_sparsification'] = tune.choice(['topk', 'threshold'])
    opt['gdc_sparsification'] = 'threshold'
    opt['exact'] = False
    # opt['gdc_method'] = tune.choice(['ppr', 'heat'])
    opt['gdc_method'] = 'ppr'
    # opt['avg_degree'] = tune.sample_from(lambda _: 2 ** np.random.randint(4, 8))  #  bug currently in pyg
    opt['gdc_threshold'] = tune.loguniform(0.0001, 0.0005)
    # opt['gdc_threshold'] = None
    opt['ppr_alpha'] = tune.uniform(0.1, 0.25)
    # opt['heat_time'] = tune.uniform(1, 5)

  return opt


def set_arxiv_search_space(opt):
  # opt["decay"] = tune.loguniform(1e-10, 1e-6)
  opt["decay"] = 0
  # opt["decay"] = 0
  if opt['regularise']:
    opt["kinetic_energy"] = tune.loguniform(0.01, 10.0)
    opt["directional_penalty"] = tune.loguniform(0.001, 10.0)

  # opt["hidden_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(5, 9))
  opt["hidden_dim"] = 128  # best choice with attention
  # opt["hidden_dim"] = 256  # best choice without attention
  opt["lr"] = tune.uniform(0.001, 0.05)
  # opt['lr'] = 0.02
  opt["input_dropout"] = tune.uniform(0., 0.6)
  # opt["input_dropout"] = 0
  opt["dropout"] = tune.uniform(0, 0.6)
  # opt["dropout"] = 0
  opt['step_size'] = tune.choice([0.25, 0.5, 1])
  # opt['step_size'] = 1 #0.5
  opt['adjoint_step_size'] = tune.choice([0.25, 0.5, 1])
  # opt['adjoint_step_size'] = 1 #0.5
  opt["time"] = tune.uniform(5.0, 20.0)
  opt["optimizer"] = tune.choice(["adam", "adamax", "rmsprop"])
  # opt['optimizer'] = 'adam'
  if opt["block"] in {'attention', 'mixed', 'hard_attention'} or opt['function'] in {'GAT', 'transformer', 'dorsey'}:
    # opt["heads"] = tune.sample_from(lambda _: 2 ** np.random.randint(0, 3))
    opt["heads"] = 4
    # opt["attention_dim"] = tune.sample_from(lambda _: 2 ** np.random.randint(3, 7))
    opt["attention_dim"] = 16 #32
    # opt['attention_norm_idx'] = tune.choice([0, 1])
    # opt["self_loop_weight"] = tune.choice([0, 0.5, 1, 2]) if opt['block'] == 'mixed' else tune.choice(
    #   [0, 1])
    # opt["self_loop_weight"] = 0.0 #1
    # opt["leaky_relu_slope"] = tune.uniform(0, 0.8)
    opt["leaky_relu_slope"] = 0.2
  else:
    # opt["self_loop_weight"] = tune.uniform(0, 3)
    opt["self_loop_weight"] = 0.0 #tune.choice([0, 1])
  # opt['data_norm'] = tune.choice(['rw', 'gcn'])
  # opt['add_source'] = tune.choice([True, False])
  opt['add_source'] = True
  # opt['att_samp_pct'] = 1 #tune.uniform(0.6, 1)
  opt['batch_norm'] = tune.choice([True, False])
  # opt['batch_norm'] = False #True

  opt['use_labels'] = True
  opt['label_rate'] = 0.5
  # opt['label_rate'] = tune.uniform(0.05, 0.5)

  # opt["method"] = tune.choice(["dopri5", "rk4"])
  # opt["method"] = tune.choice(["midpoint", "rk4"])
  # opt["tol_scale"] = tune.loguniform(10, 1e4)
  opt["method"] = "rk4"

  if opt["adjoint"]:
    # opt["tol_scale_adjoint"] = tune.loguniform(10, 1e5)
    # opt["adjoint_method"] = tune.choice(["dopri5", "adaptive_heun", "rk4"])
    # opt["adjoint_method"] = tune.choice(["adaptive_heun", "rk4"])
    opt["adjoint_method"] = "rk4"

  return opt


def set_search_space(opt):
    opt = set_rewiring_space(opt)
    if opt["dataset"] == "Cora":
        return set_cora_search_space(opt)
    elif opt["dataset"] == "Pubmed":
        return set_pubmed_search_space(opt)
    elif opt["dataset"] == "Citeseer":
        return set_citeseer_search_space(opt)
    elif opt["dataset"] == "Computers":
        return set_computers_search_space(opt)
    elif opt["dataset"] == "Photo":
        return set_photo_search_space(opt)
    elif opt["dataset"] == "CoauthorCS":
        return set_coauthors_search_space(opt)
    elif opt["dataset"] == "ogbn-arxiv":
        return set_arxiv_search_space(opt)


def main(opt):
    data_dir = os.path.abspath("../data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = set_search_space(opt)
    scheduler = ASHAScheduler(
        metric=opt['metric'],
        mode="max",
        max_t=opt["epoch"],
        grace_period=opt["grace_period"],
        reduction_factor=opt["reduction_factor"],
    )
    reporter = CLIReporter(
        metric_columns=["accuracy", "test_acc", "train_acc", "loss", "training_iteration", "forward_nfe",
                        "backward_nfe"]
    )
    # choose a search algorithm from https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    search_alg = AxSearch(metric=opt['metric'])
    search_alg = None

    train_fn = train_ray_rand

    result = tune.run(
        partial(train_fn, data_dir=data_dir),
        name=opt["name"],
        resources_per_trial={"cpu": opt["cpus"], "gpu": opt["gpus"]},
        search_alg=search_alg,
        keep_checkpoints_num=3,
        checkpoint_score_attr=opt['metric'],
        config=opt,
        num_samples=opt["num_samples"],
        scheduler=scheduler,
        max_failures=2,
        local_dir="../ray_tune",
        progress_reporter=reporter,
        raise_on_failed_trial=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_cora_defaults",
        action="store_true",
        help="Whether to run with best params for cora. Overrides the choice of dataset",
    )
    parser.add_argument(
        "--dataset", type=str, default="Cora", help="Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS"
    )
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension.")
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument("--input_dropout", type=float, default=0.5, help="Input dropout rate.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=5e-4, help="Weight decay for optimization")
    parser.add_argument("--self_loop_weight", type=float, default=1.0, help="Weight of self-loops.")
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument("--epoch", type=int, default=10, help="Number of training epochs per iteration.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Factor in front matrix A.")
    parser.add_argument("--time", type=float, default=1.0, help="End time of ODE function.")
    parser.add_argument("--augment", action="store_true",
                        help="double the length of the feature vector by appending zeros to stabilise ODE learning", )
    parser.add_argument("--alpha_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) alpha")
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument("--beta_dim", type=str, default="sc", help="choose either scalar (sc) or vector (vc) beta")
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')

    # ODE args
    parser.add_argument(
        "--method", type=str, default="dopri5", help="set the numerical solver: dopri5, euler, rk4, midpoint"
    )
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument(
        "--adjoint_method", type=str, default="adaptive_heun",
        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint"
    )
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument("--adjoint", dest='adjoint', action='store_true',
                        help="use the adjoint ODE method to reduce memory footprint")
    parser.add_argument("--tol_scale", type=float, default=1.0, help="multiplier for atol and rtol")
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--ode_blocks", type=int, default=1, help="number of ode blocks to run")
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')
    # SDE args
    parser.add_argument("--dt_min", type=float, default=1e-5, help="minimum timestep for the SDE solver")
    parser.add_argument("--dt", type=float, default=1e-3, help="fixed step size")
    parser.add_argument('--adaptive', dest='adaptive', action='store_true', help='use adaptive step sizes')
    # Attention args
    parser.add_argument(
        "--leaky_relu_slope",
        type=float,
        default=0.2,
        help="slope of the negative part of the leaky relu used in attention",
    )
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='the size to project x to before calculating att scores')
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--attention_norm_idx", type=int, default=0, help="0 = normalise rows, 1 = normalise cols")
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, SDE')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT, SDE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    # ray args
    parser.add_argument("--num_samples", type=int, default=20, help="number of ray trials")
    parser.add_argument("--gpus", type=float, default=0, help="number of gpus per trial. Can be fractional")
    parser.add_argument("--cpus", type=float, default=1, help="number of cpus per trial. Can be fractional")
    parser.add_argument(
        "--grace_period", type=int, default=5, help="number of epochs to wait before terminating trials"
    )
    parser.add_argument(
        "--reduction_factor", type=int, default=4, help="number of trials is halved after this many epochs"
    )
    parser.add_argument("--name", type=str, default="ray_exp")
    parser.add_argument("--num_splits", type=int, default=0, help="Number of random splits >= 0. 0 for planetoid split")
    parser.add_argument("--num_init", type=int, default=1, help="Number of random initializations >= 0")

    parser.add_argument("--max_nfe", type=int, default=300, help="Maximum number of function evaluations allowed.")
    parser.add_argument('--metric', type=str, default='accuracy',
                        help='metric to sort the hyperparameter tuning runs on')
    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    parser.add_argument("--baseline", action="store_true", help="Wheather to run the ICML baseline or not.")
    parser.add_argument("--regularise", dest='regularise', action='store_true', help='search over reg params')

    # rewiring args
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    parser.add_argument('--gdc_sparsification', type=str, default='topk', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                        help="above this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                        help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                        help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                        help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
    parser.add_argument('--att_samp_pct', type=float, default=1,
                        help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="addD_rvR", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
    parser.add_argument('--attention_rewiring', action='store_true',
                        help='perform DIGL using precalcualted GRAND attention')

    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
    parser.add_argument('--rewire_KNN_epoch', type=int, default=10, help="frequency of epochs to rewire")
    parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
    parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,cosine_power,pearson,rank_pearson")
    parser.add_argument('--max_epochs', type=int, default=1000, help="max epochs to train before patience")
    parser.add_argument('--patience', type=int, default=100, help="amount of patience for non improving val acc")

    args = parser.parse_args()
    opt = vars(args)
    main(opt)
