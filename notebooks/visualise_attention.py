#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import time
import networkx as nx
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)
from importlib import reload
from best_params import best_params_dict, default_args
import data as dt


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


reload(dt)


# In[33]:


from GNN import GNN
from run_GNN import get_optimizer, train, test


# In[34]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# opt = {'self_loop_weight': 1, 'leaky_relu_slope': 0.2, 'heads': 2, 'K': 10,
#         'attention_norm_idx': 0, 'simple': True, 'alpha': 0, 'alpha_dim': 'sc', 'beta_dim': 'sc',
#         'hidden_dim': 64, 'block': 'attention', 'function': 'laplacian', 'alpha_sigmoid': True, 'augment': False, 'adjoint': False,
#         'tol_scale': 70, 'time': 20, 'input_dropout': 0.5, 'dropout': 0.2, 'method': 'dopri5', 'optimizer':'adam', 'lr':0.008,
#         'decay':0.007, 'epoch':30, 'kinetic_energy':None, 'jacobian_norm2':None, 'total_deriv':None, 'directional_penalty':None}
best_opt = best_params_dict['Cora']
opt = {**default_args, **best_opt}


# In[35]:


# dataset = dt.get_dataset('Cora', '../data', use_lcc=True)
dataset = dt.get_dataset(opt, '../data', opt['not_lcc'])
# dataset = dt.get_dataset({'dataset':'Cora', 'rewiring' : None}, '/content/graph-neural-pde/data', use_lcc=True) 
print(dataset.data.num_nodes, dataset.data.num_edges)


# In[36]:


model, dat = GNN(opt, dataset, device).to(device), dataset.data.to(device)


# In[37]:


parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])


# In[38]:


best_val_acc = test_acc = best_epoch = 0
for epoch in range(1, opt['epoch']):
    start_time = time.time()

    loss = train(model, optimizer, dat)
    train_acc, val_acc, tmp_test_acc = test(model, dat)

    if val_acc > best_val_acc:
      best_val_acc = val_acc
      test_acc = tmp_test_acc
      best_epoch = epoch
    log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(
      log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, train_acc, best_val_acc, test_acc))
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d}'.format(best_val_acc, test_acc, best_epoch))


# In[39]:


attention = model.odeblock.odefunc.attention_weights
edges = model.odeblock.odefunc.edge_index
print('edges shape: {}, attention shape: {}'.format(edges.shape, attention.shape))


# In[40]:


print(attention.min(), attention.mean(), attention.max())


# In[41]:


atts = attention.detach().numpy()[:,0]
print(atts.shape)


# In[42]:


get_ipython().run_cell_magic('time', '', 'plt.hist(atts, bins=np.linspace(0,1,11))')


# In[43]:


get_ipython().run_cell_magic('time', '', 'plt.hist(atts, bins=np.linspace(0,0.01,11))')


# In[44]:


print(attention.shape, edges.shape)


# In[29]:


labels = dataset.data.y
print(len(labels))


# In[55]:


def construct_graph(edges, attention=None, threshold=0.01):
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    if attention is not None:
        edges = edges[:, attention > threshold]
    edge_list = zip(edges[0], edges[1])
    g = nx.Graph(edge_list)
    return g


# In[136]:


edges = model.odeblock.odefunc.edge_index
g = construct_graph(edges)
print(g.edges([32]))
print(g.number_of_edges(), g.number_of_nodes(), nx.number_connected_components(g))


# In[137]:


g.remove_edges_from([(32, 387), (32, 790), (32, 791), (32, 1063), (32, 32)])
print(g.number_of_edges(), g.number_of_nodes(), nx.number_connected_components(g))


# In[124]:


delete_edges = edges[:, attention[:, 0].detach().cpu().numpy() < 0.1]
print(delete_edges.shape)


# In[146]:


def remove_edges(g, edges, attention, threshold):
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy()
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    delete_edges = edges[:, attention < threshold]
    print('deleting {} edges'.format(delete_edges.shape[1]))
    edge_list = list(zip(delete_edges[0], delete_edges[1]))
    g.remove_edges_from(edge_list)
    print(g.number_of_edges(), g.number_of_nodes(), nx.number_connected_components(g))
    return g


# In[139]:


g = remove_edges(g, edges, attention, threshold=0.02)
# print(g.number_of_edges(), g.number_of_nodes())


# In[119]:


g.edges([2])


# In[114]:


list(g.edges)[0:10]


# In[62]:


print(g.number_of_edges(), g.number_of_nodes())


# In[63]:


nx.number_connected_components(g)


# In[34]:


print(edges.shape, attention.shape)


# In[76]:


np.linspace(1,0,20)


# In[90]:


np.linspace(0,1,20)


# In[148]:


for threshold in np.linspace(0,0.01,20):
    edges = model.odeblock.odefunc.edge_index
    g = construct_graph(edges)
    attention = model.odeblock.odefunc.edge_weight
    g = remove_edges(g, edges, attention, threshold)
    comps = nx.number_connected_components(g)
    print('{} remaining edges. {} connected components at threshold {}'.format(g.number_of_edges(), comps, threshold))


# In[147]:


for threshold in np.linspace(0,0.01,20):
    edges = model.odeblock.odefunc.edge_index
    g = construct_graph(edges)
    attention = model.odeblock.odefunc.attention_weights[:,0]  # just using one head for now.
    g = remove_edges(g, edges, attention, threshold)
    comps = nx.number_connected_components(g)
    print('{} remaining edges. {} connected components at threshold {}'.format(g.number_of_edges(), comps, threshold))


# In[91]:


print(len(g.edges), len(g.nodes), g.number_of_nodes())


# In[186]:


get_ipython().run_cell_magic('time', '', "nx.draw(g, with_labels=False, font_weight='bold', node_size=5, node_color=labels)")


# In[191]:


# nx.connected_components(g)
print('connected: {}, n_components: {}, directed: {}'.format(nx.is_connected(g), nx.number_connected_components(g), g.is_directed()))


# In[192]:


type(atts)
atts.shape
len(g.edges)


# In[33]:


g.edges = g.edges[att]


# In[39]:





# In[44]:


ccs = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]


# In[46]:


len(ccs[0])


# In[47]:


g0 = g.subgraph(ccs[0])


# In[51]:


print(len(g0.edges), len(g0.nodes), g0.number_of_nodes())


# In[54]:


cc_idx = list(ccs[0])


# In[55]:


get_ipython().run_cell_magic('timeit', '', "nx.draw(g0, with_labels=False, font_weight='bold', node_size=5, node_color=labels[cc_idx])")


# In[ ]:


plt.savefig("path.png")

