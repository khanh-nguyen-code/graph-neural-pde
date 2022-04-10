#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ray.tune import Analysis, ExperimentAnalysis
import json
import os
import matplotlib.pyplot as plt


# In[2]:


df_cols = ['accuracy', 'training_iteration', 'config/linear_attention', 'config/simple', 'config/ode', 'config/attention_norm_idx', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 'config/method', 'config/tol_scale', 'config/leaky_relu_slope', 'config/heads', 'config/attention_dim']


# In[3]:


cols = ['accuracy', 'training_iteration', 'config/num_init', 'config/function', 'config/block', 'config/simple', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 'config/method', 'config/tol_scale']


# In[4]:


cols_gdc = ['accuracy', 'training_iteration', 'config/num_init', 'config/reweight_attention', 'config/gdc_k', 'config/ppr_alpha', 'config/function', 'config/block', 'config/simple', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 'config/method', 'config/tol_scale']


# In[5]:


cols_OGB = ['accuracy', 'loss','training_iteration', 'config/num_init', 'config/function', 'config/block', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 'config/method', 'config/tol_scale']


# In[6]:


cols_OGB2 = ['accuracy', 'train_acc', 'loss','training_iteration', 'config/num_init', 'config/function', 'config/block', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 'config/method', 'config/tol_scale']


# In[7]:


att_cols = ['accuracy', 'training_iteration', 'config/num_init', 'config/function', 'config/block', 'config/simple', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 'config/method', 'config/tol_scale', 'config/attention_norm_idx', 'config/leaky_relu_slope', 'config/heads', 'config/attention_dim']


# In[8]:


def get_analysis(folder, cols = df_cols):
    analysis = Analysis("../ray_tune/{}".format(folder))
    df = analysis.dataframe(metric='accuracy', mode='max')
    return df.sort_values('accuracy', ascending=False)[cols]


# In[9]:


cd workspace/graph-neural-pde/src/


# In[ ]:


# df = get_analysis('OGB_test', attcols)
df = get_analysis('arxiv_att_lap', cols_OGB2)
print(len(df))


# In[11]:


df.head(50)


# In[ ]:





# In[14]:


df = get_analysis('cora_linear_attention_adjoint', att_cols)
df.head(50)


# In[11]:


df = get_analysis('cora_linear_attention', att_cols)
df.head(50)


# In[8]:


df = get_analysis('cora_gdc_attention_reweight', cols_gdc)
df.head(50)


# In[9]:


len(df)


# In[8]:


df = get_analysis('cora_gdc_attention', cols_gdc)
df.head(50)


# In[14]:


df = get_analysis('cora_gdc_weights', cols_gdc)
df.head(50)


# In[10]:


df = get_analysis('cora_gdc_search', cols_gdc)
df.tail(50)


# In[7]:


df = get_analysis('cora_gdc', cols)
df.head(50)


# In[11]:


df = get_analysis('cora_2hop', cols)
df.head(50)


# In[7]:


df = get_analysis('cora_GAT_refactor', att_cols)
df.head(50)


# In[10]:


df = get_analysis('cora_transformer_refactor_test', att_cols)
df.head(50)


# In[20]:


df = get_analysis('cora_mixed_block_refactor_test', att_cols)
df.head(50)


# In[18]:


df = get_analysis('cora_attention_block_refactor_test', att_cols)
df.head(50)


# In[11]:


cols = ['accuracy', 'training_iteration', 'config/num_init', 'config/function', 'config/block', 'config/simple', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 'config/method', 'config/tol_scale']
df = get_analysis('cora_refactor_test', cols)
df.head(50)


# In[10]:


df = get_analysis('cora_mix_att_fixed', df_cols)
df.head(50)


# In[17]:


df = get_analysis('cora_linear_att1', df_cols)
df.head(50)


# In[22]:


df = get_analysis('cora_linear_att')
df.head(50)


# In[20]:


df = get_analysis('cora_mix_att_lap1', df_cols)
df.head(50)


# In[14]:


df = get_analysis('cora_mix_att_lap', df_cols)
df.head(50)


# In[19]:


df = get_analysis('pubmed_linear_att_gp15_adjoint', df_cols)
df.head(50)


# In[5]:


df = get_analysis('citeseer_linear_att_gp15_adjoint', df_cols)
df.head(50)


# In[13]:


df.groupby(['config/attention_norm_idx'])['accuracy'].mean().plot.bar(ylim=[0.6,0.7])


# In[14]:


df.groupby(['config/simple'])['accuracy'].mean().plot.bar(ylim=[0.6,0.7])


# In[15]:


df.groupby(['config/attention_dim'])['accuracy'].mean().plot.bar(ylim=[0.6,0.7])


# In[18]:


df.groupby(['config/optimizer'])['accuracy'].mean().plot.bar(ylim=[0.6,0.7])


# In[15]:


df = get_analysis('citeseer_linear_att_gp15', df_cols)
df.head(50)


# In[13]:


df = get_analysis('citeseer_linear_att', df_cols)
df.head(50)


# In[9]:


df = get_analysis('refactor_test')
df.head(50)


# In[6]:


df = get_analysis('cora_att_mix_features')
df.head(50)


# In[7]:


df = get_analysis('cora_transformer_mix_features')
df.head(50)


# In[8]:


df = get_analysis('cora_transformer_norm_idx')
df.head(50)


# In[18]:


df = get_analysis('cora_att_dim')
print(len(df[df['config/attention_dim']==256]))
df.head(50)


# In[45]:


df = get_analysis('cora_transformer_mh')
df.head(50)


# In[40]:


df = get_analysis('cora_transformer_adjoint')
df.head(50)


# In[21]:


cols = ['accuracy', 'training_iteration', 'config/simple', 'config/ode', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/method', 'config/tol_scale', 'config/leaky_relu_slope', 'config/heads']
df = get_analysis('cora_transformer', cols)
df.head(50)


# In[7]:


df = get_analysis('pubmed_transformer')
df.head(50)


# In[7]:


cols = ['accuracy', 'config/adjoint', 'config/tol_scale_adjoint', 'training_iteration', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 
           'config/beta_dim', 'config/alpha_sigmoid', 'config/method', 'config/tol_scale', 'config/leaky_relu_slope', 'config/heads']
df = get_analysis('citeseer_transformer', cols)
df.head(50)


# In[8]:


df = get_analysis('cora_transformer1', cols)
df.head(50)


# In[16]:


df[0:100].groupby(['config/self_loop_weight'])['accuracy'].mean().plot.bar(ylim=[0.6,0.85])


# In[30]:


cols = ['accuracy', 'config/tol_scale_adjoint', 'config/adjoint', 'training_iteration', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 
           'config/beta_dim', 'config/alpha_sigmoid', 'config/method', 'config/tol_scale', 'config/leaky_relu_slope', 'config/heads']
df = get_analysis('cora_transformer', cols)
df.tail(50)


# In[25]:


analysis1 = Analysis("../ray_tune/ray_exp")


# In[30]:


df1 = analysis1.dataframe(metric='accuracy', mode='max')


# In[ ]:





# In[31]:


df1.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[2]:


analysis2 = Analysis("../ray_tune/alpha_beta")
df2 = analysis2.dataframe(metric='accuracy', mode='max')
df2.sort_values('accuracy', ascending=False).head(50)


# In[6]:


df2.sort_values('accuracy', ascending=False)['logdir'].head()


# In[7]:


best_params_dir = df2.sort_values('accuracy', ascending=False)['logdir'].iloc[0]
best_params_dir


# In[ ]:


../ray_tune/alpha_beta/DEFAULT_57f50_00076_76_...


# In[10]:


best_params_dir


# In[7]:


with open(best_params_dir + '/params.json') as f:
    best_params = json.loads(f.read())


# In[11]:


with open(best_params_dir + '/result.json') as f:
    best_params = json.loads(f.read())


# In[6]:


trials = analysis2.trial_dataframes


# In[9]:


analysis3 = Analysis("../ray_tune/ode")
df3 = analysis3.dataframe(metric='accuracy', mode='max')
df3.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[14]:


analysis4 = Analysis("../ray_tune/sigmoid")
df4 = analysis4.dataframe(metric='accuracy', mode='max')
df4.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[10]:


analysis5 = Analysis("../ray_tune/pop")
df5 = analysis5.dataframe(metric='accuracy', mode='max')
df5.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[13]:


analysis6 = Analysis("../ray_tune/method")
df6 = analysis6.dataframe(metric='accuracy', mode='max')
df6.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[18]:


analysis7 = Analysis("../ray_tune/tol")
df7 = analysis7.dataframe(metric='accuracy', mode='max')
df7.sort_values('accuracy', ascending=False)[df_cols].head(10)


# In[28]:


os.listdir(df8.sort_values('accuracy', ascending=False)['logdir'].iloc[0])


# In[26]:


analysis8 = Analysis("../ray_tune/pop1")
cols8 = ['accuracy', 'training_iteration', 'config/time', 'config/decay', 'config/hidden_dim', 'config/lr', 
 'config/input_dropout', 'config/self_loop_weight', 'config/dropout', 'config/optimizer', 'config/alpha_dim', 
           'config/beta_dim', 'config/alpha_sigmoid', 'config/method', 'config/tol_scale']
df8 = analysis8.dataframe(metric='accuracy', mode='max')
df8.sort_values('accuracy', ascending=False)[cols8].head(50)


# In[10]:


analysis9 = Analysis("../ray_tune/relu_slope")
df9 = analysis9.dataframe(metric='accuracy', mode='max')
df9.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[4]:


analysis10 = Analysis("../ray_tune/heads")
df10 = analysis10.dataframe(metric='accuracy', mode='max')
df10.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[5]:


analysis11 = Analysis("../ray_tune/heads_dims")
df11 = analysis11.dataframe(metric='accuracy', mode='max')
df11.sort_values('accuracy', ascending=False)[df_cols].head(50)


# In[12]:


df = get_analysis('citeseer')
df.head(50)


# In[13]:


df = get_analysis('citeseer_test')
df.head(50)


# In[ ]:


df = get_analysis('pubmed_test')
df.head(50)


# In[5]:


cd workspace/research-repo/DGDE/src/


# In[6]:


df = get_analysis('pubmed_theory')
df.head(50)


# In[13]:


df = get_analysis('theory')
df.tail(50)


# In[21]:


analysis = Analysis("../ray_tune/pop")
dfs = analysis.fetch_trial_dataframes()
# This plots everything on the same plot
ax = None
for d in dfs.values():
    ax = d.plot("training_iteration", "accuracy", ax=ax, legend=False)

plt.xlabel("iterations")
plt.ylabel("Test Accuracy")

print('best config:', analysis.get_best_config("mean_accuracy"))


# In[30]:


df = get_analysis('pop3')
df.head(50)


# In[ ]:




