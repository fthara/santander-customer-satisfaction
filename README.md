# santander-customer-satisfaction

Este projeto tem como objetivo prever a satisfação de um cliente do banco santander. No dataset oferecido pelo kaggle existem diversas features onde o maior desafio é realizar uma boa feature selection para poder descobrir o melhor resultado para o problema.

Vamos começar a nossa análise carregando as bibliotecas básicas para análise exploratória e em seguidas iremos carregar o dataset e ver algumas informações primordiais.


```python
#Bibliotecas para ler o dataframe e manipular os dados.
import pandas as pd
import numpy as np

#Bibliotecas para construir gráficos em Python.
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Permite ver todas as colunas e linhas do dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```


```python
# Carregando o dataset
df = pd.read_csv("train.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>imp_op_var40_ult1</th>
      <th>imp_op_var41_comer_ult1</th>
      <th>imp_op_var41_comer_ult3</th>
      <th>imp_op_var41_efect_ult1</th>
      <th>imp_op_var41_efect_ult3</th>
      <th>imp_op_var41_ult1</th>
      <th>imp_op_var39_efect_ult1</th>
      <th>imp_op_var39_efect_ult3</th>
      <th>imp_op_var39_ult1</th>
      <th>imp_sal_var16_ult1</th>
      <th>ind_var1_0</th>
      <th>ind_var1</th>
      <th>ind_var2_0</th>
      <th>ind_var2</th>
      <th>ind_var5_0</th>
      <th>ind_var5</th>
      <th>ind_var6_0</th>
      <th>ind_var6</th>
      <th>ind_var8_0</th>
      <th>ind_var8</th>
      <th>ind_var12_0</th>
      <th>ind_var12</th>
      <th>ind_var13_0</th>
      <th>ind_var13_corto_0</th>
      <th>ind_var13_corto</th>
      <th>ind_var13_largo_0</th>
      <th>ind_var13_largo</th>
      <th>ind_var13_medio_0</th>
      <th>ind_var13_medio</th>
      <th>ind_var13</th>
      <th>ind_var14_0</th>
      <th>ind_var14</th>
      <th>ind_var17_0</th>
      <th>ind_var17</th>
      <th>ind_var18_0</th>
      <th>ind_var18</th>
      <th>ind_var19</th>
      <th>ind_var20_0</th>
      <th>ind_var20</th>
      <th>ind_var24_0</th>
      <th>ind_var24</th>
      <th>ind_var25_cte</th>
      <th>ind_var26_0</th>
      <th>ind_var26_cte</th>
      <th>ind_var26</th>
      <th>ind_var25_0</th>
      <th>ind_var25</th>
      <th>ind_var27_0</th>
      <th>ind_var28_0</th>
      <th>ind_var28</th>
      <th>ind_var27</th>
      <th>ind_var29_0</th>
      <th>ind_var29</th>
      <th>ind_var30_0</th>
      <th>ind_var30</th>
      <th>ind_var31_0</th>
      <th>ind_var31</th>
      <th>ind_var32_cte</th>
      <th>ind_var32_0</th>
      <th>ind_var32</th>
      <th>ind_var33_0</th>
      <th>ind_var33</th>
      <th>ind_var34_0</th>
      <th>ind_var34</th>
      <th>ind_var37_cte</th>
      <th>ind_var37_0</th>
      <th>ind_var37</th>
      <th>ind_var39_0</th>
      <th>ind_var40_0</th>
      <th>ind_var40</th>
      <th>ind_var41_0</th>
      <th>ind_var41</th>
      <th>ind_var39</th>
      <th>ind_var44_0</th>
      <th>ind_var44</th>
      <th>ind_var46_0</th>
      <th>ind_var46</th>
      <th>num_var1_0</th>
      <th>num_var1</th>
      <th>num_var4</th>
      <th>num_var5_0</th>
      <th>num_var5</th>
      <th>num_var6_0</th>
      <th>num_var6</th>
      <th>num_var8_0</th>
      <th>num_var8</th>
      <th>num_var12_0</th>
      <th>num_var12</th>
      <th>num_var13_0</th>
      <th>num_var13_corto_0</th>
      <th>num_var13_corto</th>
      <th>num_var13_largo_0</th>
      <th>num_var13_largo</th>
      <th>num_var13_medio_0</th>
      <th>num_var13_medio</th>
      <th>num_var13</th>
      <th>num_var14_0</th>
      <th>num_var14</th>
      <th>num_var17_0</th>
      <th>num_var17</th>
      <th>num_var18_0</th>
      <th>num_var18</th>
      <th>num_var20_0</th>
      <th>num_var20</th>
      <th>num_var24_0</th>
      <th>num_var24</th>
      <th>num_var26_0</th>
      <th>num_var26</th>
      <th>num_var25_0</th>
      <th>num_var25</th>
      <th>num_op_var40_hace2</th>
      <th>num_op_var40_hace3</th>
      <th>num_op_var40_ult1</th>
      <th>num_op_var40_ult3</th>
      <th>num_op_var41_hace2</th>
      <th>num_op_var41_hace3</th>
      <th>num_op_var41_ult1</th>
      <th>num_op_var41_ult3</th>
      <th>num_op_var39_hace2</th>
      <th>num_op_var39_hace3</th>
      <th>num_op_var39_ult1</th>
      <th>num_op_var39_ult3</th>
      <th>num_var27_0</th>
      <th>num_var28_0</th>
      <th>num_var28</th>
      <th>num_var27</th>
      <th>num_var29_0</th>
      <th>num_var29</th>
      <th>num_var30_0</th>
      <th>num_var30</th>
      <th>num_var31_0</th>
      <th>num_var31</th>
      <th>num_var32_0</th>
      <th>num_var32</th>
      <th>num_var33_0</th>
      <th>num_var33</th>
      <th>num_var34_0</th>
      <th>num_var34</th>
      <th>num_var35</th>
      <th>num_var37_med_ult2</th>
      <th>num_var37_0</th>
      <th>num_var37</th>
      <th>num_var39_0</th>
      <th>num_var40_0</th>
      <th>num_var40</th>
      <th>num_var41_0</th>
      <th>num_var41</th>
      <th>num_var39</th>
      <th>num_var42_0</th>
      <th>num_var42</th>
      <th>num_var44_0</th>
      <th>num_var44</th>
      <th>num_var46_0</th>
      <th>num_var46</th>
      <th>saldo_var1</th>
      <th>saldo_var5</th>
      <th>saldo_var6</th>
      <th>saldo_var8</th>
      <th>saldo_var12</th>
      <th>saldo_var13_corto</th>
      <th>saldo_var13_largo</th>
      <th>saldo_var13_medio</th>
      <th>saldo_var13</th>
      <th>saldo_var14</th>
      <th>saldo_var17</th>
      <th>saldo_var18</th>
      <th>saldo_var20</th>
      <th>saldo_var24</th>
      <th>saldo_var26</th>
      <th>saldo_var25</th>
      <th>saldo_var28</th>
      <th>saldo_var27</th>
      <th>saldo_var29</th>
      <th>saldo_var30</th>
      <th>saldo_var31</th>
      <th>saldo_var32</th>
      <th>saldo_var33</th>
      <th>saldo_var34</th>
      <th>saldo_var37</th>
      <th>saldo_var40</th>
      <th>saldo_var41</th>
      <th>saldo_var42</th>
      <th>saldo_var44</th>
      <th>saldo_var46</th>
      <th>var36</th>
      <th>delta_imp_amort_var18_1y3</th>
      <th>delta_imp_amort_var34_1y3</th>
      <th>delta_imp_aport_var13_1y3</th>
      <th>delta_imp_aport_var17_1y3</th>
      <th>delta_imp_aport_var33_1y3</th>
      <th>delta_imp_compra_var44_1y3</th>
      <th>delta_imp_reemb_var13_1y3</th>
      <th>delta_imp_reemb_var17_1y3</th>
      <th>delta_imp_reemb_var33_1y3</th>
      <th>delta_imp_trasp_var17_in_1y3</th>
      <th>delta_imp_trasp_var17_out_1y3</th>
      <th>delta_imp_trasp_var33_in_1y3</th>
      <th>delta_imp_trasp_var33_out_1y3</th>
      <th>delta_imp_venta_var44_1y3</th>
      <th>delta_num_aport_var13_1y3</th>
      <th>delta_num_aport_var17_1y3</th>
      <th>delta_num_aport_var33_1y3</th>
      <th>delta_num_compra_var44_1y3</th>
      <th>delta_num_reemb_var13_1y3</th>
      <th>delta_num_reemb_var17_1y3</th>
      <th>delta_num_reemb_var33_1y3</th>
      <th>delta_num_trasp_var17_in_1y3</th>
      <th>delta_num_trasp_var17_out_1y3</th>
      <th>delta_num_trasp_var33_in_1y3</th>
      <th>delta_num_trasp_var33_out_1y3</th>
      <th>delta_num_venta_var44_1y3</th>
      <th>imp_amort_var18_hace3</th>
      <th>imp_amort_var18_ult1</th>
      <th>imp_amort_var34_hace3</th>
      <th>imp_amort_var34_ult1</th>
      <th>imp_aport_var13_hace3</th>
      <th>imp_aport_var13_ult1</th>
      <th>imp_aport_var17_hace3</th>
      <th>imp_aport_var17_ult1</th>
      <th>imp_aport_var33_hace3</th>
      <th>imp_aport_var33_ult1</th>
      <th>imp_var7_emit_ult1</th>
      <th>imp_var7_recib_ult1</th>
      <th>imp_compra_var44_hace3</th>
      <th>imp_compra_var44_ult1</th>
      <th>imp_reemb_var13_hace3</th>
      <th>imp_reemb_var13_ult1</th>
      <th>imp_reemb_var17_hace3</th>
      <th>imp_reemb_var17_ult1</th>
      <th>imp_reemb_var33_hace3</th>
      <th>imp_reemb_var33_ult1</th>
      <th>imp_var43_emit_ult1</th>
      <th>imp_trans_var37_ult1</th>
      <th>imp_trasp_var17_in_hace3</th>
      <th>imp_trasp_var17_in_ult1</th>
      <th>imp_trasp_var17_out_hace3</th>
      <th>imp_trasp_var17_out_ult1</th>
      <th>imp_trasp_var33_in_hace3</th>
      <th>imp_trasp_var33_in_ult1</th>
      <th>imp_trasp_var33_out_hace3</th>
      <th>imp_trasp_var33_out_ult1</th>
      <th>imp_venta_var44_hace3</th>
      <th>imp_venta_var44_ult1</th>
      <th>ind_var7_emit_ult1</th>
      <th>ind_var7_recib_ult1</th>
      <th>ind_var10_ult1</th>
      <th>ind_var10cte_ult1</th>
      <th>ind_var9_cte_ult1</th>
      <th>ind_var9_ult1</th>
      <th>ind_var43_emit_ult1</th>
      <th>ind_var43_recib_ult1</th>
      <th>var21</th>
      <th>num_var2_0_ult1</th>
      <th>num_var2_ult1</th>
      <th>num_aport_var13_hace3</th>
      <th>num_aport_var13_ult1</th>
      <th>num_aport_var17_hace3</th>
      <th>num_aport_var17_ult1</th>
      <th>num_aport_var33_hace3</th>
      <th>num_aport_var33_ult1</th>
      <th>num_var7_emit_ult1</th>
      <th>num_var7_recib_ult1</th>
      <th>num_compra_var44_hace3</th>
      <th>num_compra_var44_ult1</th>
      <th>num_ent_var16_ult1</th>
      <th>num_var22_hace2</th>
      <th>num_var22_hace3</th>
      <th>num_var22_ult1</th>
      <th>num_var22_ult3</th>
      <th>num_med_var22_ult3</th>
      <th>num_med_var45_ult3</th>
      <th>num_meses_var5_ult3</th>
      <th>num_meses_var8_ult3</th>
      <th>num_meses_var12_ult3</th>
      <th>num_meses_var13_corto_ult3</th>
      <th>num_meses_var13_largo_ult3</th>
      <th>num_meses_var13_medio_ult3</th>
      <th>num_meses_var17_ult3</th>
      <th>num_meses_var29_ult3</th>
      <th>num_meses_var33_ult3</th>
      <th>num_meses_var39_vig_ult3</th>
      <th>num_meses_var44_ult3</th>
      <th>num_op_var39_comer_ult1</th>
      <th>num_op_var39_comer_ult3</th>
      <th>num_op_var40_comer_ult1</th>
      <th>num_op_var40_comer_ult3</th>
      <th>num_op_var40_efect_ult1</th>
      <th>num_op_var40_efect_ult3</th>
      <th>num_op_var41_comer_ult1</th>
      <th>num_op_var41_comer_ult3</th>
      <th>num_op_var41_efect_ult1</th>
      <th>num_op_var41_efect_ult3</th>
      <th>num_op_var39_efect_ult1</th>
      <th>num_op_var39_efect_ult3</th>
      <th>num_reemb_var13_hace3</th>
      <th>num_reemb_var13_ult1</th>
      <th>num_reemb_var17_hace3</th>
      <th>num_reemb_var17_ult1</th>
      <th>num_reemb_var33_hace3</th>
      <th>num_reemb_var33_ult1</th>
      <th>num_sal_var16_ult1</th>
      <th>num_var43_emit_ult1</th>
      <th>num_var43_recib_ult1</th>
      <th>num_trasp_var11_ult1</th>
      <th>num_trasp_var17_in_hace3</th>
      <th>num_trasp_var17_in_ult1</th>
      <th>num_trasp_var17_out_hace3</th>
      <th>num_trasp_var17_out_ult1</th>
      <th>num_trasp_var33_in_hace3</th>
      <th>num_trasp_var33_in_ult1</th>
      <th>num_trasp_var33_out_hace3</th>
      <th>num_trasp_var33_out_ult1</th>
      <th>num_venta_var44_hace3</th>
      <th>num_venta_var44_ult1</th>
      <th>num_var45_hace2</th>
      <th>num_var45_hace3</th>
      <th>num_var45_ult1</th>
      <th>num_var45_ult3</th>
      <th>saldo_var2_ult1</th>
      <th>saldo_medio_var5_hace2</th>
      <th>saldo_medio_var5_hace3</th>
      <th>saldo_medio_var5_ult1</th>
      <th>saldo_medio_var5_ult3</th>
      <th>saldo_medio_var8_hace2</th>
      <th>saldo_medio_var8_hace3</th>
      <th>saldo_medio_var8_ult1</th>
      <th>saldo_medio_var8_ult3</th>
      <th>saldo_medio_var12_hace2</th>
      <th>saldo_medio_var12_hace3</th>
      <th>saldo_medio_var12_ult1</th>
      <th>saldo_medio_var12_ult3</th>
      <th>saldo_medio_var13_corto_hace2</th>
      <th>saldo_medio_var13_corto_hace3</th>
      <th>saldo_medio_var13_corto_ult1</th>
      <th>saldo_medio_var13_corto_ult3</th>
      <th>saldo_medio_var13_largo_hace2</th>
      <th>saldo_medio_var13_largo_hace3</th>
      <th>saldo_medio_var13_largo_ult1</th>
      <th>saldo_medio_var13_largo_ult3</th>
      <th>saldo_medio_var13_medio_hace2</th>
      <th>saldo_medio_var13_medio_hace3</th>
      <th>saldo_medio_var13_medio_ult1</th>
      <th>saldo_medio_var13_medio_ult3</th>
      <th>saldo_medio_var17_hace2</th>
      <th>saldo_medio_var17_hace3</th>
      <th>saldo_medio_var17_ult1</th>
      <th>saldo_medio_var17_ult3</th>
      <th>saldo_medio_var29_hace2</th>
      <th>saldo_medio_var29_hace3</th>
      <th>saldo_medio_var29_ult1</th>
      <th>saldo_medio_var29_ult3</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39205.170000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>300.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>300.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>88.89</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>300.0</td>
      <td>122.22</td>
      <td>300.0</td>
      <td>240.75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49278.030000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.18</td>
      <td>3.00</td>
      <td>2.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67333.770000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>37</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>70.62</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>70.62</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>34.95</td>
      <td>0.0</td>
      <td>0</td>
      <td>70.62</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>15</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
      <td>3</td>
      <td>18</td>
      <td>48</td>
      <td>0</td>
      <td>186.09</td>
      <td>0.00</td>
      <td>91.56</td>
      <td>138.84</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64007.970000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>2</td>
      <td>39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>135003.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>135003.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>135003.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>135003.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>135003.0</td>
      <td>270003.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.30</td>
      <td>40501.08</td>
      <td>13501.47</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>85501.89</td>
      <td>85501.89</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117310.979016</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



O arquivo possui muitas colunas, mas pela descrição todas são numéricas. Nesta análise exploratória vamos ver quais delas possui valores nulos. Também vamos verificar a correlação entre elas, fazer uma feature selection para descobrir quais são as features mais relevantes ao problema.


```python
df.shape
```




    (76020, 371)




```python
df.isna().sum()
```




    ID                               0
    var3                             0
    var15                            0
    imp_ent_var16_ult1               0
    imp_op_var39_comer_ult1          0
    imp_op_var39_comer_ult3          0
    imp_op_var40_comer_ult1          0
    imp_op_var40_comer_ult3          0
    imp_op_var40_efect_ult1          0
    imp_op_var40_efect_ult3          0
    imp_op_var40_ult1                0
    imp_op_var41_comer_ult1          0
    imp_op_var41_comer_ult3          0
    imp_op_var41_efect_ult1          0
    imp_op_var41_efect_ult3          0
    imp_op_var41_ult1                0
    imp_op_var39_efect_ult1          0
    imp_op_var39_efect_ult3          0
    imp_op_var39_ult1                0
    imp_sal_var16_ult1               0
    ind_var1_0                       0
    ind_var1                         0
    ind_var2_0                       0
    ind_var2                         0
    ind_var5_0                       0
    ind_var5                         0
    ind_var6_0                       0
    ind_var6                         0
    ind_var8_0                       0
    ind_var8                         0
    ind_var12_0                      0
    ind_var12                        0
    ind_var13_0                      0
    ind_var13_corto_0                0
    ind_var13_corto                  0
    ind_var13_largo_0                0
    ind_var13_largo                  0
    ind_var13_medio_0                0
    ind_var13_medio                  0
    ind_var13                        0
    ind_var14_0                      0
    ind_var14                        0
    ind_var17_0                      0
    ind_var17                        0
    ind_var18_0                      0
    ind_var18                        0
    ind_var19                        0
    ind_var20_0                      0
    ind_var20                        0
    ind_var24_0                      0
    ind_var24                        0
    ind_var25_cte                    0
    ind_var26_0                      0
    ind_var26_cte                    0
    ind_var26                        0
    ind_var25_0                      0
    ind_var25                        0
    ind_var27_0                      0
    ind_var28_0                      0
    ind_var28                        0
    ind_var27                        0
    ind_var29_0                      0
    ind_var29                        0
    ind_var30_0                      0
    ind_var30                        0
    ind_var31_0                      0
    ind_var31                        0
    ind_var32_cte                    0
    ind_var32_0                      0
    ind_var32                        0
    ind_var33_0                      0
    ind_var33                        0
    ind_var34_0                      0
    ind_var34                        0
    ind_var37_cte                    0
    ind_var37_0                      0
    ind_var37                        0
    ind_var39_0                      0
    ind_var40_0                      0
    ind_var40                        0
    ind_var41_0                      0
    ind_var41                        0
    ind_var39                        0
    ind_var44_0                      0
    ind_var44                        0
    ind_var46_0                      0
    ind_var46                        0
    num_var1_0                       0
    num_var1                         0
    num_var4                         0
    num_var5_0                       0
    num_var5                         0
    num_var6_0                       0
    num_var6                         0
    num_var8_0                       0
    num_var8                         0
    num_var12_0                      0
    num_var12                        0
    num_var13_0                      0
    num_var13_corto_0                0
    num_var13_corto                  0
    num_var13_largo_0                0
    num_var13_largo                  0
    num_var13_medio_0                0
    num_var13_medio                  0
    num_var13                        0
    num_var14_0                      0
    num_var14                        0
    num_var17_0                      0
    num_var17                        0
    num_var18_0                      0
    num_var18                        0
    num_var20_0                      0
    num_var20                        0
    num_var24_0                      0
    num_var24                        0
    num_var26_0                      0
    num_var26                        0
    num_var25_0                      0
    num_var25                        0
    num_op_var40_hace2               0
    num_op_var40_hace3               0
    num_op_var40_ult1                0
    num_op_var40_ult3                0
    num_op_var41_hace2               0
    num_op_var41_hace3               0
    num_op_var41_ult1                0
    num_op_var41_ult3                0
    num_op_var39_hace2               0
    num_op_var39_hace3               0
    num_op_var39_ult1                0
    num_op_var39_ult3                0
    num_var27_0                      0
    num_var28_0                      0
    num_var28                        0
    num_var27                        0
    num_var29_0                      0
    num_var29                        0
    num_var30_0                      0
    num_var30                        0
    num_var31_0                      0
    num_var31                        0
    num_var32_0                      0
    num_var32                        0
    num_var33_0                      0
    num_var33                        0
    num_var34_0                      0
    num_var34                        0
    num_var35                        0
    num_var37_med_ult2               0
    num_var37_0                      0
    num_var37                        0
    num_var39_0                      0
    num_var40_0                      0
    num_var40                        0
    num_var41_0                      0
    num_var41                        0
    num_var39                        0
    num_var42_0                      0
    num_var42                        0
    num_var44_0                      0
    num_var44                        0
    num_var46_0                      0
    num_var46                        0
    saldo_var1                       0
    saldo_var5                       0
    saldo_var6                       0
    saldo_var8                       0
    saldo_var12                      0
    saldo_var13_corto                0
    saldo_var13_largo                0
    saldo_var13_medio                0
    saldo_var13                      0
    saldo_var14                      0
    saldo_var17                      0
    saldo_var18                      0
    saldo_var20                      0
    saldo_var24                      0
    saldo_var26                      0
    saldo_var25                      0
    saldo_var28                      0
    saldo_var27                      0
    saldo_var29                      0
    saldo_var30                      0
    saldo_var31                      0
    saldo_var32                      0
    saldo_var33                      0
    saldo_var34                      0
    saldo_var37                      0
    saldo_var40                      0
    saldo_var41                      0
    saldo_var42                      0
    saldo_var44                      0
    saldo_var46                      0
    var36                            0
    delta_imp_amort_var18_1y3        0
    delta_imp_amort_var34_1y3        0
    delta_imp_aport_var13_1y3        0
    delta_imp_aport_var17_1y3        0
    delta_imp_aport_var33_1y3        0
    delta_imp_compra_var44_1y3       0
    delta_imp_reemb_var13_1y3        0
    delta_imp_reemb_var17_1y3        0
    delta_imp_reemb_var33_1y3        0
    delta_imp_trasp_var17_in_1y3     0
    delta_imp_trasp_var17_out_1y3    0
    delta_imp_trasp_var33_in_1y3     0
    delta_imp_trasp_var33_out_1y3    0
    delta_imp_venta_var44_1y3        0
    delta_num_aport_var13_1y3        0
    delta_num_aport_var17_1y3        0
    delta_num_aport_var33_1y3        0
    delta_num_compra_var44_1y3       0
    delta_num_reemb_var13_1y3        0
    delta_num_reemb_var17_1y3        0
    delta_num_reemb_var33_1y3        0
    delta_num_trasp_var17_in_1y3     0
    delta_num_trasp_var17_out_1y3    0
    delta_num_trasp_var33_in_1y3     0
    delta_num_trasp_var33_out_1y3    0
    delta_num_venta_var44_1y3        0
    imp_amort_var18_hace3            0
    imp_amort_var18_ult1             0
    imp_amort_var34_hace3            0
    imp_amort_var34_ult1             0
    imp_aport_var13_hace3            0
    imp_aport_var13_ult1             0
    imp_aport_var17_hace3            0
    imp_aport_var17_ult1             0
    imp_aport_var33_hace3            0
    imp_aport_var33_ult1             0
    imp_var7_emit_ult1               0
    imp_var7_recib_ult1              0
    imp_compra_var44_hace3           0
    imp_compra_var44_ult1            0
    imp_reemb_var13_hace3            0
    imp_reemb_var13_ult1             0
    imp_reemb_var17_hace3            0
    imp_reemb_var17_ult1             0
    imp_reemb_var33_hace3            0
    imp_reemb_var33_ult1             0
    imp_var43_emit_ult1              0
    imp_trans_var37_ult1             0
    imp_trasp_var17_in_hace3         0
    imp_trasp_var17_in_ult1          0
    imp_trasp_var17_out_hace3        0
    imp_trasp_var17_out_ult1         0
    imp_trasp_var33_in_hace3         0
    imp_trasp_var33_in_ult1          0
    imp_trasp_var33_out_hace3        0
    imp_trasp_var33_out_ult1         0
    imp_venta_var44_hace3            0
    imp_venta_var44_ult1             0
    ind_var7_emit_ult1               0
    ind_var7_recib_ult1              0
    ind_var10_ult1                   0
    ind_var10cte_ult1                0
    ind_var9_cte_ult1                0
    ind_var9_ult1                    0
    ind_var43_emit_ult1              0
    ind_var43_recib_ult1             0
    var21                            0
    num_var2_0_ult1                  0
    num_var2_ult1                    0
    num_aport_var13_hace3            0
    num_aport_var13_ult1             0
    num_aport_var17_hace3            0
    num_aport_var17_ult1             0
    num_aport_var33_hace3            0
    num_aport_var33_ult1             0
    num_var7_emit_ult1               0
    num_var7_recib_ult1              0
    num_compra_var44_hace3           0
    num_compra_var44_ult1            0
    num_ent_var16_ult1               0
    num_var22_hace2                  0
    num_var22_hace3                  0
    num_var22_ult1                   0
    num_var22_ult3                   0
    num_med_var22_ult3               0
    num_med_var45_ult3               0
    num_meses_var5_ult3              0
    num_meses_var8_ult3              0
    num_meses_var12_ult3             0
    num_meses_var13_corto_ult3       0
    num_meses_var13_largo_ult3       0
    num_meses_var13_medio_ult3       0
    num_meses_var17_ult3             0
    num_meses_var29_ult3             0
    num_meses_var33_ult3             0
    num_meses_var39_vig_ult3         0
    num_meses_var44_ult3             0
    num_op_var39_comer_ult1          0
    num_op_var39_comer_ult3          0
    num_op_var40_comer_ult1          0
    num_op_var40_comer_ult3          0
    num_op_var40_efect_ult1          0
    num_op_var40_efect_ult3          0
    num_op_var41_comer_ult1          0
    num_op_var41_comer_ult3          0
    num_op_var41_efect_ult1          0
    num_op_var41_efect_ult3          0
    num_op_var39_efect_ult1          0
    num_op_var39_efect_ult3          0
    num_reemb_var13_hace3            0
    num_reemb_var13_ult1             0
    num_reemb_var17_hace3            0
    num_reemb_var17_ult1             0
    num_reemb_var33_hace3            0
    num_reemb_var33_ult1             0
    num_sal_var16_ult1               0
    num_var43_emit_ult1              0
    num_var43_recib_ult1             0
    num_trasp_var11_ult1             0
    num_trasp_var17_in_hace3         0
    num_trasp_var17_in_ult1          0
    num_trasp_var17_out_hace3        0
    num_trasp_var17_out_ult1         0
    num_trasp_var33_in_hace3         0
    num_trasp_var33_in_ult1          0
    num_trasp_var33_out_hace3        0
    num_trasp_var33_out_ult1         0
    num_venta_var44_hace3            0
    num_venta_var44_ult1             0
    num_var45_hace2                  0
    num_var45_hace3                  0
    num_var45_ult1                   0
    num_var45_ult3                   0
    saldo_var2_ult1                  0
    saldo_medio_var5_hace2           0
    saldo_medio_var5_hace3           0
    saldo_medio_var5_ult1            0
    saldo_medio_var5_ult3            0
    saldo_medio_var8_hace2           0
    saldo_medio_var8_hace3           0
    saldo_medio_var8_ult1            0
    saldo_medio_var8_ult3            0
    saldo_medio_var12_hace2          0
    saldo_medio_var12_hace3          0
    saldo_medio_var12_ult1           0
    saldo_medio_var12_ult3           0
    saldo_medio_var13_corto_hace2    0
    saldo_medio_var13_corto_hace3    0
    saldo_medio_var13_corto_ult1     0
    saldo_medio_var13_corto_ult3     0
    saldo_medio_var13_largo_hace2    0
    saldo_medio_var13_largo_hace3    0
    saldo_medio_var13_largo_ult1     0
    saldo_medio_var13_largo_ult3     0
    saldo_medio_var13_medio_hace2    0
    saldo_medio_var13_medio_hace3    0
    saldo_medio_var13_medio_ult1     0
    saldo_medio_var13_medio_ult3     0
    saldo_medio_var17_hace2          0
    saldo_medio_var17_hace3          0
    saldo_medio_var17_ult1           0
    saldo_medio_var17_ult3           0
    saldo_medio_var29_hace2          0
    saldo_medio_var29_hace3          0
    saldo_medio_var29_ult1           0
    saldo_medio_var29_ult3           0
    saldo_medio_var33_hace2          0
    saldo_medio_var33_hace3          0
    saldo_medio_var33_ult1           0
    saldo_medio_var33_ult3           0
    saldo_medio_var44_hace2          0
    saldo_medio_var44_hace3          0
    saldo_medio_var44_ult1           0
    saldo_medio_var44_ult3           0
    var38                            0
    TARGET                           0
    dtype: int64



Como podemos perceber não há nenhum valor nulo neste dataset, mas vendo o head percebo muitos valores 0 nas features. Vamos analizar o dataset para ver se existem colunas com apenas um valor.


```python
cols_0 = df.columns[df.nunique() <= 1]
len(cols_0)
```




    34



Existem 34 colunas que possuem um único elemento. Para fins de análise preditiva isso não vai ajudar em nada, por isso serão excluídos do dataset.


```python
cols_list_0 = cols_0.tolist()
df_columns = df.columns.tolist()
cols_list = list(set(df_columns) - set(cols_list_0))
len(cols_list)
```




    337




```python
df_01 = df[cols_list]
```


```python
df_01.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_op_var41_comer_ult3</th>
      <th>delta_imp_reemb_var33_1y3</th>
      <th>ind_var33_0</th>
      <th>num_var24_0</th>
      <th>num_var39</th>
      <th>ind_var18</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>ind_var25_cte</th>
      <th>num_trasp_var33_in_hace3</th>
      <th>num_op_var40_efect_ult1</th>
      <th>ind_var34</th>
      <th>ind_var32</th>
      <th>saldo_var32</th>
      <th>saldo_var13</th>
      <th>saldo_medio_var5_hace3</th>
      <th>ind_var19</th>
      <th>num_venta_var44_ult1</th>
      <th>num_op_var39_hace2</th>
      <th>var36</th>
      <th>num_var45_ult3</th>
      <th>ind_var6</th>
      <th>num_var32</th>
      <th>saldo_var12</th>
      <th>imp_op_var41_comer_ult3</th>
      <th>saldo_var40</th>
      <th>saldo_medio_var29_ult3</th>
      <th>delta_imp_amort_var34_1y3</th>
      <th>num_op_var41_comer_ult1</th>
      <th>num_reemb_var17_hace3</th>
      <th>num_var44_0</th>
      <th>num_var12_0</th>
      <th>saldo_var37</th>
      <th>delta_imp_aport_var13_1y3</th>
      <th>num_meses_var5_ult3</th>
      <th>ind_var18_0</th>
      <th>imp_aport_var33_hace3</th>
      <th>num_meses_var44_ult3</th>
      <th>num_var43_emit_ult1</th>
      <th>num_var13_0</th>
      <th>delta_num_aport_var17_1y3</th>
      <th>num_var18_0</th>
      <th>imp_reemb_var17_hace3</th>
      <th>num_op_var39_efect_ult1</th>
      <th>ind_var13_corto_0</th>
      <th>num_var34</th>
      <th>saldo_medio_var33_ult3</th>
      <th>ind_var1</th>
      <th>ind_var40</th>
      <th>delta_imp_venta_var44_1y3</th>
      <th>saldo_medio_var13_largo_ult3</th>
      <th>num_var45_hace2</th>
      <th>num_var13_largo</th>
      <th>num_var7_emit_ult1</th>
      <th>imp_op_var41_ult1</th>
      <th>ind_var30</th>
      <th>saldo_medio_var13_largo_ult1</th>
      <th>num_var4</th>
      <th>saldo_medio_var5_ult3</th>
      <th>saldo_var34</th>
      <th>saldo_medio_var13_corto_ult1</th>
      <th>ind_var14_0</th>
      <th>saldo_var42</th>
      <th>imp_aport_var17_ult1</th>
      <th>saldo_medio_var13_corto_hace3</th>
      <th>num_var13_corto_0</th>
      <th>num_var6</th>
      <th>num_op_var41_hace2</th>
      <th>delta_imp_compra_var44_1y3</th>
      <th>num_op_var41_efect_ult3</th>
      <th>num_trasp_var33_in_ult1</th>
      <th>num_var13_corto</th>
      <th>num_reemb_var17_ult1</th>
      <th>imp_aport_var13_ult1</th>
      <th>saldo_medio_var8_ult1</th>
      <th>num_op_var39_hace3</th>
      <th>ind_var13_largo</th>
      <th>num_var7_recib_ult1</th>
      <th>num_var41_0</th>
      <th>num_var1_0</th>
      <th>num_var31_0</th>
      <th>num_var40</th>
      <th>imp_trasp_var17_in_hace3</th>
      <th>delta_imp_amort_var18_1y3</th>
      <th>delta_imp_aport_var33_1y3</th>
      <th>delta_num_reemb_var33_1y3</th>
      <th>ind_var13_largo_0</th>
      <th>delta_num_aport_var33_1y3</th>
      <th>num_trasp_var33_out_ult1</th>
      <th>delta_num_reemb_var17_1y3</th>
      <th>saldo_medio_var17_hace2</th>
      <th>ind_var20_0</th>
      <th>num_var42</th>
      <th>num_op_var40_ult1</th>
      <th>saldo_medio_var13_largo_hace2</th>
      <th>num_reemb_var33_ult1</th>
      <th>saldo_medio_var33_ult1</th>
      <th>ind_var24_0</th>
      <th>num_var26</th>
      <th>num_meses_var39_vig_ult3</th>
      <th>num_var42_0</th>
      <th>imp_compra_var44_ult1</th>
      <th>delta_num_trasp_var17_out_1y3</th>
      <th>num_aport_var17_hace3</th>
      <th>ind_var39</th>
      <th>imp_amort_var34_ult1</th>
      <th>num_var30_0</th>
      <th>num_var45_ult1</th>
      <th>imp_reemb_var17_ult1</th>
      <th>delta_imp_reemb_var17_1y3</th>
      <th>num_op_var40_ult3</th>
      <th>saldo_medio_var12_hace3</th>
      <th>saldo_medio_var13_largo_hace3</th>
      <th>ind_var20</th>
      <th>num_var14</th>
      <th>num_var32_0</th>
      <th>num_var29</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>saldo_medio_var44_ult1</th>
      <th>num_trasp_var17_in_ult1</th>
      <th>saldo_var44</th>
      <th>saldo_medio_var33_hace2</th>
      <th>ind_var31_0</th>
      <th>saldo_medio_var12_ult1</th>
      <th>ind_var26</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>saldo_var14</th>
      <th>imp_var7_recib_ult1</th>
      <th>num_compra_var44_hace3</th>
      <th>num_med_var45_ult3</th>
      <th>num_var31</th>
      <th>num_var17</th>
      <th>num_var22_hace3</th>
      <th>num_aport_var33_ult1</th>
      <th>imp_op_var39_efect_ult1</th>
      <th>imp_sal_var16_ult1</th>
      <th>num_var37_med_ult2</th>
      <th>imp_ent_var16_ult1</th>
      <th>num_var8_0</th>
      <th>ind_var12</th>
      <th>num_aport_var13_ult1</th>
      <th>ind_var13_medio_0</th>
      <th>num_op_var39_ult1</th>
      <th>num_var26_0</th>
      <th>ind_var33</th>
      <th>num_var13_medio_0</th>
      <th>num_var13</th>
      <th>imp_op_var41_efect_ult3</th>
      <th>var21</th>
      <th>saldo_var1</th>
      <th>ind_var31</th>
      <th>num_meses_var13_corto_ult3</th>
      <th>num_op_var39_comer_ult1</th>
      <th>var15</th>
      <th>num_var20_0</th>
      <th>saldo_medio_var13_corto_hace2</th>
      <th>ind_var37_cte</th>
      <th>saldo_var13_medio</th>
      <th>ind_var40_0</th>
      <th>saldo_medio_var17_hace3</th>
      <th>num_var22_ult1</th>
      <th>num_var30</th>
      <th>saldo_medio_var44_ult3</th>
      <th>saldo_var6</th>
      <th>saldo_var18</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>saldo_var20</th>
      <th>num_var5_0</th>
      <th>ind_var32_0</th>
      <th>num_op_var41_efect_ult1</th>
      <th>num_var6_0</th>
      <th>num_trasp_var17_out_ult1</th>
      <th>num_op_var41_ult1</th>
      <th>saldo_var26</th>
      <th>ind_var6_0</th>
      <th>num_var20</th>
      <th>num_var18</th>
      <th>imp_reemb_var13_ult1</th>
      <th>ind_var26_cte</th>
      <th>saldo_var17</th>
      <th>ind_var26_0</th>
      <th>imp_venta_var44_ult1</th>
      <th>saldo_medio_var29_hace2</th>
      <th>imp_venta_var44_hace3</th>
      <th>num_var33_0</th>
      <th>saldo_var30</th>
      <th>imp_trasp_var33_out_ult1</th>
      <th>num_var43_recib_ult1</th>
      <th>ind_var29_0</th>
      <th>num_compra_var44_ult1</th>
      <th>delta_imp_aport_var17_1y3</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>ind_var37</th>
      <th>saldo_var24</th>
      <th>saldo_var13_corto</th>
      <th>TARGET</th>
      <th>num_var37</th>
      <th>saldo_var31</th>
      <th>saldo_medio_var44_hace2</th>
      <th>ind_var12_0</th>
      <th>ind_var44</th>
      <th>num_meses_var13_medio_ult3</th>
      <th>num_var12</th>
      <th>ind_var17</th>
      <th>num_meses_var17_ult3</th>
      <th>delta_imp_trasp_var33_out_1y3</th>
      <th>ind_var39_0</th>
      <th>num_var33</th>
      <th>imp_trasp_var33_in_ult1</th>
      <th>num_var22_hace2</th>
      <th>ind_var10cte_ult1</th>
      <th>num_var8</th>
      <th>imp_compra_var44_hace3</th>
      <th>num_sal_var16_ult1</th>
      <th>imp_var43_emit_ult1</th>
      <th>imp_op_var39_efect_ult3</th>
      <th>ind_var5</th>
      <th>saldo_medio_var12_ult3</th>
      <th>num_var1</th>
      <th>num_meses_var33_ult3</th>
      <th>ind_var9_ult1</th>
      <th>saldo_medio_var44_hace3</th>
      <th>ind_var30_0</th>
      <th>ID</th>
      <th>ind_var13_medio</th>
      <th>delta_imp_trasp_var33_in_1y3</th>
      <th>imp_trasp_var17_out_ult1</th>
      <th>delta_imp_reemb_var13_1y3</th>
      <th>saldo_var25</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>num_op_var41_ult3</th>
      <th>var38</th>
      <th>num_trasp_var17_in_hace3</th>
      <th>ind_var13_0</th>
      <th>saldo_var29</th>
      <th>imp_amort_var18_ult1</th>
      <th>saldo_medio_var13_medio_ult3</th>
      <th>saldo_medio_var17_ult3</th>
      <th>num_op_var39_comer_ult3</th>
      <th>num_var22_ult3</th>
      <th>ind_var24</th>
      <th>num_var29_0</th>
      <th>saldo_medio_var17_ult1</th>
      <th>imp_var7_emit_ult1</th>
      <th>ind_var9_cte_ult1</th>
      <th>delta_imp_trasp_var17_out_1y3</th>
      <th>num_op_var40_hace2</th>
      <th>ind_var7_recib_ult1</th>
      <th>num_var13_largo_0</th>
      <th>imp_trans_var37_ult1</th>
      <th>num_var13_medio</th>
      <th>num_venta_var44_hace3</th>
      <th>imp_op_var40_ult1</th>
      <th>saldo_medio_var5_hace2</th>
      <th>saldo_var33</th>
      <th>num_aport_var17_ult1</th>
      <th>num_var14_0</th>
      <th>saldo_medio_var13_corto_ult3</th>
      <th>num_var44</th>
      <th>num_meses_var13_largo_ult3</th>
      <th>num_meses_var12_ult3</th>
      <th>ind_var5_0</th>
      <th>num_reemb_var13_ult1</th>
      <th>num_op_var40_efect_ult3</th>
      <th>num_op_var41_hace3</th>
      <th>delta_num_venta_var44_1y3</th>
      <th>ind_var8</th>
      <th>num_meses_var29_ult3</th>
      <th>imp_reemb_var33_ult1</th>
      <th>num_aport_var33_hace3</th>
      <th>imp_trasp_var33_in_hace3</th>
      <th>saldo_medio_var29_ult1</th>
      <th>ind_var43_recib_ult1</th>
      <th>saldo_medio_var5_ult1</th>
      <th>saldo_var8</th>
      <th>ind_var10_ult1</th>
      <th>ind_var25_0</th>
      <th>num_var34_0</th>
      <th>num_trasp_var11_ult1</th>
      <th>imp_aport_var17_hace3</th>
      <th>saldo_medio_var33_hace3</th>
      <th>ind_var32_cte</th>
      <th>num_var17_0</th>
      <th>num_var35</th>
      <th>num_ent_var16_ult1</th>
      <th>ind_var41_0</th>
      <th>num_var37_0</th>
      <th>ind_var13</th>
      <th>delta_imp_trasp_var17_in_1y3</th>
      <th>ind_var13_corto</th>
      <th>imp_trasp_var17_in_ult1</th>
      <th>delta_num_trasp_var33_out_1y3</th>
      <th>num_var25</th>
      <th>num_op_var39_efect_ult3</th>
      <th>ind_var34_0</th>
      <th>delta_num_trasp_var33_in_1y3</th>
      <th>ind_var43_emit_ult1</th>
      <th>num_var24</th>
      <th>saldo_var13_largo</th>
      <th>imp_aport_var13_hace3</th>
      <th>num_var45_hace3</th>
      <th>num_var40_0</th>
      <th>saldo_medio_var29_hace3</th>
      <th>delta_num_compra_var44_1y3</th>
      <th>ind_var37_0</th>
      <th>imp_aport_var33_ult1</th>
      <th>saldo_medio_var13_medio_ult1</th>
      <th>saldo_medio_var8_ult3</th>
      <th>ind_var25</th>
      <th>imp_op_var41_comer_ult1</th>
      <th>num_op_var40_comer_ult1</th>
      <th>ind_var8_0</th>
      <th>num_op_var39_ult3</th>
      <th>num_aport_var13_hace3</th>
      <th>num_op_var40_hace3</th>
      <th>ind_var44_0</th>
      <th>imp_op_var41_efect_ult1</th>
      <th>saldo_medio_var8_hace2</th>
      <th>saldo_medio_var12_hace2</th>
      <th>num_op_var40_comer_ult3</th>
      <th>ind_var29</th>
      <th>saldo_medio_var8_hace3</th>
      <th>ind_var14</th>
      <th>num_meses_var8_ult3</th>
      <th>num_var5</th>
      <th>num_var25_0</th>
      <th>delta_num_trasp_var17_in_1y3</th>
      <th>var3</th>
      <th>delta_num_aport_var13_1y3</th>
      <th>ind_var1_0</th>
      <th>num_med_var22_ult3</th>
      <th>ind_var7_emit_ult1</th>
      <th>saldo_var5</th>
      <th>imp_op_var39_ult1</th>
      <th>saldo_medio_var13_medio_hace2</th>
      <th>delta_num_reemb_var13_1y3</th>
      <th>ind_var17_0</th>
      <th>num_var39_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>39205.170000</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>88.89</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>-1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>300.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>122.22</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>34</td>
      <td>0</td>
      <td>300.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>300.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>49278.030000</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>240.75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>-1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.07</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>67333.770000</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>48</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34.95</td>
      <td>0.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>195.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
      <td>138.84</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>70.62</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>18</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>37</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>70.62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>9</td>
      <td>64007.970000</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>186.09</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>91.56</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>195.0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70.62</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>135003.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>13501.47</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>135003.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>85501.89</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>135003.00</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>135003.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>135003.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>85501.89</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>117310.979016</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>270003.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>40501.08</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Como este dataset possui muitas features fica um pouco complicada a análise exploratória através de gráficos, portanto para escolher as melhores features para nosso algoritmo vou utilizar algumas técnicas e fazer o treinamento do modelo. Logo neste caso quem fará a feature selction será o computador, mas com a ajuda de algumas técnicas.

# Correlação

A primeira técnica que irei utilizar será a correlação, que é muito utilizada para variáveis quantitativas e mostra qual é a relação existente entre duas variáveis. A correlação varia de -1 a 1, onde quanto mais próximo de -1 maior a correlação negativa, ou seja a diferença entre eles (ex: quando um sobe outro desce), e quanto mais próximo de 1, maior será a correlação positiva, ou seja maior a semelhança entre eles (ex: quando um sobe o outro também sobe). Portanto as melhores variáveis são aquelas mais próximas de -1 ou de 1. As que estão próximas de 0 siginifica que não existe quase nenhuma correlação entre elas.


```python
matriz_corr = df_01.corr()
matriz_corr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_op_var41_comer_ult3</th>
      <th>delta_imp_reemb_var33_1y3</th>
      <th>ind_var33_0</th>
      <th>num_var24_0</th>
      <th>num_var39</th>
      <th>ind_var18</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>ind_var25_cte</th>
      <th>num_trasp_var33_in_hace3</th>
      <th>num_op_var40_efect_ult1</th>
      <th>ind_var34</th>
      <th>ind_var32</th>
      <th>saldo_var32</th>
      <th>saldo_var13</th>
      <th>saldo_medio_var5_hace3</th>
      <th>ind_var19</th>
      <th>num_venta_var44_ult1</th>
      <th>num_op_var39_hace2</th>
      <th>var36</th>
      <th>num_var45_ult3</th>
      <th>ind_var6</th>
      <th>num_var32</th>
      <th>saldo_var12</th>
      <th>imp_op_var41_comer_ult3</th>
      <th>saldo_var40</th>
      <th>saldo_medio_var29_ult3</th>
      <th>delta_imp_amort_var34_1y3</th>
      <th>num_op_var41_comer_ult1</th>
      <th>num_reemb_var17_hace3</th>
      <th>num_var44_0</th>
      <th>num_var12_0</th>
      <th>saldo_var37</th>
      <th>delta_imp_aport_var13_1y3</th>
      <th>num_meses_var5_ult3</th>
      <th>ind_var18_0</th>
      <th>imp_aport_var33_hace3</th>
      <th>num_meses_var44_ult3</th>
      <th>num_var43_emit_ult1</th>
      <th>num_var13_0</th>
      <th>delta_num_aport_var17_1y3</th>
      <th>num_var18_0</th>
      <th>imp_reemb_var17_hace3</th>
      <th>num_op_var39_efect_ult1</th>
      <th>ind_var13_corto_0</th>
      <th>num_var34</th>
      <th>saldo_medio_var33_ult3</th>
      <th>ind_var1</th>
      <th>ind_var40</th>
      <th>delta_imp_venta_var44_1y3</th>
      <th>saldo_medio_var13_largo_ult3</th>
      <th>num_var45_hace2</th>
      <th>num_var13_largo</th>
      <th>num_var7_emit_ult1</th>
      <th>imp_op_var41_ult1</th>
      <th>ind_var30</th>
      <th>saldo_medio_var13_largo_ult1</th>
      <th>num_var4</th>
      <th>saldo_medio_var5_ult3</th>
      <th>saldo_var34</th>
      <th>saldo_medio_var13_corto_ult1</th>
      <th>ind_var14_0</th>
      <th>saldo_var42</th>
      <th>imp_aport_var17_ult1</th>
      <th>saldo_medio_var13_corto_hace3</th>
      <th>num_var13_corto_0</th>
      <th>num_var6</th>
      <th>num_op_var41_hace2</th>
      <th>delta_imp_compra_var44_1y3</th>
      <th>num_op_var41_efect_ult3</th>
      <th>num_trasp_var33_in_ult1</th>
      <th>num_var13_corto</th>
      <th>num_reemb_var17_ult1</th>
      <th>imp_aport_var13_ult1</th>
      <th>saldo_medio_var8_ult1</th>
      <th>num_op_var39_hace3</th>
      <th>ind_var13_largo</th>
      <th>num_var7_recib_ult1</th>
      <th>num_var41_0</th>
      <th>num_var1_0</th>
      <th>num_var31_0</th>
      <th>num_var40</th>
      <th>imp_trasp_var17_in_hace3</th>
      <th>delta_imp_amort_var18_1y3</th>
      <th>delta_imp_aport_var33_1y3</th>
      <th>delta_num_reemb_var33_1y3</th>
      <th>ind_var13_largo_0</th>
      <th>delta_num_aport_var33_1y3</th>
      <th>num_trasp_var33_out_ult1</th>
      <th>delta_num_reemb_var17_1y3</th>
      <th>saldo_medio_var17_hace2</th>
      <th>ind_var20_0</th>
      <th>num_var42</th>
      <th>num_op_var40_ult1</th>
      <th>saldo_medio_var13_largo_hace2</th>
      <th>num_reemb_var33_ult1</th>
      <th>saldo_medio_var33_ult1</th>
      <th>ind_var24_0</th>
      <th>num_var26</th>
      <th>num_meses_var39_vig_ult3</th>
      <th>num_var42_0</th>
      <th>imp_compra_var44_ult1</th>
      <th>delta_num_trasp_var17_out_1y3</th>
      <th>num_aport_var17_hace3</th>
      <th>ind_var39</th>
      <th>imp_amort_var34_ult1</th>
      <th>num_var30_0</th>
      <th>num_var45_ult1</th>
      <th>imp_reemb_var17_ult1</th>
      <th>delta_imp_reemb_var17_1y3</th>
      <th>num_op_var40_ult3</th>
      <th>saldo_medio_var12_hace3</th>
      <th>saldo_medio_var13_largo_hace3</th>
      <th>ind_var20</th>
      <th>num_var14</th>
      <th>num_var32_0</th>
      <th>num_var29</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>saldo_medio_var44_ult1</th>
      <th>num_trasp_var17_in_ult1</th>
      <th>saldo_var44</th>
      <th>saldo_medio_var33_hace2</th>
      <th>ind_var31_0</th>
      <th>saldo_medio_var12_ult1</th>
      <th>ind_var26</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>saldo_var14</th>
      <th>imp_var7_recib_ult1</th>
      <th>num_compra_var44_hace3</th>
      <th>num_med_var45_ult3</th>
      <th>num_var31</th>
      <th>num_var17</th>
      <th>num_var22_hace3</th>
      <th>num_aport_var33_ult1</th>
      <th>imp_op_var39_efect_ult1</th>
      <th>imp_sal_var16_ult1</th>
      <th>num_var37_med_ult2</th>
      <th>imp_ent_var16_ult1</th>
      <th>num_var8_0</th>
      <th>ind_var12</th>
      <th>num_aport_var13_ult1</th>
      <th>ind_var13_medio_0</th>
      <th>num_op_var39_ult1</th>
      <th>num_var26_0</th>
      <th>ind_var33</th>
      <th>num_var13_medio_0</th>
      <th>num_var13</th>
      <th>imp_op_var41_efect_ult3</th>
      <th>var21</th>
      <th>saldo_var1</th>
      <th>ind_var31</th>
      <th>num_meses_var13_corto_ult3</th>
      <th>num_op_var39_comer_ult1</th>
      <th>var15</th>
      <th>num_var20_0</th>
      <th>saldo_medio_var13_corto_hace2</th>
      <th>ind_var37_cte</th>
      <th>saldo_var13_medio</th>
      <th>ind_var40_0</th>
      <th>saldo_medio_var17_hace3</th>
      <th>num_var22_ult1</th>
      <th>num_var30</th>
      <th>saldo_medio_var44_ult3</th>
      <th>saldo_var6</th>
      <th>saldo_var18</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>saldo_var20</th>
      <th>num_var5_0</th>
      <th>ind_var32_0</th>
      <th>num_op_var41_efect_ult1</th>
      <th>num_var6_0</th>
      <th>num_trasp_var17_out_ult1</th>
      <th>num_op_var41_ult1</th>
      <th>saldo_var26</th>
      <th>ind_var6_0</th>
      <th>num_var20</th>
      <th>num_var18</th>
      <th>imp_reemb_var13_ult1</th>
      <th>ind_var26_cte</th>
      <th>saldo_var17</th>
      <th>ind_var26_0</th>
      <th>imp_venta_var44_ult1</th>
      <th>saldo_medio_var29_hace2</th>
      <th>imp_venta_var44_hace3</th>
      <th>num_var33_0</th>
      <th>saldo_var30</th>
      <th>imp_trasp_var33_out_ult1</th>
      <th>num_var43_recib_ult1</th>
      <th>ind_var29_0</th>
      <th>num_compra_var44_ult1</th>
      <th>delta_imp_aport_var17_1y3</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>ind_var37</th>
      <th>saldo_var24</th>
      <th>saldo_var13_corto</th>
      <th>TARGET</th>
      <th>num_var37</th>
      <th>saldo_var31</th>
      <th>saldo_medio_var44_hace2</th>
      <th>ind_var12_0</th>
      <th>ind_var44</th>
      <th>num_meses_var13_medio_ult3</th>
      <th>num_var12</th>
      <th>ind_var17</th>
      <th>num_meses_var17_ult3</th>
      <th>delta_imp_trasp_var33_out_1y3</th>
      <th>ind_var39_0</th>
      <th>num_var33</th>
      <th>imp_trasp_var33_in_ult1</th>
      <th>num_var22_hace2</th>
      <th>ind_var10cte_ult1</th>
      <th>num_var8</th>
      <th>imp_compra_var44_hace3</th>
      <th>num_sal_var16_ult1</th>
      <th>imp_var43_emit_ult1</th>
      <th>imp_op_var39_efect_ult3</th>
      <th>ind_var5</th>
      <th>saldo_medio_var12_ult3</th>
      <th>num_var1</th>
      <th>num_meses_var33_ult3</th>
      <th>ind_var9_ult1</th>
      <th>saldo_medio_var44_hace3</th>
      <th>ind_var30_0</th>
      <th>ID</th>
      <th>ind_var13_medio</th>
      <th>delta_imp_trasp_var33_in_1y3</th>
      <th>imp_trasp_var17_out_ult1</th>
      <th>delta_imp_reemb_var13_1y3</th>
      <th>saldo_var25</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>num_op_var41_ult3</th>
      <th>var38</th>
      <th>num_trasp_var17_in_hace3</th>
      <th>ind_var13_0</th>
      <th>saldo_var29</th>
      <th>imp_amort_var18_ult1</th>
      <th>saldo_medio_var13_medio_ult3</th>
      <th>saldo_medio_var17_ult3</th>
      <th>num_op_var39_comer_ult3</th>
      <th>num_var22_ult3</th>
      <th>ind_var24</th>
      <th>num_var29_0</th>
      <th>saldo_medio_var17_ult1</th>
      <th>imp_var7_emit_ult1</th>
      <th>ind_var9_cte_ult1</th>
      <th>delta_imp_trasp_var17_out_1y3</th>
      <th>num_op_var40_hace2</th>
      <th>ind_var7_recib_ult1</th>
      <th>num_var13_largo_0</th>
      <th>imp_trans_var37_ult1</th>
      <th>num_var13_medio</th>
      <th>num_venta_var44_hace3</th>
      <th>imp_op_var40_ult1</th>
      <th>saldo_medio_var5_hace2</th>
      <th>saldo_var33</th>
      <th>num_aport_var17_ult1</th>
      <th>num_var14_0</th>
      <th>saldo_medio_var13_corto_ult3</th>
      <th>num_var44</th>
      <th>num_meses_var13_largo_ult3</th>
      <th>num_meses_var12_ult3</th>
      <th>ind_var5_0</th>
      <th>num_reemb_var13_ult1</th>
      <th>num_op_var40_efect_ult3</th>
      <th>num_op_var41_hace3</th>
      <th>delta_num_venta_var44_1y3</th>
      <th>ind_var8</th>
      <th>num_meses_var29_ult3</th>
      <th>imp_reemb_var33_ult1</th>
      <th>num_aport_var33_hace3</th>
      <th>imp_trasp_var33_in_hace3</th>
      <th>saldo_medio_var29_ult1</th>
      <th>ind_var43_recib_ult1</th>
      <th>saldo_medio_var5_ult1</th>
      <th>saldo_var8</th>
      <th>ind_var10_ult1</th>
      <th>ind_var25_0</th>
      <th>num_var34_0</th>
      <th>num_trasp_var11_ult1</th>
      <th>imp_aport_var17_hace3</th>
      <th>saldo_medio_var33_hace3</th>
      <th>ind_var32_cte</th>
      <th>num_var17_0</th>
      <th>num_var35</th>
      <th>num_ent_var16_ult1</th>
      <th>ind_var41_0</th>
      <th>num_var37_0</th>
      <th>ind_var13</th>
      <th>delta_imp_trasp_var17_in_1y3</th>
      <th>ind_var13_corto</th>
      <th>imp_trasp_var17_in_ult1</th>
      <th>delta_num_trasp_var33_out_1y3</th>
      <th>num_var25</th>
      <th>num_op_var39_efect_ult3</th>
      <th>ind_var34_0</th>
      <th>delta_num_trasp_var33_in_1y3</th>
      <th>ind_var43_emit_ult1</th>
      <th>num_var24</th>
      <th>saldo_var13_largo</th>
      <th>imp_aport_var13_hace3</th>
      <th>num_var45_hace3</th>
      <th>num_var40_0</th>
      <th>saldo_medio_var29_hace3</th>
      <th>delta_num_compra_var44_1y3</th>
      <th>ind_var37_0</th>
      <th>imp_aport_var33_ult1</th>
      <th>saldo_medio_var13_medio_ult1</th>
      <th>saldo_medio_var8_ult3</th>
      <th>ind_var25</th>
      <th>imp_op_var41_comer_ult1</th>
      <th>num_op_var40_comer_ult1</th>
      <th>ind_var8_0</th>
      <th>num_op_var39_ult3</th>
      <th>num_aport_var13_hace3</th>
      <th>num_op_var40_hace3</th>
      <th>ind_var44_0</th>
      <th>imp_op_var41_efect_ult1</th>
      <th>saldo_medio_var8_hace2</th>
      <th>saldo_medio_var12_hace2</th>
      <th>num_op_var40_comer_ult3</th>
      <th>ind_var29</th>
      <th>saldo_medio_var8_hace3</th>
      <th>ind_var14</th>
      <th>num_meses_var8_ult3</th>
      <th>num_var5</th>
      <th>num_var25_0</th>
      <th>delta_num_trasp_var17_in_1y3</th>
      <th>var3</th>
      <th>delta_num_aport_var13_1y3</th>
      <th>ind_var1_0</th>
      <th>num_med_var22_ult3</th>
      <th>ind_var7_emit_ult1</th>
      <th>saldo_var5</th>
      <th>imp_op_var39_ult1</th>
      <th>saldo_medio_var13_medio_hace2</th>
      <th>delta_num_reemb_var13_1y3</th>
      <th>ind_var17_0</th>
      <th>num_var39_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>num_op_var41_comer_ult3</th>
      <td>1.000000</td>
      <td>-0.000888</td>
      <td>0.034280</td>
      <td>-0.004987</td>
      <td>0.113940</td>
      <td>0.006362</td>
      <td>0.019320</td>
      <td>0.256917</td>
      <td>0.007249</td>
      <td>0.023682</td>
      <td>0.007450</td>
      <td>0.005217</td>
      <td>0.004580</td>
      <td>-0.032022</td>
      <td>-0.007514</td>
      <td>0.050851</td>
      <td>0.008388</td>
      <td>0.799489</td>
      <td>-0.176631</td>
      <td>0.318759</td>
      <td>-0.000168</td>
      <td>0.004765</td>
      <td>-0.019284</td>
      <td>0.824280</td>
      <td>0.061530</td>
      <td>-0.000844</td>
      <td>0.007450</td>
      <td>0.919436</td>
      <td>-0.000888</td>
      <td>0.023636</td>
      <td>0.003181</td>
      <td>0.097000</td>
      <td>-0.011188</td>
      <td>0.074992</td>
      <td>0.006362</td>
      <td>0.024006</td>
      <td>0.022728</td>
      <td>0.227851</td>
      <td>-0.032856</td>
      <td>-0.003082</td>
      <td>0.006362</td>
      <td>-0.000888</td>
      <td>0.446961</td>
      <td>-0.031608</td>
      <td>0.007450</td>
      <td>0.012677</td>
      <td>0.113915</td>
      <td>0.113940</td>
      <td>0.023697</td>
      <td>-0.009029</td>
      <td>0.296154</td>
      <td>-0.012900</td>
      <td>-0.001538</td>
      <td>0.504227</td>
      <td>0.128069</td>
      <td>-0.008948</td>
      <td>0.424879</td>
      <td>0.014046</td>
      <td>0.003574</td>
      <td>-0.029561</td>
      <td>0.012837</td>
      <td>-0.008796</td>
      <td>-0.002803</td>
      <td>-0.014731</td>
      <td>-0.031385</td>
      <td>-0.000168</td>
      <td>0.804746</td>
      <td>0.015110</td>
      <td>0.517406</td>
      <td>0.004373</td>
      <td>-0.031170</td>
      <td>-0.000478</td>
      <td>-0.011192</td>
      <td>0.092552</td>
      <td>0.279782</td>
      <td>-0.012907</td>
      <td>0.040284</td>
      <td>0.121184</td>
      <td>0.138883</td>
      <td>0.014092</td>
      <td>0.113940</td>
      <td>-0.000525</td>
      <td>0.006362</td>
      <td>0.001420</td>
      <td>-0.000888</td>
      <td>-0.012131</td>
      <td>0.001420</td>
      <td>-0.000119</td>
      <td>-0.002079</td>
      <td>-0.000300</td>
      <td>-0.014780</td>
      <td>0.133813</td>
      <td>0.042808</td>
      <td>-0.007838</td>
      <td>-0.000888</td>
      <td>0.011456</td>
      <td>-0.004992</td>
      <td>0.229078</td>
      <td>0.103758</td>
      <td>-0.004745</td>
      <td>0.000170</td>
      <td>-0.001391</td>
      <td>0.002553</td>
      <td>0.113940</td>
      <td>0.001931</td>
      <td>-0.021684</td>
      <td>0.280080</td>
      <td>-0.001892</td>
      <td>-0.002079</td>
      <td>0.036538</td>
      <td>-0.010562</td>
      <td>-0.002784</td>
      <td>-0.012732</td>
      <td>0.058240</td>
      <td>0.004765</td>
      <td>-0.000168</td>
      <td>0.062980</td>
      <td>0.006325</td>
      <td>-0.001391</td>
      <td>3.917894e-03</td>
      <td>0.016903</td>
      <td>0.026174</td>
      <td>-0.019166</td>
      <td>0.244629</td>
      <td>0.011176</td>
      <td>0.023050</td>
      <td>0.000019</td>
      <td>1.315767e-02</td>
      <td>0.312865</td>
      <td>0.011773</td>
      <td>0.000300</td>
      <td>0.193798</td>
      <td>0.003837</td>
      <td>0.241984</td>
      <td>0.010255</td>
      <td>0.238022</td>
      <td>0.030157</td>
      <td>0.215853</td>
      <td>0.010537</td>
      <td>-0.007716</td>
      <td>0.000921</td>
      <td>0.845178</td>
      <td>0.229078</td>
      <td>0.030730</td>
      <td>0.000921</td>
      <td>-0.033054</td>
      <td>0.253530</td>
      <td>0.112070</td>
      <td>0.000787</td>
      <td>0.019107</td>
      <td>-0.030526</td>
      <td>0.896654</td>
      <td>0.076783</td>
      <td>-0.014780</td>
      <td>-0.026758</td>
      <td>0.305516</td>
      <td>0.001842</td>
      <td>0.138938</td>
      <td>-0.000951</td>
      <td>0.090280</td>
      <td>0.112940</td>
      <td>0.005079</td>
      <td>-0.000417</td>
      <td>0.000100</td>
      <td>0.055908</td>
      <td>-0.002707</td>
      <td>-0.186983</td>
      <td>0.005217</td>
      <td>0.448007</td>
      <td>-0.000879</td>
      <td>-0.001391</td>
      <td>0.855584</td>
      <td>0.196716</td>
      <td>-0.000879</td>
      <td>-0.012732</td>
      <td>0.006362</td>
      <td>-0.001541</td>
      <td>0.252003</td>
      <td>-0.001028</td>
      <td>0.244629</td>
      <td>0.000579</td>
      <td>-0.001231</td>
      <td>0.003185</td>
      <td>0.028551</td>
      <td>-0.026383</td>
      <td>-0.000119</td>
      <td>0.237589</td>
      <td>-0.000879</td>
      <td>0.007022</td>
      <td>-0.003082</td>
      <td>0.788542</td>
      <td>0.303987</td>
      <td>-0.020562</td>
      <td>-0.029932</td>
      <td>0.004403</td>
      <td>0.260970</td>
      <td>0.000265</td>
      <td>0.006014</td>
      <td>0.000813</td>
      <td>0.012213</td>
      <td>0.000921</td>
      <td>0.011052</td>
      <td>0.000444</td>
      <td>0.001881</td>
      <td>-0.000119</td>
      <td>0.085717</td>
      <td>0.026748</td>
      <td>0.002201</td>
      <td>0.154626</td>
      <td>0.337783</td>
      <td>0.235757</td>
      <td>0.014480</td>
      <td>0.031271</td>
      <td>0.010753</td>
      <td>0.253341</td>
      <td>0.057529</td>
      <td>-0.019362</td>
      <td>0.113904</td>
      <td>0.030383</td>
      <td>0.336575</td>
      <td>0.003909</td>
      <td>-0.070797</td>
      <td>-0.004714</td>
      <td>0.000921</td>
      <td>0.005585</td>
      <td>-0.001286</td>
      <td>-0.001605</td>
      <td>0.199408</td>
      <td>0.710507</td>
      <td>0.920593</td>
      <td>0.007422</td>
      <td>-0.000503</td>
      <td>-0.033773</td>
      <td>-0.000417</td>
      <td>0.000296</td>
      <td>0.001723</td>
      <td>-0.000697</td>
      <td>0.957859</td>
      <td>0.218481</td>
      <td>-0.007203</td>
      <td>-0.000879</td>
      <td>-0.000785</td>
      <td>-0.001202</td>
      <td>0.336601</td>
      <td>-0.001391</td>
      <td>0.017265</td>
      <td>0.044527</td>
      <td>-0.012590</td>
      <td>0.005703</td>
      <td>0.000921</td>
      <td>0.002320</td>
      <td>0.048452</td>
      <td>0.008158</td>
      <td>0.011363</td>
      <td>-0.002417</td>
      <td>0.013970</td>
      <td>-0.029470</td>
      <td>0.012213</td>
      <td>-0.008707</td>
      <td>0.011746</td>
      <td>-0.207852</td>
      <td>-0.001605</td>
      <td>0.033962</td>
      <td>0.283049</td>
      <td>0.023697</td>
      <td>0.235757</td>
      <td>0.000158</td>
      <td>-0.000888</td>
      <td>0.017408</td>
      <td>0.005330</td>
      <td>-0.000833</td>
      <td>0.293670</td>
      <td>0.024693</td>
      <td>0.093116</td>
      <td>0.338670</td>
      <td>0.249229</td>
      <td>0.007450</td>
      <td>0.151762</td>
      <td>-0.000820</td>
      <td>0.014706</td>
      <td>0.004481</td>
      <td>-0.000748</td>
      <td>0.426646</td>
      <td>0.161279</td>
      <td>0.080511</td>
      <td>0.260970</td>
      <td>-0.033480</td>
      <td>-0.001391</td>
      <td>-0.031168</td>
      <td>-0.001130</td>
      <td>-0.000119</td>
      <td>0.234160</td>
      <td>0.516663</td>
      <td>0.007450</td>
      <td>0.005585</td>
      <td>0.251087</td>
      <td>-0.007038</td>
      <td>-0.012692</td>
      <td>-0.019348</td>
      <td>0.219636</td>
      <td>0.138693</td>
      <td>0.002190</td>
      <td>0.015110</td>
      <td>0.303987</td>
      <td>0.001935</td>
      <td>0.001842</td>
      <td>0.088700</td>
      <td>0.249229</td>
      <td>0.735967</td>
      <td>0.046491</td>
      <td>0.215814</td>
      <td>0.910808</td>
      <td>-0.020159</td>
      <td>-0.001390</td>
      <td>0.023957</td>
      <td>0.242968</td>
      <td>0.064975</td>
      <td>-0.015966</td>
      <td>0.035061</td>
      <td>-0.000168</td>
      <td>0.032426</td>
      <td>0.058413</td>
      <td>0.235926</td>
      <td>0.058790</td>
      <td>0.234160</td>
      <td>-0.001391</td>
      <td>0.007378</td>
      <td>-0.011188</td>
      <td>0.139030</td>
      <td>0.183616</td>
      <td>-0.001538</td>
      <td>0.026583</td>
      <td>0.500053</td>
      <td>0.001256</td>
      <td>-0.001605</td>
      <td>-0.000599</td>
      <td>0.147360</td>
    </tr>
    <tr>
      <th>delta_imp_reemb_var33_1y3</th>
      <td>-0.000888</td>
      <td>1.000000</td>
      <td>0.132404</td>
      <td>-0.000761</td>
      <td>-0.000222</td>
      <td>-0.000019</td>
      <td>-0.000056</td>
      <td>-0.000598</td>
      <td>-0.000032</td>
      <td>-0.000061</td>
      <td>-0.000019</td>
      <td>-0.000119</td>
      <td>-0.000096</td>
      <td>-0.000613</td>
      <td>-0.000327</td>
      <td>-0.000235</td>
      <td>-0.000061</td>
      <td>-0.000796</td>
      <td>-0.003021</td>
      <td>0.001780</td>
      <td>-0.000019</td>
      <td>-0.000110</td>
      <td>-0.000454</td>
      <td>-0.000801</td>
      <td>-0.000139</td>
      <td>-0.000021</td>
      <td>-0.000019</td>
      <td>-0.000874</td>
      <td>-0.000013</td>
      <td>-0.000157</td>
      <td>-0.000861</td>
      <td>-0.000284</td>
      <td>-0.000254</td>
      <td>-0.005529</td>
      <td>-0.000019</td>
      <td>-0.000048</td>
      <td>-0.000147</td>
      <td>-0.000643</td>
      <td>-0.000816</td>
      <td>-0.000082</td>
      <td>-0.000019</td>
      <td>-0.000013</td>
      <td>-0.000812</td>
      <td>-0.000768</td>
      <td>-0.000019</td>
      <td>0.618167</td>
      <td>-0.000223</td>
      <td>-0.000222</td>
      <td>-0.000085</td>
      <td>-0.000219</td>
      <td>0.002404</td>
      <td>-0.000335</td>
      <td>-0.000023</td>
      <td>-0.000713</td>
      <td>-0.006007</td>
      <td>-0.000217</td>
      <td>-0.000317</td>
      <td>-0.000464</td>
      <td>-0.000017</td>
      <td>-0.000552</td>
      <td>-0.000565</td>
      <td>-0.000531</td>
      <td>-0.000046</td>
      <td>-0.000281</td>
      <td>-0.000764</td>
      <td>-0.000019</td>
      <td>-0.000793</td>
      <td>-0.000110</td>
      <td>-0.000852</td>
      <td>-0.000028</td>
      <td>-0.000754</td>
      <td>-0.000044</td>
      <td>-0.000200</td>
      <td>-0.000205</td>
      <td>-0.000280</td>
      <td>-0.000364</td>
      <td>-0.000163</td>
      <td>-0.008857</td>
      <td>-0.000390</td>
      <td>0.026579</td>
      <td>-0.000222</td>
      <td>-0.000018</td>
      <td>-0.000019</td>
      <td>-0.000013</td>
      <td>1.000000</td>
      <td>-0.000368</td>
      <td>-0.000013</td>
      <td>-0.000013</td>
      <td>-0.000059</td>
      <td>-0.000021</td>
      <td>-0.000219</td>
      <td>-0.005371</td>
      <td>-0.000113</td>
      <td>-0.000214</td>
      <td>1.000000</td>
      <td>0.642858</td>
      <td>-0.000763</td>
      <td>-0.000516</td>
      <td>-0.008027</td>
      <td>-0.012309</td>
      <td>-0.000031</td>
      <td>-0.000026</td>
      <td>-0.000053</td>
      <td>-0.000222</td>
      <td>-0.000016</td>
      <td>-0.009149</td>
      <td>0.001923</td>
      <td>-0.000042</td>
      <td>-0.000059</td>
      <td>-0.000111</td>
      <td>-0.000239</td>
      <td>-0.000125</td>
      <td>-0.000189</td>
      <td>-0.000261</td>
      <td>-0.000110</td>
      <td>-0.000019</td>
      <td>-0.000139</td>
      <td>-0.000068</td>
      <td>-0.000026</td>
      <td>-6.696781e-05</td>
      <td>0.355767</td>
      <td>0.055352</td>
      <td>-0.000448</td>
      <td>-0.000576</td>
      <td>-0.000049</td>
      <td>-0.000088</td>
      <td>-0.000073</td>
      <td>-6.787500e-05</td>
      <td>0.001651</td>
      <td>0.032637</td>
      <td>-0.000112</td>
      <td>-0.001317</td>
      <td>-0.000033</td>
      <td>-0.000465</td>
      <td>-0.000043</td>
      <td>-0.000579</td>
      <td>-0.000194</td>
      <td>-0.000668</td>
      <td>-0.000792</td>
      <td>-0.000226</td>
      <td>-0.000019</td>
      <td>-0.000952</td>
      <td>-0.000516</td>
      <td>0.144293</td>
      <td>-0.000019</td>
      <td>-0.000813</td>
      <td>-0.000432</td>
      <td>-0.000300</td>
      <td>-0.000016</td>
      <td>0.059759</td>
      <td>-0.000738</td>
      <td>-0.000872</td>
      <td>0.011698</td>
      <td>-0.000219</td>
      <td>-0.000501</td>
      <td>-0.001012</td>
      <td>-0.000016</td>
      <td>-0.000390</td>
      <td>-0.000015</td>
      <td>-0.000966</td>
      <td>-0.005261</td>
      <td>-0.000072</td>
      <td>-0.000018</td>
      <td>-0.000014</td>
      <td>-0.000153</td>
      <td>-0.000040</td>
      <td>-0.015989</td>
      <td>-0.000119</td>
      <td>-0.000813</td>
      <td>-0.000037</td>
      <td>-0.000026</td>
      <td>-0.000952</td>
      <td>-0.000373</td>
      <td>-0.000037</td>
      <td>-0.000189</td>
      <td>-0.000019</td>
      <td>-0.000059</td>
      <td>-0.000611</td>
      <td>-0.000029</td>
      <td>-0.000576</td>
      <td>-0.000026</td>
      <td>-0.000018</td>
      <td>-0.000017</td>
      <td>0.107155</td>
      <td>-0.000787</td>
      <td>-0.000013</td>
      <td>-0.000831</td>
      <td>-0.000037</td>
      <td>-0.000087</td>
      <td>-0.000082</td>
      <td>-0.000794</td>
      <td>-0.000958</td>
      <td>-0.000448</td>
      <td>-0.000555</td>
      <td>-0.000736</td>
      <td>-0.000678</td>
      <td>0.022055</td>
      <td>-0.000057</td>
      <td>-0.000976</td>
      <td>-0.000150</td>
      <td>-0.000019</td>
      <td>-0.000785</td>
      <td>-0.000138</td>
      <td>-0.000138</td>
      <td>-0.000013</td>
      <td>-0.009857</td>
      <td>0.125938</td>
      <td>-0.000021</td>
      <td>-0.001365</td>
      <td>-0.001156</td>
      <td>-0.000622</td>
      <td>-0.000044</td>
      <td>-0.000115</td>
      <td>-0.000217</td>
      <td>-0.000433</td>
      <td>-0.005096</td>
      <td>-0.000450</td>
      <td>-0.000222</td>
      <td>0.115999</td>
      <td>-0.001112</td>
      <td>-0.000046</td>
      <td>-0.053873</td>
      <td>-0.002162</td>
      <td>-0.000019</td>
      <td>-0.000029</td>
      <td>-0.000019</td>
      <td>-0.000081</td>
      <td>-0.000363</td>
      <td>-0.000773</td>
      <td>-0.000978</td>
      <td>0.002174</td>
      <td>-0.000018</td>
      <td>-0.000852</td>
      <td>-0.000018</td>
      <td>-0.000015</td>
      <td>-0.000017</td>
      <td>-0.000030</td>
      <td>-0.000877</td>
      <td>-0.001779</td>
      <td>-0.000720</td>
      <td>-0.000037</td>
      <td>-0.000032</td>
      <td>-0.000018</td>
      <td>-0.001188</td>
      <td>-0.000026</td>
      <td>-0.000082</td>
      <td>-0.000189</td>
      <td>-0.000334</td>
      <td>-0.000276</td>
      <td>-0.000019</td>
      <td>-0.000021</td>
      <td>-0.000120</td>
      <td>-0.000471</td>
      <td>0.645828</td>
      <td>-0.000071</td>
      <td>-0.000432</td>
      <td>-0.000547</td>
      <td>-0.000150</td>
      <td>-0.000296</td>
      <td>-0.000759</td>
      <td>-0.017327</td>
      <td>-0.000081</td>
      <td>-0.000066</td>
      <td>-0.000280</td>
      <td>-0.000085</td>
      <td>-0.000622</td>
      <td>-0.000028</td>
      <td>1.000000</td>
      <td>-0.000055</td>
      <td>-0.000031</td>
      <td>-0.000018</td>
      <td>-0.001398</td>
      <td>-0.000406</td>
      <td>-0.000204</td>
      <td>-0.001076</td>
      <td>-0.000564</td>
      <td>-0.000019</td>
      <td>-0.000373</td>
      <td>-0.000016</td>
      <td>-0.000043</td>
      <td>-0.000126</td>
      <td>-0.000119</td>
      <td>-0.000379</td>
      <td>-0.000685</td>
      <td>-0.009789</td>
      <td>-0.000678</td>
      <td>-0.000840</td>
      <td>-0.000026</td>
      <td>-0.000754</td>
      <td>-0.000018</td>
      <td>-0.000013</td>
      <td>-0.000505</td>
      <td>-0.000851</td>
      <td>-0.000019</td>
      <td>-0.000029</td>
      <td>-0.000969</td>
      <td>-0.000719</td>
      <td>-0.000271</td>
      <td>-0.000404</td>
      <td>-0.000311</td>
      <td>-0.000390</td>
      <td>-0.000013</td>
      <td>-0.000110</td>
      <td>-0.000958</td>
      <td>-0.000029</td>
      <td>-0.000016</td>
      <td>-0.000206</td>
      <td>-0.000564</td>
      <td>-0.000781</td>
      <td>-0.000130</td>
      <td>-0.000668</td>
      <td>-0.000978</td>
      <td>-0.000498</td>
      <td>-0.000021</td>
      <td>-0.000157</td>
      <td>-0.000465</td>
      <td>-0.000143</td>
      <td>-0.000384</td>
      <td>-0.000122</td>
      <td>-0.000019</td>
      <td>-0.000066</td>
      <td>-0.000265</td>
      <td>-0.000581</td>
      <td>-0.005064</td>
      <td>-0.000505</td>
      <td>-0.000026</td>
      <td>0.000151</td>
      <td>-0.000254</td>
      <td>-0.000390</td>
      <td>-0.001257</td>
      <td>-0.000023</td>
      <td>-0.000379</td>
      <td>-0.000714</td>
      <td>-0.000018</td>
      <td>-0.000081</td>
      <td>-0.000154</td>
      <td>-0.008676</td>
    </tr>
    <tr>
      <th>ind_var33_0</th>
      <td>0.034280</td>
      <td>0.132404</td>
      <td>1.000000</td>
      <td>0.003730</td>
      <td>0.006217</td>
      <td>-0.000141</td>
      <td>-0.000426</td>
      <td>0.013463</td>
      <td>0.324334</td>
      <td>-0.000463</td>
      <td>-0.000141</td>
      <td>-0.000900</td>
      <td>-0.000724</td>
      <td>-0.000798</td>
      <td>-0.001368</td>
      <td>0.013091</td>
      <td>0.004998</td>
      <td>0.009598</td>
      <td>-0.017459</td>
      <td>0.051369</td>
      <td>-0.000141</td>
      <td>-0.000828</td>
      <td>-0.000872</td>
      <td>0.047467</td>
      <td>0.007049</td>
      <td>-0.000160</td>
      <td>-0.000141</td>
      <td>0.027972</td>
      <td>-0.000099</td>
      <td>0.020767</td>
      <td>0.004836</td>
      <td>0.014040</td>
      <td>0.004990</td>
      <td>-0.017707</td>
      <td>-0.000141</td>
      <td>0.480466</td>
      <td>0.020690</td>
      <td>0.039388</td>
      <td>0.005438</td>
      <td>0.020602</td>
      <td>-0.000141</td>
      <td>-0.000099</td>
      <td>0.009511</td>
      <td>0.006052</td>
      <td>-0.000141</td>
      <td>0.595560</td>
      <td>0.006166</td>
      <td>0.006217</td>
      <td>0.019807</td>
      <td>-0.001656</td>
      <td>0.046996</td>
      <td>0.001249</td>
      <td>-0.000172</td>
      <td>0.020748</td>
      <td>-0.017129</td>
      <td>-0.001638</td>
      <td>0.037235</td>
      <td>-0.000814</td>
      <td>-0.000130</td>
      <td>-0.001851</td>
      <td>0.005224</td>
      <td>-0.000866</td>
      <td>0.000216</td>
      <td>-0.002121</td>
      <td>0.005881</td>
      <td>-0.000141</td>
      <td>0.009762</td>
      <td>0.015013</td>
      <td>0.015078</td>
      <td>0.280879</td>
      <td>0.006343</td>
      <td>-0.000329</td>
      <td>-0.000932</td>
      <td>0.000665</td>
      <td>-0.002117</td>
      <td>0.002078</td>
      <td>0.005069</td>
      <td>-0.026460</td>
      <td>0.015055</td>
      <td>0.246838</td>
      <td>0.006217</td>
      <td>-0.000132</td>
      <td>-0.000141</td>
      <td>0.132404</td>
      <td>0.132404</td>
      <td>0.002014</td>
      <td>0.132404</td>
      <td>0.132404</td>
      <td>-0.000444</td>
      <td>0.000842</td>
      <td>-0.001654</td>
      <td>-0.010726</td>
      <td>0.017291</td>
      <td>-0.001615</td>
      <td>0.132404</td>
      <td>0.569382</td>
      <td>0.003781</td>
      <td>0.009866</td>
      <td>-0.023232</td>
      <td>-0.041046</td>
      <td>-0.000110</td>
      <td>-0.000199</td>
      <td>-0.000402</td>
      <td>0.006217</td>
      <td>-0.000121</td>
      <td>-0.025957</td>
      <td>0.044143</td>
      <td>-0.000315</td>
      <td>-0.000444</td>
      <td>0.012124</td>
      <td>0.003846</td>
      <td>-0.000945</td>
      <td>-0.001424</td>
      <td>0.010886</td>
      <td>-0.000828</td>
      <td>-0.000141</td>
      <td>0.024138</td>
      <td>0.001283</td>
      <td>-0.000199</td>
      <td>8.382980e-04</td>
      <td>0.635479</td>
      <td>0.418050</td>
      <td>-0.000718</td>
      <td>0.014247</td>
      <td>-0.000370</td>
      <td>0.000523</td>
      <td>-0.000096</td>
      <td>-5.126339e-04</td>
      <td>0.051380</td>
      <td>0.246492</td>
      <td>0.014136</td>
      <td>0.020096</td>
      <td>0.334972</td>
      <td>0.017599</td>
      <td>-0.000322</td>
      <td>0.031445</td>
      <td>0.001929</td>
      <td>0.008430</td>
      <td>0.007864</td>
      <td>0.003290</td>
      <td>-0.000141</td>
      <td>0.016814</td>
      <td>0.009866</td>
      <td>0.917609</td>
      <td>-0.000141</td>
      <td>0.005992</td>
      <td>0.018906</td>
      <td>0.009817</td>
      <td>-0.000037</td>
      <td>0.379810</td>
      <td>0.003324</td>
      <td>0.033204</td>
      <td>0.036048</td>
      <td>-0.001654</td>
      <td>-0.001853</td>
      <td>0.018332</td>
      <td>-0.000124</td>
      <td>0.015149</td>
      <td>-0.000116</td>
      <td>0.031753</td>
      <td>-0.006384</td>
      <td>0.001500</td>
      <td>-0.000137</td>
      <td>-0.000109</td>
      <td>0.025507</td>
      <td>-0.000303</td>
      <td>-0.072444</td>
      <td>-0.000900</td>
      <td>0.009578</td>
      <td>-0.000281</td>
      <td>-0.000199</td>
      <td>0.014261</td>
      <td>0.011819</td>
      <td>-0.000281</td>
      <td>-0.001424</td>
      <td>-0.000141</td>
      <td>-0.000442</td>
      <td>0.013002</td>
      <td>0.001551</td>
      <td>0.014247</td>
      <td>-0.000053</td>
      <td>-0.000140</td>
      <td>-0.000128</td>
      <td>0.922980</td>
      <td>-0.001162</td>
      <td>0.132404</td>
      <td>0.015203</td>
      <td>-0.000281</td>
      <td>0.003916</td>
      <td>0.020602</td>
      <td>0.051681</td>
      <td>0.020003</td>
      <td>-0.000890</td>
      <td>-0.001895</td>
      <td>-0.005560</td>
      <td>0.030906</td>
      <td>0.021319</td>
      <td>0.001754</td>
      <td>0.006035</td>
      <td>0.010547</td>
      <td>-0.000141</td>
      <td>0.007642</td>
      <td>0.024243</td>
      <td>0.017523</td>
      <td>0.132404</td>
      <td>-0.032925</td>
      <td>0.884367</td>
      <td>0.215109</td>
      <td>0.028550</td>
      <td>0.017855</td>
      <td>0.009717</td>
      <td>-0.000332</td>
      <td>-0.000872</td>
      <td>0.003033</td>
      <td>0.018820</td>
      <td>-0.017125</td>
      <td>-0.000520</td>
      <td>0.006129</td>
      <td>0.883785</td>
      <td>0.019041</td>
      <td>-0.000344</td>
      <td>-0.213276</td>
      <td>-0.000967</td>
      <td>-0.000141</td>
      <td>0.296073</td>
      <td>-0.000147</td>
      <td>-0.000613</td>
      <td>0.012155</td>
      <td>0.041240</td>
      <td>0.013270</td>
      <td>0.008020</td>
      <td>-0.000133</td>
      <td>0.006526</td>
      <td>-0.000137</td>
      <td>-0.000110</td>
      <td>-0.000128</td>
      <td>0.001902</td>
      <td>0.039568</td>
      <td>0.037206</td>
      <td>0.004633</td>
      <td>-0.000281</td>
      <td>0.002404</td>
      <td>-0.000135</td>
      <td>0.018650</td>
      <td>-0.000199</td>
      <td>-0.000618</td>
      <td>0.007842</td>
      <td>0.001035</td>
      <td>-0.000889</td>
      <td>-0.000141</td>
      <td>-0.000162</td>
      <td>0.004895</td>
      <td>-0.000850</td>
      <td>0.573433</td>
      <td>0.016032</td>
      <td>0.003821</td>
      <td>-0.001588</td>
      <td>0.010547</td>
      <td>-0.002237</td>
      <td>0.007078</td>
      <td>-0.078143</td>
      <td>-0.000613</td>
      <td>-0.000498</td>
      <td>-0.002117</td>
      <td>0.019807</td>
      <td>0.009717</td>
      <td>-0.000212</td>
      <td>0.132404</td>
      <td>0.558370</td>
      <td>0.314513</td>
      <td>-0.000134</td>
      <td>0.010927</td>
      <td>-0.000157</td>
      <td>-0.000739</td>
      <td>0.018315</td>
      <td>0.014718</td>
      <td>-0.000141</td>
      <td>0.032850</td>
      <td>-0.000122</td>
      <td>0.437315</td>
      <td>-0.000954</td>
      <td>0.011035</td>
      <td>0.039867</td>
      <td>0.012213</td>
      <td>-0.034102</td>
      <td>0.030906</td>
      <td>0.006784</td>
      <td>-0.000199</td>
      <td>0.006353</td>
      <td>-0.000135</td>
      <td>0.132404</td>
      <td>0.010321</td>
      <td>0.014998</td>
      <td>-0.000141</td>
      <td>0.296073</td>
      <td>0.021598</td>
      <td>0.004620</td>
      <td>0.001557</td>
      <td>-0.003053</td>
      <td>0.037785</td>
      <td>0.015120</td>
      <td>-0.000099</td>
      <td>0.015013</td>
      <td>0.020003</td>
      <td>0.291829</td>
      <td>-0.000124</td>
      <td>0.001232</td>
      <td>0.014718</td>
      <td>0.036748</td>
      <td>0.027308</td>
      <td>0.008437</td>
      <td>0.014856</td>
      <td>-0.003763</td>
      <td>-0.000155</td>
      <td>0.020993</td>
      <td>0.017738</td>
      <td>0.000373</td>
      <td>0.000373</td>
      <td>0.024627</td>
      <td>-0.000141</td>
      <td>-0.000501</td>
      <td>0.011236</td>
      <td>0.005660</td>
      <td>-0.017101</td>
      <td>0.010321</td>
      <td>-0.000199</td>
      <td>0.001108</td>
      <td>0.004990</td>
      <td>0.015113</td>
      <td>0.036858</td>
      <td>-0.000172</td>
      <td>0.000130</td>
      <td>0.020964</td>
      <td>-0.000139</td>
      <td>-0.000613</td>
      <td>0.021497</td>
      <td>-0.022495</td>
    </tr>
    <tr>
      <th>num_var24_0</th>
      <td>-0.004987</td>
      <td>-0.000761</td>
      <td>0.003730</td>
      <td>1.000000</td>
      <td>0.030838</td>
      <td>-0.001076</td>
      <td>0.003983</td>
      <td>0.017592</td>
      <td>-0.001864</td>
      <td>0.001744</td>
      <td>-0.001076</td>
      <td>0.008912</td>
      <td>0.007602</td>
      <td>0.024488</td>
      <td>0.119698</td>
      <td>0.150921</td>
      <td>0.004590</td>
      <td>-0.002854</td>
      <td>-0.164818</td>
      <td>0.182590</td>
      <td>-0.001076</td>
      <td>0.004800</td>
      <td>0.584836</td>
      <td>0.001368</td>
      <td>0.021327</td>
      <td>-0.001228</td>
      <td>-0.001076</td>
      <td>-0.005591</td>
      <td>-0.000761</td>
      <td>0.014624</td>
      <td>0.700306</td>
      <td>0.013859</td>
      <td>0.012352</td>
      <td>0.017914</td>
      <td>-0.001076</td>
      <td>-0.002761</td>
      <td>0.010610</td>
      <td>0.153875</td>
      <td>0.049343</td>
      <td>0.023888</td>
      <td>-0.001076</td>
      <td>-0.000761</td>
      <td>-0.003043</td>
      <td>0.045784</td>
      <td>-0.001076</td>
      <td>0.002523</td>
      <td>0.030541</td>
      <td>0.030838</td>
      <td>0.008867</td>
      <td>0.017678</td>
      <td>0.173754</td>
      <td>0.021937</td>
      <td>-0.001318</td>
      <td>0.007963</td>
      <td>0.123310</td>
      <td>0.017771</td>
      <td>0.189529</td>
      <td>0.113314</td>
      <td>-0.000995</td>
      <td>0.017321</td>
      <td>0.037339</td>
      <td>0.578591</td>
      <td>0.015623</td>
      <td>0.011316</td>
      <td>0.044799</td>
      <td>-0.001076</td>
      <td>-0.004134</td>
      <td>0.015013</td>
      <td>-0.002832</td>
      <td>0.004708</td>
      <td>0.026562</td>
      <td>0.003403</td>
      <td>0.011193</td>
      <td>0.010352</td>
      <td>-0.006396</td>
      <td>0.021290</td>
      <td>0.031376</td>
      <td>0.068707</td>
      <td>0.094042</td>
      <td>0.016411</td>
      <td>0.030838</td>
      <td>-0.001013</td>
      <td>-0.001076</td>
      <td>-0.000761</td>
      <td>-0.000761</td>
      <td>0.022694</td>
      <td>-0.000761</td>
      <td>-0.000761</td>
      <td>0.008594</td>
      <td>-0.000878</td>
      <td>-0.012665</td>
      <td>0.311172</td>
      <td>0.023053</td>
      <td>0.011648</td>
      <td>-0.000761</td>
      <td>0.002594</td>
      <td>0.997448</td>
      <td>0.015666</td>
      <td>0.076722</td>
      <td>0.663563</td>
      <td>0.001156</td>
      <td>0.007419</td>
      <td>-0.001223</td>
      <td>0.030838</td>
      <td>-0.000924</td>
      <td>0.496229</td>
      <td>0.120925</td>
      <td>0.006655</td>
      <td>0.008594</td>
      <td>0.019970</td>
      <td>0.309717</td>
      <td>0.001678</td>
      <td>-0.010910</td>
      <td>0.016133</td>
      <td>0.004800</td>
      <td>-0.001076</td>
      <td>0.019886</td>
      <td>0.000646</td>
      <td>-0.001522</td>
      <td>-2.132363e-07</td>
      <td>0.000035</td>
      <td>0.022038</td>
      <td>0.578271</td>
      <td>0.018532</td>
      <td>-0.001691</td>
      <td>-0.003670</td>
      <td>0.045544</td>
      <td>-1.291195e-07</td>
      <td>0.177084</td>
      <td>0.018001</td>
      <td>0.014436</td>
      <td>0.236958</td>
      <td>-0.001925</td>
      <td>0.008294</td>
      <td>0.002510</td>
      <td>0.007760</td>
      <td>0.044548</td>
      <td>0.021002</td>
      <td>0.855530</td>
      <td>0.011211</td>
      <td>-0.001076</td>
      <td>-0.000968</td>
      <td>0.015666</td>
      <td>0.002472</td>
      <td>-0.001076</td>
      <td>0.034015</td>
      <td>0.004497</td>
      <td>0.086210</td>
      <td>-0.000623</td>
      <td>0.019442</td>
      <td>0.033890</td>
      <td>-0.001549</td>
      <td>0.250622</td>
      <td>-0.012665</td>
      <td>0.014282</td>
      <td>0.002033</td>
      <td>-0.000948</td>
      <td>0.094657</td>
      <td>-0.000888</td>
      <td>0.083991</td>
      <td>0.298283</td>
      <td>0.000710</td>
      <td>-0.001047</td>
      <td>-0.000833</td>
      <td>0.015950</td>
      <td>-0.002320</td>
      <td>-0.012371</td>
      <td>0.008912</td>
      <td>-0.003137</td>
      <td>-0.002152</td>
      <td>0.007419</td>
      <td>-0.004858</td>
      <td>0.029377</td>
      <td>-0.002152</td>
      <td>-0.010910</td>
      <td>-0.001076</td>
      <td>0.017350</td>
      <td>0.018956</td>
      <td>0.001099</td>
      <td>0.018532</td>
      <td>0.001441</td>
      <td>-0.001069</td>
      <td>-0.000979</td>
      <td>0.004284</td>
      <td>0.466178</td>
      <td>-0.000761</td>
      <td>0.160087</td>
      <td>-0.002152</td>
      <td>0.008556</td>
      <td>0.023888</td>
      <td>0.005771</td>
      <td>0.001808</td>
      <td>0.586832</td>
      <td>0.016696</td>
      <td>-0.030276</td>
      <td>0.008110</td>
      <td>0.001188</td>
      <td>-0.000190</td>
      <td>0.779688</td>
      <td>0.011835</td>
      <td>-0.001076</td>
      <td>0.850988</td>
      <td>0.017606</td>
      <td>0.010378</td>
      <td>-0.000761</td>
      <td>0.070596</td>
      <td>0.003932</td>
      <td>0.014802</td>
      <td>0.218045</td>
      <td>0.035172</td>
      <td>0.024315</td>
      <td>-0.001893</td>
      <td>0.014667</td>
      <td>0.205177</td>
      <td>0.004633</td>
      <td>-0.062363</td>
      <td>0.580674</td>
      <td>0.030337</td>
      <td>0.000110</td>
      <td>0.030104</td>
      <td>-0.002385</td>
      <td>0.014125</td>
      <td>-0.005645</td>
      <td>-0.001076</td>
      <td>0.006296</td>
      <td>-0.000544</td>
      <td>0.018520</td>
      <td>0.028574</td>
      <td>0.007557</td>
      <td>-0.005375</td>
      <td>0.010338</td>
      <td>-0.001021</td>
      <td>0.050996</td>
      <td>-0.001047</td>
      <td>-0.000846</td>
      <td>-0.000984</td>
      <td>0.000831</td>
      <td>-0.001703</td>
      <td>0.274300</td>
      <td>0.941475</td>
      <td>-0.002152</td>
      <td>0.000626</td>
      <td>-0.001030</td>
      <td>0.054298</td>
      <td>0.007419</td>
      <td>0.010225</td>
      <td>0.035363</td>
      <td>0.022466</td>
      <td>0.221581</td>
      <td>-0.001076</td>
      <td>-0.001243</td>
      <td>0.020919</td>
      <td>0.168527</td>
      <td>0.003460</td>
      <td>0.010441</td>
      <td>0.028227</td>
      <td>0.016967</td>
      <td>0.011835</td>
      <td>0.015431</td>
      <td>0.831551</td>
      <td>-0.012681</td>
      <td>0.018520</td>
      <td>0.005818</td>
      <td>-0.006289</td>
      <td>0.008867</td>
      <td>0.024315</td>
      <td>-0.001627</td>
      <td>-0.000761</td>
      <td>-0.003209</td>
      <td>-0.001808</td>
      <td>-0.001023</td>
      <td>0.179697</td>
      <td>0.064629</td>
      <td>0.003437</td>
      <td>0.006512</td>
      <td>0.017302</td>
      <td>-0.001076</td>
      <td>0.234195</td>
      <td>-0.000849</td>
      <td>-0.002513</td>
      <td>0.009486</td>
      <td>0.011912</td>
      <td>0.185951</td>
      <td>0.009447</td>
      <td>0.071171</td>
      <td>0.008110</td>
      <td>0.033796</td>
      <td>-0.001522</td>
      <td>0.026616</td>
      <td>-0.001036</td>
      <td>-0.000761</td>
      <td>0.014993</td>
      <td>-0.002594</td>
      <td>-0.001076</td>
      <td>0.006296</td>
      <td>0.192920</td>
      <td>0.941980</td>
      <td>0.019797</td>
      <td>0.012249</td>
      <td>0.174727</td>
      <td>0.094467</td>
      <td>-0.000761</td>
      <td>0.015013</td>
      <td>0.001808</td>
      <td>-0.001677</td>
      <td>-0.000948</td>
      <td>0.017980</td>
      <td>0.017302</td>
      <td>0.002227</td>
      <td>0.016775</td>
      <td>0.021030</td>
      <td>-0.002312</td>
      <td>0.018019</td>
      <td>-0.001191</td>
      <td>0.014840</td>
      <td>0.008447</td>
      <td>0.024883</td>
      <td>0.498385</td>
      <td>0.010519</td>
      <td>-0.001076</td>
      <td>0.010440</td>
      <td>0.014157</td>
      <td>0.025483</td>
      <td>-0.061931</td>
      <td>0.014993</td>
      <td>-0.001522</td>
      <td>0.008202</td>
      <td>0.012352</td>
      <td>0.094418</td>
      <td>0.251823</td>
      <td>-0.001318</td>
      <td>0.027375</td>
      <td>0.010591</td>
      <td>-0.001062</td>
      <td>0.018520</td>
      <td>0.017080</td>
      <td>0.091823</td>
    </tr>
    <tr>
      <th>num_var39</th>
      <td>0.113940</td>
      <td>-0.000222</td>
      <td>0.006217</td>
      <td>0.030838</td>
      <td>1.000000</td>
      <td>0.041799</td>
      <td>0.246384</td>
      <td>0.097658</td>
      <td>-0.000543</td>
      <td>0.267553</td>
      <td>-0.000314</td>
      <td>0.011152</td>
      <td>0.023764</td>
      <td>-0.001240</td>
      <td>0.013072</td>
      <td>0.059519</td>
      <td>-0.001022</td>
      <td>0.138382</td>
      <td>-0.033685</td>
      <td>0.102272</td>
      <td>-0.000314</td>
      <td>0.012068</td>
      <td>0.006910</td>
      <td>0.133298</td>
      <td>0.627076</td>
      <td>-0.000358</td>
      <td>-0.000314</td>
      <td>0.102952</td>
      <td>-0.000222</td>
      <td>0.002289</td>
      <td>0.031352</td>
      <td>0.049434</td>
      <td>0.005036</td>
      <td>-0.024168</td>
      <td>0.041799</td>
      <td>-0.000805</td>
      <td>0.002419</td>
      <td>0.102045</td>
      <td>0.005368</td>
      <td>0.017693</td>
      <td>0.041799</td>
      <td>-0.000222</td>
      <td>0.114263</td>
      <td>0.004101</td>
      <td>-0.000314</td>
      <td>-0.000997</td>
      <td>0.994722</td>
      <td>1.000000</td>
      <td>-0.001437</td>
      <td>-0.001837</td>
      <td>0.093035</td>
      <td>0.002847</td>
      <td>-0.000384</td>
      <td>0.126775</td>
      <td>-0.000191</td>
      <td>-0.001780</td>
      <td>0.153296</td>
      <td>0.028938</td>
      <td>-0.000290</td>
      <td>0.000161</td>
      <td>0.020336</td>
      <td>0.012570</td>
      <td>0.009907</td>
      <td>0.008617</td>
      <td>0.003882</td>
      <td>-0.000314</td>
      <td>0.098586</td>
      <td>-0.001856</td>
      <td>0.116111</td>
      <td>-0.000470</td>
      <td>0.004605</td>
      <td>0.005843</td>
      <td>-0.001667</td>
      <td>0.039460</td>
      <td>0.033253</td>
      <td>0.004713</td>
      <td>0.028404</td>
      <td>-0.009163</td>
      <td>0.567850</td>
      <td>0.016098</td>
      <td>1.000000</td>
      <td>-0.000295</td>
      <td>0.041799</td>
      <td>-0.000222</td>
      <td>-0.000222</td>
      <td>0.004569</td>
      <td>-0.000222</td>
      <td>-0.000222</td>
      <td>0.012327</td>
      <td>-0.000018</td>
      <td>-0.003690</td>
      <td>0.022831</td>
      <td>0.507665</td>
      <td>-0.001766</td>
      <td>-0.000222</td>
      <td>-0.000953</td>
      <td>0.031107</td>
      <td>0.080994</td>
      <td>0.032188</td>
      <td>-0.018022</td>
      <td>-0.000524</td>
      <td>-0.000443</td>
      <td>0.023808</td>
      <td>1.000000</td>
      <td>-0.000269</td>
      <td>-0.009734</td>
      <td>0.091506</td>
      <td>0.022858</td>
      <td>0.012327</td>
      <td>0.485990</td>
      <td>0.005079</td>
      <td>-0.001408</td>
      <td>-0.003179</td>
      <td>0.036054</td>
      <td>0.012068</td>
      <td>-0.000314</td>
      <td>0.517071</td>
      <td>-0.000706</td>
      <td>-0.000443</td>
      <td>-7.970245e-04</td>
      <td>-0.001064</td>
      <td>0.012548</td>
      <td>0.006017</td>
      <td>0.087820</td>
      <td>0.214381</td>
      <td>0.023060</td>
      <td>0.004017</td>
      <td>-1.143958e-03</td>
      <td>0.101516</td>
      <td>0.018535</td>
      <td>0.020561</td>
      <td>0.078282</td>
      <td>-0.000561</td>
      <td>0.109244</td>
      <td>0.023299</td>
      <td>0.082122</td>
      <td>0.014127</td>
      <td>0.089284</td>
      <td>0.041615</td>
      <td>0.009664</td>
      <td>0.041799</td>
      <td>0.191722</td>
      <td>0.080994</td>
      <td>-0.001537</td>
      <td>0.041799</td>
      <td>0.006292</td>
      <td>0.096328</td>
      <td>0.452853</td>
      <td>0.012212</td>
      <td>0.010578</td>
      <td>0.005335</td>
      <td>0.201180</td>
      <td>0.054294</td>
      <td>-0.003690</td>
      <td>0.001691</td>
      <td>0.077182</td>
      <td>0.056768</td>
      <td>0.568787</td>
      <td>-0.000139</td>
      <td>0.076092</td>
      <td>0.024147</td>
      <td>-0.000844</td>
      <td>-0.000305</td>
      <td>0.005683</td>
      <td>0.478277</td>
      <td>-0.000676</td>
      <td>-0.141158</td>
      <td>0.011152</td>
      <td>0.102545</td>
      <td>-0.000627</td>
      <td>-0.000443</td>
      <td>0.110496</td>
      <td>0.105341</td>
      <td>-0.000627</td>
      <td>-0.003179</td>
      <td>0.041799</td>
      <td>-0.000987</td>
      <td>0.096585</td>
      <td>0.001026</td>
      <td>0.087820</td>
      <td>-0.000441</td>
      <td>-0.000311</td>
      <td>-0.000285</td>
      <td>0.004841</td>
      <td>0.009048</td>
      <td>-0.000222</td>
      <td>0.044282</td>
      <td>-0.000627</td>
      <td>-0.001461</td>
      <td>0.017693</td>
      <td>0.259577</td>
      <td>0.080049</td>
      <td>0.005600</td>
      <td>-0.000093</td>
      <td>0.009753</td>
      <td>0.090933</td>
      <td>0.000788</td>
      <td>-0.000802</td>
      <td>0.035200</td>
      <td>0.002728</td>
      <td>0.041799</td>
      <td>0.041673</td>
      <td>0.014720</td>
      <td>0.014361</td>
      <td>-0.000222</td>
      <td>0.022492</td>
      <td>-0.001481</td>
      <td>-0.000360</td>
      <td>0.080476</td>
      <td>0.191855</td>
      <td>0.097075</td>
      <td>-0.000741</td>
      <td>0.044056</td>
      <td>0.019355</td>
      <td>0.105409</td>
      <td>-0.028276</td>
      <td>0.007158</td>
      <td>0.993036</td>
      <td>-0.001480</td>
      <td>0.199392</td>
      <td>-0.000769</td>
      <td>-0.231159</td>
      <td>-0.007844</td>
      <td>0.041799</td>
      <td>-0.000496</td>
      <td>-0.000327</td>
      <td>-0.001367</td>
      <td>0.103068</td>
      <td>0.256795</td>
      <td>0.115485</td>
      <td>0.003698</td>
      <td>-0.000297</td>
      <td>0.006031</td>
      <td>-0.000305</td>
      <td>0.006850</td>
      <td>0.055264</td>
      <td>0.000646</td>
      <td>0.205555</td>
      <td>0.111704</td>
      <td>0.033125</td>
      <td>-0.000627</td>
      <td>0.000578</td>
      <td>-0.000300</td>
      <td>0.186647</td>
      <td>-0.000443</td>
      <td>0.335033</td>
      <td>0.021812</td>
      <td>0.002366</td>
      <td>0.010294</td>
      <td>0.041799</td>
      <td>-0.000362</td>
      <td>0.539313</td>
      <td>0.026913</td>
      <td>-0.000960</td>
      <td>0.006254</td>
      <td>0.016065</td>
      <td>0.000131</td>
      <td>0.002728</td>
      <td>0.006158</td>
      <td>0.039029</td>
      <td>-0.153082</td>
      <td>-0.001367</td>
      <td>0.287506</td>
      <td>0.021404</td>
      <td>-0.001437</td>
      <td>0.097075</td>
      <td>-0.000474</td>
      <td>-0.000222</td>
      <td>-0.000935</td>
      <td>-0.000527</td>
      <td>-0.000298</td>
      <td>0.059485</td>
      <td>0.024571</td>
      <td>0.040908</td>
      <td>0.206074</td>
      <td>0.088593</td>
      <td>-0.000314</td>
      <td>0.068892</td>
      <td>-0.000050</td>
      <td>-0.000732</td>
      <td>0.010298</td>
      <td>0.015874</td>
      <td>0.153575</td>
      <td>0.020364</td>
      <td>-0.025748</td>
      <td>0.090933</td>
      <td>0.006497</td>
      <td>-0.000443</td>
      <td>0.004617</td>
      <td>-0.000302</td>
      <td>-0.000222</td>
      <td>0.080421</td>
      <td>0.126860</td>
      <td>-0.000314</td>
      <td>-0.000496</td>
      <td>0.109301</td>
      <td>0.033060</td>
      <td>-0.002551</td>
      <td>0.003420</td>
      <td>0.070955</td>
      <td>0.567795</td>
      <td>-0.000222</td>
      <td>-0.001856</td>
      <td>0.080049</td>
      <td>-0.000489</td>
      <td>0.056768</td>
      <td>0.042507</td>
      <td>0.088593</td>
      <td>0.121920</td>
      <td>0.445644</td>
      <td>0.089343</td>
      <td>0.185960</td>
      <td>0.005678</td>
      <td>0.082104</td>
      <td>0.002331</td>
      <td>0.097643</td>
      <td>0.034818</td>
      <td>0.004795</td>
      <td>0.339455</td>
      <td>-0.000314</td>
      <td>0.047124</td>
      <td>0.034206</td>
      <td>0.108243</td>
      <td>-0.027872</td>
      <td>0.080421</td>
      <td>-0.000443</td>
      <td>0.002411</td>
      <td>0.005036</td>
      <td>0.567795</td>
      <td>0.110521</td>
      <td>-0.000384</td>
      <td>0.018495</td>
      <td>0.196181</td>
      <td>0.047986</td>
      <td>-0.001367</td>
      <td>0.012681</td>
      <td>0.145594</td>
    </tr>
  </tbody>
</table>
</div>



A visualização dessa maneira fica ruim devido à grande quantidade de variáveis, portanto vamos selecioanr apenas a variável target e ordenar os valores.


```python
TargetCorr = matriz_corr['TARGET']
TargetCorr.sort_values()
```




    ind_var30                       -0.149811
    num_meses_var5_ult3             -0.148253
    num_var30                       -0.138289
    num_var42                       -0.135693
    ind_var5                        -0.135349
    num_var5                        -0.134095
    num_var4                        -0.080194
    num_var35                       -0.076872
    ind_var13                       -0.039612
    ind_var13_0                     -0.039471
    num_var13                       -0.038400
    ind_var12_0                     -0.038215
    num_var13_0                     -0.038045
    saldo_var30                     -0.037092
    ind_var39_0                     -0.035045
    ind_var13_corto                 -0.034438
    num_var13_corto                 -0.034432
    num_meses_var13_corto_ult3      -0.034367
    ind_var13_corto_0               -0.034337
    ind_var12                       -0.034255
    num_var13_corto_0               -0.034236
    ind_var41_0                     -0.034149
    num_var12                       -0.034108
    num_var30_0                     -0.033903
    ind_var5_0                      -0.032888
    ind_var24                       -0.032148
    num_var24                       -0.032139
    num_meses_var12_ult3            -0.031958
    num_var41_0                     -0.030302
    ind_var24_0                     -0.030300
    num_var24_0                     -0.030276
    saldo_var13                     -0.029548
    num_var39_0                     -0.029181
    num_var5_0                      -0.028912
    num_aport_var13_hace3           -0.026048
    saldo_var13_corto               -0.025843
    num_var12_0                     -0.025653
    saldo_medio_var13_corto_ult1    -0.025621
    saldo_medio_var13_corto_ult3    -0.024932
    saldo_var42                     -0.024462
    saldo_medio_var13_corto_hace2   -0.023094
    saldo_var12                     -0.021882
    saldo_var24                     -0.021568
    saldo_medio_var12_ult3          -0.021203
    ind_var43_recib_ult1            -0.021108
    saldo_medio_var12_ult1          -0.021008
    imp_aport_var13_hace3           -0.020979
    saldo_medio_var5_hace2          -0.020079
    ind_var14_0                     -0.020046
    var38                           -0.019510
    ind_var13_largo_0               -0.019227
    ind_var13_largo                 -0.019040
    saldo_medio_var5_ult3           -0.018989
    num_var42_0                     -0.017944
    num_var13_largo_0               -0.017718
    saldo_medio_var12_hace2         -0.017708
    num_var13_largo                 -0.017694
    num_meses_var13_largo_ult3      -0.016577
    num_var43_recib_ult1            -0.016511
    saldo_medio_var5_hace3          -0.016034
    saldo_medio_var5_ult1           -0.015666
    saldo_var13_largo               -0.014590
    saldo_var5                      -0.014132
    saldo_medio_var13_corto_hace3   -0.013977
    saldo_medio_var13_largo_ult3    -0.012270
    num_var20_0                     -0.012252
    ind_var20_0                     -0.012252
    saldo_medio_var13_largo_ult1    -0.012135
    saldo_medio_var13_largo_hace2   -0.011966
    saldo_medio_var12_hace3         -0.011867
    num_meses_var39_vig_ult3        -0.011169
    imp_trans_var37_ult1            -0.010928
    ind_var20                       -0.010555
    num_var20                       -0.010555
    num_trasp_var11_ult1            -0.009326
    imp_aport_var13_ult1            -0.009317
    ind_var31_0                     -0.009163
    num_aport_var13_ult1            -0.008415
    delta_imp_aport_var13_1y3       -0.008378
    delta_num_aport_var13_1y3       -0.008378
    ind_var19                       -0.007957
    ind_var31                       -0.007855
    num_var14                       -0.007387
    ind_var14                       -0.007384
    saldo_medio_var13_largo_hace3   -0.007005
    ind_var43_emit_ult1             -0.006847
    num_meses_var44_ult3            -0.006706
    num_var45_hace3                 -0.006117
    num_meses_var17_ult3            -0.005999
    num_var44_0                     -0.005699
    ind_var44_0                     -0.005697
    ind_var33_0                     -0.005560
    ind_var17_0                     -0.005442
    num_var33_0                     -0.005132
    ind_var33                       -0.005102
    num_var44                       -0.005089
    ind_var44                       -0.005089
    num_var31_0                     -0.005068
    saldo_medio_var8_hace2          -0.004952
    num_var33                       -0.004917
    num_meses_var33_ult3            -0.004914
    delta_num_venta_var44_1y3       -0.004772
    delta_imp_venta_var44_1y3       -0.004772
    saldo_var14                     -0.004393
    num_var31                       -0.004337
    imp_var43_emit_ult1             -0.004261
    ind_var17                       -0.004176
    num_compra_var44_hace3          -0.003799
    saldo_var8                      -0.003744
    var21                           -0.003719
    num_var14_0                     -0.003608
    num_compra_var44_ult1           -0.003566
    saldo_medio_var33_hace2         -0.003533
    num_venta_var44_ult1            -0.003395
    saldo_medio_var33_ult3          -0.003311
    saldo_medio_var8_ult1           -0.003294
    saldo_var33                     -0.003188
    saldo_medio_var44_hace2         -0.003177
    saldo_medio_var33_ult1          -0.003166
    num_aport_var33_hace3           -0.003105
    saldo_medio_var44_ult1          -0.003104
    imp_var7_recib_ult1             -0.003052
    saldo_medio_var8_hace3          -0.003049
    saldo_medio_var44_ult3          -0.003015
    num_op_var40_hace2              -0.002999
    num_aport_var17_hace3           -0.002977
    saldo_medio_var8_ult3           -0.002939
    num_var22_hace3                 -0.002856
    imp_aport_var33_hace3           -0.002671
    saldo_medio_var44_hace3         -0.002553
    imp_compra_var44_hace3          -0.002462
    saldo_medio_var33_hace3         -0.002431
    num_var43_emit_ult1             -0.002363
    saldo_var44                     -0.002274
    saldo_var20                     -0.002244
    num_op_var40_comer_ult3         -0.002229
    num_var17_0                     -0.002184
    ind_var6_0                      -0.002082
    ind_var29_0                     -0.002082
    num_var29_0                     -0.002082
    num_var6_0                      -0.002082
    num_aport_var33_ult1            -0.001862
    num_trasp_var33_in_hace3        -0.001803
    imp_trasp_var33_in_hace3        -0.001749
    delta_num_compra_var44_1y3      -0.001713
    delta_imp_compra_var44_1y3      -0.001713
    delta_imp_trasp_var33_in_1y3    -0.001646
    delta_num_trasp_var33_in_1y3    -0.001646
    imp_aport_var33_ult1            -0.001623
    delta_num_aport_var17_1y3       -0.001619
    delta_imp_aport_var17_1y3       -0.001619
    num_meses_var29_ult3            -0.001574
    num_trasp_var33_in_ult1         -0.001562
    num_reemb_var13_ult1            -0.001520
    delta_num_reemb_var13_1y3       -0.001520
    delta_imp_reemb_var13_1y3       -0.001520
    delta_imp_trasp_var17_out_1y3   -0.001472
    delta_num_trasp_var17_out_1y3   -0.001472
    num_trasp_var17_out_ult1        -0.001472
    num_trasp_var17_in_ult1         -0.001472
    delta_num_trasp_var17_in_1y3    -0.001472
    delta_imp_trasp_var17_in_1y3    -0.001472
    num_op_var40_comer_ult1         -0.001471
    imp_venta_var44_ult1            -0.001465
    saldo_var31                     -0.001372
    num_op_var39_hace3              -0.001352
    num_var17                       -0.001334
    num_var7_emit_ult1              -0.001275
    ind_var7_emit_ult1              -0.001275
    saldo_medio_var17_hace2         -0.001202
    num_venta_var44_hace3           -0.001202
    imp_trasp_var33_in_ult1         -0.001196
    num_op_var41_hace3              -0.001195
    imp_compra_var44_ult1           -0.001192
    saldo_medio_var29_ult3          -0.001188
    num_op_var40_hace3              -0.001152
    imp_trasp_var17_out_ult1        -0.001087
    ind_var6                        -0.001041
    ind_var29                       -0.001041
    ind_var13_medio_0               -0.001041
    ind_var13_medio                 -0.001041
    num_meses_var13_medio_ult3      -0.001041
    num_var13_medio_0               -0.001041
    num_var13_medio                 -0.001041
    ind_var18                       -0.001041
    ind_var18_0                     -0.001041
    ind_var34_0                     -0.001041
    ind_var34                       -0.001041
    num_var34                       -0.001041
    num_var34_0                     -0.001041
    num_var6                        -0.001041
    num_var29                       -0.001041
    num_var18_0                     -0.001041
    num_var18                       -0.001041
    delta_imp_amort_var18_1y3       -0.001041
    delta_imp_amort_var34_1y3       -0.001041
    saldo_medio_var29_hace2         -0.001034
    saldo_medio_var13_medio_hace2   -0.001028
    saldo_var6                      -0.001012
    saldo_var29                     -0.001012
    imp_trasp_var17_in_ult1         -0.001002
    imp_var7_emit_ult1              -0.000997
    saldo_medio_var29_ult1          -0.000990
    num_trasp_var17_in_hace3        -0.000988
    num_var45_hace2                 -0.000986
    imp_trasp_var17_in_hace3        -0.000980
    saldo_var34                     -0.000963
    saldo_medio_var13_medio_ult3    -0.000952
    imp_venta_var44_hace3           -0.000947
    saldo_medio_var13_medio_ult1    -0.000917
    saldo_var13_medio               -0.000917
    imp_aport_var17_hace3           -0.000906
    imp_amort_var34_ult1            -0.000894
    saldo_medio_var17_hace3         -0.000859
    imp_amort_var18_ult1            -0.000819
    saldo_var18                     -0.000806
    saldo_var17                     -0.000774
    delta_imp_trasp_var33_out_1y3   -0.000736
    delta_num_trasp_var33_out_1y3   -0.000736
    delta_num_reemb_var33_1y3       -0.000736
    delta_imp_reemb_var33_1y3       -0.000736
    num_reemb_var17_hace3           -0.000736
    imp_reemb_var17_hace3           -0.000736
    saldo_medio_var29_hace3         -0.000736
    imp_reemb_var33_ult1            -0.000736
    num_reemb_var33_ult1            -0.000736
    num_trasp_var33_out_ult1        -0.000736
    imp_trasp_var33_out_ult1        -0.000736
    delta_num_aport_var33_1y3       -0.000736
    delta_imp_aport_var33_1y3       -0.000736
    saldo_var1                      -0.000695
    saldo_medio_var17_ult1          -0.000583
    ind_var32_0                     -0.000503
    ind_var32                       -0.000503
    imp_op_var40_comer_ult3         -0.000358
    num_var32_0                     -0.000339
    num_var32                       -0.000339
    saldo_medio_var17_ult3          -0.000328
    num_var7_recib_ult1             -0.000290
    num_op_var40_ult3               -0.000119
    num_var37_med_ult2              -0.000029
    imp_ent_var16_ult1              -0.000017
    imp_sal_var16_ult1               0.000509
    saldo_var32                      0.000621
    delta_num_reemb_var17_1y3        0.000868
    delta_imp_reemb_var17_1y3        0.000868
    num_var45_ult3                   0.001121
    imp_aport_var17_ult1             0.001124
    ind_var7_recib_ult1              0.001156
    num_op_var40_ult1                0.001421
    ind_var1_0                       0.001608
    num_var37_0                      0.001635
    num_var37                        0.001635
    imp_reemb_var13_ult1             0.001677
    ind_var40_0                      0.001686
    num_med_var45_ult3               0.002067
    num_var1_0                       0.002184
    num_var40_0                      0.002292
    ind_var37_cte                    0.002483
    ind_var32_cte                    0.002639
    imp_op_var40_ult1                0.003087
    imp_op_var40_comer_ult1          0.003119
    ID                               0.003148
    ind_var37                        0.003197
    ind_var37_0                      0.003197
    imp_op_var39_comer_ult3          0.003517
    num_op_var39_comer_ult3          0.003532
    imp_op_var41_comer_ult3          0.003859
    num_aport_var17_ult1             0.004184
    num_op_var41_comer_ult3          0.004403
    var3                             0.004475
    saldo_var37                      0.004481
    num_op_var39_comer_ult1          0.004996
    saldo_var25                      0.005091
    saldo_var26                      0.005108
    num_op_var41_comer_ult1          0.005534
    num_ent_var16_ult1               0.007026
    num_op_var39_hace2               0.007185
    num_op_var41_hace2               0.007619
    ind_var9_cte_ult1                0.007669
    num_var45_ult1                   0.008008
    ind_var9_ult1                    0.008567
    imp_reemb_var17_ult1             0.008909
    num_sal_var16_ult1               0.009216
    ind_var10cte_ult1                0.009281
    num_var1                         0.009478
    ind_var1                         0.009571
    ind_var30_0                      0.009638
    num_var40                        0.009753
    num_var39                        0.009753
    ind_var40                        0.009753
    ind_var39                        0.009753
    num_var22_hace2                  0.009789
    imp_op_var41_comer_ult1          0.010082
    num_op_var39_ult3                0.010271
    ind_var10_ult1                   0.010329
    imp_op_var39_comer_ult1          0.010353
    num_op_var41_ult3                0.010462
    num_op_var41_ult1                0.011242
    num_op_var39_ult1                0.011246
    saldo_var40                      0.011784
    num_reemb_var17_ult1             0.011944
    num_var22_ult3                   0.012481
    num_med_var22_ult3               0.016262
    num_var26                        0.018156
    num_var26_0                      0.018156
    num_op_var40_efect_ult3          0.018353
    num_op_var40_efect_ult1          0.018579
    num_var25_0                      0.018722
    num_var25                        0.018722
    ind_var26                        0.019104
    ind_var26_0                      0.019104
    imp_op_var40_efect_ult1          0.019221
    ind_var25                        0.019497
    ind_var25_0                      0.019497
    num_op_var41_efect_ult3          0.019599
    imp_op_var40_efect_ult3          0.019965
    num_op_var39_efect_ult3          0.020237
    num_op_var41_efect_ult1          0.021036
    imp_op_var41_efect_ult3          0.021486
    num_op_var39_efect_ult1          0.021783
    imp_op_var39_efect_ult3          0.022172
    ind_var25_cte                    0.023351
    ind_var26_cte                    0.023538
    num_var22_ult1                   0.025189
    num_meses_var8_ult3              0.025943
    imp_op_var39_ult1                0.027416
    imp_op_var41_ult1                0.027586
    ind_var8                         0.027926
    num_var8                         0.027926
    imp_op_var41_efect_ult1          0.029479
    imp_op_var39_efect_ult1          0.030380
    num_var8_0                       0.046622
    ind_var8_0                       0.046665
    var15                            0.101322
    var36                            0.102919
    TARGET                           1.000000
    Name: TARGET, dtype: float64



Dessa forma ficou mais clara a visualização, agora podemos colocar um critério para selecionar as variáveis. Neste caso vou esolher as variáveis que possuem correlação maior do que 0.01 e menor do que -0.01. Assim teremos em nosso data set as variaveis com maiores correlação positiva e negativa com a vairável TARGET.


```python
vars_corr = []
for i in range(0,len(TargetCorr)):
    if TargetCorr[i] > 0.01:
        vars_corr.append(TargetCorr.index[i])
    elif TargetCorr[i] < -0.01:
        vars_corr.append(TargetCorr.index[i])
    else:
        0
len(vars_corr)
```




    118



Desta forma conseguimos diminuir consideravelmente as variáveis com maior importancia. Vamos gerar o modelo de machine learning com essas variáveis e ver o resultado.


```python
dfBestCorr = df_01.loc[:, vars_corr]
dfBestCorr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_var24_0</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>ind_var25_cte</th>
      <th>num_op_var40_efect_ult1</th>
      <th>saldo_var13</th>
      <th>saldo_medio_var5_hace3</th>
      <th>var36</th>
      <th>saldo_var12</th>
      <th>saldo_var40</th>
      <th>num_var12_0</th>
      <th>num_meses_var5_ult3</th>
      <th>num_var13_0</th>
      <th>num_op_var39_efect_ult1</th>
      <th>ind_var13_corto_0</th>
      <th>saldo_medio_var13_largo_ult3</th>
      <th>num_var13_largo</th>
      <th>imp_op_var41_ult1</th>
      <th>ind_var30</th>
      <th>saldo_medio_var13_largo_ult1</th>
      <th>num_var4</th>
      <th>saldo_medio_var5_ult3</th>
      <th>saldo_medio_var13_corto_ult1</th>
      <th>ind_var14_0</th>
      <th>saldo_var42</th>
      <th>saldo_medio_var13_corto_hace3</th>
      <th>num_var13_corto_0</th>
      <th>num_op_var41_efect_ult3</th>
      <th>num_var13_corto</th>
      <th>num_reemb_var17_ult1</th>
      <th>ind_var13_largo</th>
      <th>num_var41_0</th>
      <th>ind_var13_largo_0</th>
      <th>ind_var20_0</th>
      <th>num_var42</th>
      <th>saldo_medio_var13_largo_hace2</th>
      <th>ind_var24_0</th>
      <th>num_var26</th>
      <th>num_meses_var39_vig_ult3</th>
      <th>num_var42_0</th>
      <th>num_var30_0</th>
      <th>saldo_medio_var12_hace3</th>
      <th>ind_var20</th>
      <th>saldo_medio_var12_ult1</th>
      <th>ind_var26</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var39_efect_ult1</th>
      <th>num_var8_0</th>
      <th>ind_var12</th>
      <th>num_op_var39_ult1</th>
      <th>num_var26_0</th>
      <th>num_var13</th>
      <th>imp_op_var41_efect_ult3</th>
      <th>num_meses_var13_corto_ult3</th>
      <th>var15</th>
      <th>num_var20_0</th>
      <th>saldo_medio_var13_corto_hace2</th>
      <th>num_var22_ult1</th>
      <th>num_var30</th>
      <th>num_var5_0</th>
      <th>num_op_var41_efect_ult1</th>
      <th>num_op_var41_ult1</th>
      <th>num_var20</th>
      <th>ind_var26_cte</th>
      <th>ind_var26_0</th>
      <th>saldo_var30</th>
      <th>num_var43_recib_ult1</th>
      <th>saldo_var24</th>
      <th>saldo_var13_corto</th>
      <th>TARGET</th>
      <th>ind_var12_0</th>
      <th>num_var12</th>
      <th>ind_var39_0</th>
      <th>num_var8</th>
      <th>imp_op_var39_efect_ult3</th>
      <th>ind_var5</th>
      <th>saldo_medio_var12_ult3</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>num_op_var41_ult3</th>
      <th>var38</th>
      <th>ind_var13_0</th>
      <th>num_var22_ult3</th>
      <th>ind_var24</th>
      <th>num_var13_largo_0</th>
      <th>imp_trans_var37_ult1</th>
      <th>saldo_medio_var5_hace2</th>
      <th>saldo_medio_var13_corto_ult3</th>
      <th>num_meses_var13_largo_ult3</th>
      <th>num_meses_var12_ult3</th>
      <th>ind_var5_0</th>
      <th>num_op_var40_efect_ult3</th>
      <th>ind_var8</th>
      <th>ind_var43_recib_ult1</th>
      <th>saldo_medio_var5_ult1</th>
      <th>ind_var10_ult1</th>
      <th>ind_var25_0</th>
      <th>num_var35</th>
      <th>ind_var41_0</th>
      <th>ind_var13</th>
      <th>ind_var13_corto</th>
      <th>num_var25</th>
      <th>num_op_var39_efect_ult3</th>
      <th>num_var24</th>
      <th>saldo_var13_largo</th>
      <th>imp_aport_var13_hace3</th>
      <th>ind_var25</th>
      <th>imp_op_var41_comer_ult1</th>
      <th>ind_var8_0</th>
      <th>num_op_var39_ult3</th>
      <th>num_aport_var13_hace3</th>
      <th>imp_op_var41_efect_ult1</th>
      <th>saldo_medio_var12_hace2</th>
      <th>num_meses_var8_ult3</th>
      <th>num_var5</th>
      <th>num_var25_0</th>
      <th>num_med_var22_ult3</th>
      <th>saldo_var5</th>
      <th>imp_op_var39_ult1</th>
      <th>num_var39_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>99</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>39205.170000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>300.0</td>
      <td>88.89</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>300.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>122.22</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>3</td>
      <td>34</td>
      <td>0</td>
      <td>300.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>300.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>49278.030000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>240.75</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>300.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.18</td>
      <td>99</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>2.07</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>23</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0</td>
      <td>67333.770000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>195.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
      <td>138.84</td>
      <td>0.0</td>
      <td>0</td>
      <td>70.62</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>37</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70.62</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>195.0</td>
      <td>9</td>
      <td>64007.970000</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>186.09</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>91.56</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>195.0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>70.62</td>
      <td>195.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.30</td>
      <td>1</td>
      <td>135003.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1</td>
      <td>13501.47</td>
      <td>0.0</td>
      <td>0</td>
      <td>135003.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>0.0</td>
      <td>0</td>
      <td>85501.89</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>39</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>135003.00</td>
      <td>6</td>
      <td>135003.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>85501.89</td>
      <td>0.0</td>
      <td>0</td>
      <td>117310.979016</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>270003.0</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>40501.08</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## 1º modelo preditivo


```python
# Carregando as bibliotecas para gerar o modelo.
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
```

Para nosso primeiro modelos iremos utilizar os métodos de árvore de decisão e random forest classifier, esses métodos são relativamente simples, mas bem robustos, gerando resultados bem satisfatórios em geral.


```python
# Separando a variável target do dataset
X = dfBestCorr.drop(columns='TARGET')
y = dfBestCorr['TARGET']

#Fazendo o train test split para obeter variáveis de treino e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Gerando o modelo de árvore.
modelTreeCorr = DecisionTreeClassifier(random_state=0, max_depth=2)
modelTreeCorr = modelTreeCorr.fit(X_train, y_train)
predTreeCorr = modelTreeCorr.predict(X_test)
# Calculando a acurácia
accuracyTree = accuracy_score(y_test, predTreeCorr)
print('Acurácia para DecisionTree: %.3f' % accuracyTree)
# Calculando a métrica auc
pred_prob_Tree = modelTreeCorr.predict_proba(X_test)
aucTree = roc_auc_score(y_test, pred_prob_Tree[:,1])
print('AUC para DecisionTree: %.3f' % aucTree)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob_Tree[:,1], pos_label=1)

# Gerando o modelo random forest.
modelRF = RandomForestClassifier(n_estimators=100)
modelRF = modelRF.fit(X_train, y_train)
predRF = modelRF.predict(X_test)
# Calculando a acurácia
accuracyRF = accuracy_score(y_test, predRF)
print('Acurácia para Random Forest: %.3f' % accuracyRF)
# Calculando a métrica auc
pred_prob_RF = modelRF.predict_proba(X_test)
aucRF = roc_auc_score(y_test, pred_prob_RF[:,1])
print('AUC para DecisionTree: %.3f' % aucRF)
fpr2, tpr2, _ = roc_curve(y_test, pred_prob_RF[:,1])

# Gerando o gráfico da curva auc
plt.plot([0, 1], [0, 1], 'blue')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Decision Tree AUC Curve')
plt.plot(fpr2, tpr2, marker='.',color='green', label='Random Forest AUC Curve')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
```

    Acurácia para DecisionTree: 0.960
    AUC para DecisionTree: 0.765
    Acurácia para Random Forest: 0.952
    AUC para DecisionTree: 0.736



![png](output_27_1.png)


O cálculo dos erros nos dois algoritmos são bem próximos, mas podemos ver que a pelo cálculo do AUC o dataset encontra-se desbalanceado, precisamos balancear o modelo antes de tentar outros algoritmos.


```python
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))

ax = sns.countplot(x=df_01['TARGET'], data=df_01)
ax.xaxis.set_label_text("Variável Target",fontdict= {'size':16})
ax.yaxis.set_label_text("Quantidade", fontdict={'size':16})
ax.set_title("Quantidade de repostas positivas ou negativas", fontdict={'size':18})
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1, height ,ha="center", fontdict={'size':14})
plt.show()
```


![png](output_29_0.png)


## Balanceando o dataset

Para o balanceamento do dataset é importante que o mesmo já esteja divido em treino e teste, preservando, assim os dados de teste e impedindo que eles fiquem enviessados. Esse método será feito com o pacote SMOTE.


```python
#!pip install imbalanced-learn
# Carregando a biblioteca SMOTE que fará o balanceamento dos dados
from imblearn.over_sampling import SMOTE
```


```python
# Dividido o dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Criando os X e y balanceados
X_resampled, y_resampled = SMOTE(random_state=1).fit_sample(X_train, y_train)

# Gráfico para visualizar os dados divididos.
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))

ax = sns.countplot(x=y_resampled)
ax.xaxis.set_label_text("Variável Target",fontdict= {'size':16})
ax.yaxis.set_label_text("Quantidade", fontdict={'size':16})
ax.set_title("Quantidade de repostas positivas ou negativas", fontdict={'size':18})
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1, height ,ha="center", fontdict={'size':14})
plt.show()
```


![png](output_33_0.png)


Agora que as variáveis estão balanceadas podemos criar o modelo novamente.


```python
X_train = X_resampled
y_train = y_resampled

# Gerando o modelo de árvore.
modelTreeCorr = DecisionTreeClassifier(random_state=0, max_depth=2)
modelTreeCorr = modelTreeCorr.fit(X_train, y_train)
predTreeCorr = modelTreeCorr.predict(X_test)
# Calculando a acurácia
accuracyTree = accuracy_score(y_test, predTreeCorr)
print('Acurácia para DecisionTree: %.3f' % accuracyTree)
# Calculando a métrica auc
pred_prob_Tree = modelTreeCorr.predict_proba(X_test)
aucTree = roc_auc_score(y_test, pred_prob_Tree[:,1])
print('AUC para DecisionTree: %.3f' % aucTree)
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob_Tree[:,1], pos_label=1)

# Gerando o modelo random forest.
modelRF = RandomForestClassifier(n_estimators=100)
modelRF = modelRF.fit(X_train, y_train)
predRF = modelRF.predict(X_test)
# Calculando a acurácia
accuracyRF = accuracy_score(y_test, predRF)
print('Acurácia para Random Forest: %.3f' % accuracyRF)
# Calculando a métrica auc
pred_prob_RF = modelRF.predict_proba(X_test)
aucRF = roc_auc_score(y_test, pred_prob_RF[:,1])
print('AUC para Random Forest: %.3f' % aucRF)
fpr2, tpr2, _ = roc_curve(y_test, pred_prob_RF[:,1])

# Gerando o gráfico da curva auc
plt.plot([0, 1], [0, 1], 'blue')
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Decision Tree AUC Curve')
plt.plot(fpr2, tpr2, marker='.',color='green', label='Random Forest AUC Curve')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
```

    Acurácia para DecisionTree: 0.853
    AUC para DecisionTree: 0.763
    Acurácia para Random Forest: 0.912
    AUC para Random Forest: 0.758



![png](output_35_1.png)


O modelo melhorou um pouco, mas ainda não é o suficiente, vamos testar outras ferramentas e ver qual obterá o melhor resultado para o problema, mas antes iremos carregar o dataset de teste para ver o resultado no kaggle.


```python
test = pd.read_csv("test.csv")
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>imp_op_var40_ult1</th>
      <th>imp_op_var41_comer_ult1</th>
      <th>imp_op_var41_comer_ult3</th>
      <th>imp_op_var41_efect_ult1</th>
      <th>imp_op_var41_efect_ult3</th>
      <th>imp_op_var41_ult1</th>
      <th>imp_op_var39_efect_ult1</th>
      <th>imp_op_var39_efect_ult3</th>
      <th>imp_op_var39_ult1</th>
      <th>imp_sal_var16_ult1</th>
      <th>ind_var1_0</th>
      <th>ind_var1</th>
      <th>ind_var2_0</th>
      <th>ind_var2</th>
      <th>ind_var5_0</th>
      <th>ind_var5</th>
      <th>ind_var6_0</th>
      <th>ind_var6</th>
      <th>ind_var8_0</th>
      <th>ind_var8</th>
      <th>ind_var12_0</th>
      <th>ind_var12</th>
      <th>ind_var13_0</th>
      <th>ind_var13_corto_0</th>
      <th>ind_var13_corto</th>
      <th>ind_var13_largo_0</th>
      <th>ind_var13_largo</th>
      <th>ind_var13_medio_0</th>
      <th>ind_var13_medio</th>
      <th>ind_var13</th>
      <th>ind_var14_0</th>
      <th>ind_var14</th>
      <th>ind_var17_0</th>
      <th>ind_var17</th>
      <th>ind_var18_0</th>
      <th>ind_var18</th>
      <th>ind_var19</th>
      <th>ind_var20_0</th>
      <th>ind_var20</th>
      <th>ind_var24_0</th>
      <th>ind_var24</th>
      <th>ind_var25_cte</th>
      <th>ind_var26_0</th>
      <th>ind_var26_cte</th>
      <th>ind_var26</th>
      <th>ind_var25_0</th>
      <th>ind_var25</th>
      <th>ind_var27_0</th>
      <th>ind_var28_0</th>
      <th>ind_var28</th>
      <th>ind_var27</th>
      <th>ind_var29_0</th>
      <th>ind_var29</th>
      <th>ind_var30_0</th>
      <th>ind_var30</th>
      <th>ind_var31_0</th>
      <th>ind_var31</th>
      <th>ind_var32_cte</th>
      <th>ind_var32_0</th>
      <th>ind_var32</th>
      <th>ind_var33_0</th>
      <th>ind_var33</th>
      <th>ind_var34_0</th>
      <th>ind_var34</th>
      <th>ind_var37_cte</th>
      <th>ind_var37_0</th>
      <th>ind_var37</th>
      <th>ind_var39_0</th>
      <th>ind_var40_0</th>
      <th>ind_var40</th>
      <th>ind_var41_0</th>
      <th>ind_var41</th>
      <th>ind_var39</th>
      <th>ind_var44_0</th>
      <th>ind_var44</th>
      <th>ind_var46_0</th>
      <th>ind_var46</th>
      <th>num_var1_0</th>
      <th>num_var1</th>
      <th>num_var4</th>
      <th>num_var5_0</th>
      <th>num_var5</th>
      <th>num_var6_0</th>
      <th>num_var6</th>
      <th>num_var8_0</th>
      <th>num_var8</th>
      <th>num_var12_0</th>
      <th>num_var12</th>
      <th>num_var13_0</th>
      <th>num_var13_corto_0</th>
      <th>num_var13_corto</th>
      <th>num_var13_largo_0</th>
      <th>num_var13_largo</th>
      <th>num_var13_medio_0</th>
      <th>num_var13_medio</th>
      <th>num_var13</th>
      <th>num_var14_0</th>
      <th>num_var14</th>
      <th>num_var17_0</th>
      <th>num_var17</th>
      <th>num_var18_0</th>
      <th>num_var18</th>
      <th>num_var20_0</th>
      <th>num_var20</th>
      <th>num_var24_0</th>
      <th>num_var24</th>
      <th>num_var26_0</th>
      <th>num_var26</th>
      <th>num_var25_0</th>
      <th>num_var25</th>
      <th>num_op_var40_hace2</th>
      <th>num_op_var40_hace3</th>
      <th>num_op_var40_ult1</th>
      <th>num_op_var40_ult3</th>
      <th>num_op_var41_hace2</th>
      <th>num_op_var41_hace3</th>
      <th>num_op_var41_ult1</th>
      <th>num_op_var41_ult3</th>
      <th>num_op_var39_hace2</th>
      <th>num_op_var39_hace3</th>
      <th>num_op_var39_ult1</th>
      <th>num_op_var39_ult3</th>
      <th>num_var27_0</th>
      <th>num_var28_0</th>
      <th>num_var28</th>
      <th>num_var27</th>
      <th>num_var29_0</th>
      <th>num_var29</th>
      <th>num_var30_0</th>
      <th>num_var30</th>
      <th>num_var31_0</th>
      <th>num_var31</th>
      <th>num_var32_0</th>
      <th>num_var32</th>
      <th>num_var33_0</th>
      <th>num_var33</th>
      <th>num_var34_0</th>
      <th>num_var34</th>
      <th>num_var35</th>
      <th>num_var37_med_ult2</th>
      <th>num_var37_0</th>
      <th>num_var37</th>
      <th>num_var39_0</th>
      <th>num_var40_0</th>
      <th>num_var40</th>
      <th>num_var41_0</th>
      <th>num_var41</th>
      <th>num_var39</th>
      <th>num_var42_0</th>
      <th>num_var42</th>
      <th>num_var44_0</th>
      <th>num_var44</th>
      <th>num_var46_0</th>
      <th>num_var46</th>
      <th>saldo_var1</th>
      <th>saldo_var5</th>
      <th>saldo_var6</th>
      <th>saldo_var8</th>
      <th>saldo_var12</th>
      <th>saldo_var13_corto</th>
      <th>saldo_var13_largo</th>
      <th>saldo_var13_medio</th>
      <th>saldo_var13</th>
      <th>saldo_var14</th>
      <th>saldo_var17</th>
      <th>saldo_var18</th>
      <th>saldo_var20</th>
      <th>saldo_var24</th>
      <th>saldo_var26</th>
      <th>saldo_var25</th>
      <th>saldo_var28</th>
      <th>saldo_var27</th>
      <th>saldo_var29</th>
      <th>saldo_var30</th>
      <th>saldo_var31</th>
      <th>saldo_var32</th>
      <th>saldo_var33</th>
      <th>saldo_var34</th>
      <th>saldo_var37</th>
      <th>saldo_var40</th>
      <th>saldo_var41</th>
      <th>saldo_var42</th>
      <th>saldo_var44</th>
      <th>saldo_var46</th>
      <th>var36</th>
      <th>delta_imp_amort_var18_1y3</th>
      <th>delta_imp_amort_var34_1y3</th>
      <th>delta_imp_aport_var13_1y3</th>
      <th>delta_imp_aport_var17_1y3</th>
      <th>delta_imp_aport_var33_1y3</th>
      <th>delta_imp_compra_var44_1y3</th>
      <th>delta_imp_reemb_var13_1y3</th>
      <th>delta_imp_reemb_var17_1y3</th>
      <th>delta_imp_reemb_var33_1y3</th>
      <th>delta_imp_trasp_var17_in_1y3</th>
      <th>delta_imp_trasp_var17_out_1y3</th>
      <th>delta_imp_trasp_var33_in_1y3</th>
      <th>delta_imp_trasp_var33_out_1y3</th>
      <th>delta_imp_venta_var44_1y3</th>
      <th>delta_num_aport_var13_1y3</th>
      <th>delta_num_aport_var17_1y3</th>
      <th>delta_num_aport_var33_1y3</th>
      <th>delta_num_compra_var44_1y3</th>
      <th>delta_num_reemb_var13_1y3</th>
      <th>delta_num_reemb_var17_1y3</th>
      <th>delta_num_reemb_var33_1y3</th>
      <th>delta_num_trasp_var17_in_1y3</th>
      <th>delta_num_trasp_var17_out_1y3</th>
      <th>delta_num_trasp_var33_in_1y3</th>
      <th>delta_num_trasp_var33_out_1y3</th>
      <th>delta_num_venta_var44_1y3</th>
      <th>imp_amort_var18_hace3</th>
      <th>imp_amort_var18_ult1</th>
      <th>imp_amort_var34_hace3</th>
      <th>imp_amort_var34_ult1</th>
      <th>imp_aport_var13_hace3</th>
      <th>imp_aport_var13_ult1</th>
      <th>imp_aport_var17_hace3</th>
      <th>imp_aport_var17_ult1</th>
      <th>imp_aport_var33_hace3</th>
      <th>imp_aport_var33_ult1</th>
      <th>imp_var7_emit_ult1</th>
      <th>imp_var7_recib_ult1</th>
      <th>imp_compra_var44_hace3</th>
      <th>imp_compra_var44_ult1</th>
      <th>imp_reemb_var13_hace3</th>
      <th>imp_reemb_var13_ult1</th>
      <th>imp_reemb_var17_hace3</th>
      <th>imp_reemb_var17_ult1</th>
      <th>imp_reemb_var33_hace3</th>
      <th>imp_reemb_var33_ult1</th>
      <th>imp_var43_emit_ult1</th>
      <th>imp_trans_var37_ult1</th>
      <th>imp_trasp_var17_in_hace3</th>
      <th>imp_trasp_var17_in_ult1</th>
      <th>imp_trasp_var17_out_hace3</th>
      <th>imp_trasp_var17_out_ult1</th>
      <th>imp_trasp_var33_in_hace3</th>
      <th>imp_trasp_var33_in_ult1</th>
      <th>imp_trasp_var33_out_hace3</th>
      <th>imp_trasp_var33_out_ult1</th>
      <th>imp_venta_var44_hace3</th>
      <th>imp_venta_var44_ult1</th>
      <th>ind_var7_emit_ult1</th>
      <th>ind_var7_recib_ult1</th>
      <th>ind_var10_ult1</th>
      <th>ind_var10cte_ult1</th>
      <th>ind_var9_cte_ult1</th>
      <th>ind_var9_ult1</th>
      <th>ind_var43_emit_ult1</th>
      <th>ind_var43_recib_ult1</th>
      <th>var21</th>
      <th>num_var2_0_ult1</th>
      <th>num_var2_ult1</th>
      <th>num_aport_var13_hace3</th>
      <th>num_aport_var13_ult1</th>
      <th>num_aport_var17_hace3</th>
      <th>num_aport_var17_ult1</th>
      <th>num_aport_var33_hace3</th>
      <th>num_aport_var33_ult1</th>
      <th>num_var7_emit_ult1</th>
      <th>num_var7_recib_ult1</th>
      <th>num_compra_var44_hace3</th>
      <th>num_compra_var44_ult1</th>
      <th>num_ent_var16_ult1</th>
      <th>num_var22_hace2</th>
      <th>num_var22_hace3</th>
      <th>num_var22_ult1</th>
      <th>num_var22_ult3</th>
      <th>num_med_var22_ult3</th>
      <th>num_med_var45_ult3</th>
      <th>num_meses_var5_ult3</th>
      <th>num_meses_var8_ult3</th>
      <th>num_meses_var12_ult3</th>
      <th>num_meses_var13_corto_ult3</th>
      <th>num_meses_var13_largo_ult3</th>
      <th>num_meses_var13_medio_ult3</th>
      <th>num_meses_var17_ult3</th>
      <th>num_meses_var29_ult3</th>
      <th>num_meses_var33_ult3</th>
      <th>num_meses_var39_vig_ult3</th>
      <th>num_meses_var44_ult3</th>
      <th>num_op_var39_comer_ult1</th>
      <th>num_op_var39_comer_ult3</th>
      <th>num_op_var40_comer_ult1</th>
      <th>num_op_var40_comer_ult3</th>
      <th>num_op_var40_efect_ult1</th>
      <th>num_op_var40_efect_ult3</th>
      <th>num_op_var41_comer_ult1</th>
      <th>num_op_var41_comer_ult3</th>
      <th>num_op_var41_efect_ult1</th>
      <th>num_op_var41_efect_ult3</th>
      <th>num_op_var39_efect_ult1</th>
      <th>num_op_var39_efect_ult3</th>
      <th>num_reemb_var13_hace3</th>
      <th>num_reemb_var13_ult1</th>
      <th>num_reemb_var17_hace3</th>
      <th>num_reemb_var17_ult1</th>
      <th>num_reemb_var33_hace3</th>
      <th>num_reemb_var33_ult1</th>
      <th>num_sal_var16_ult1</th>
      <th>num_var43_emit_ult1</th>
      <th>num_var43_recib_ult1</th>
      <th>num_trasp_var11_ult1</th>
      <th>num_trasp_var17_in_hace3</th>
      <th>num_trasp_var17_in_ult1</th>
      <th>num_trasp_var17_out_hace3</th>
      <th>num_trasp_var17_out_ult1</th>
      <th>num_trasp_var33_in_hace3</th>
      <th>num_trasp_var33_in_ult1</th>
      <th>num_trasp_var33_out_hace3</th>
      <th>num_trasp_var33_out_ult1</th>
      <th>num_venta_var44_hace3</th>
      <th>num_venta_var44_ult1</th>
      <th>num_var45_hace2</th>
      <th>num_var45_hace3</th>
      <th>num_var45_ult1</th>
      <th>num_var45_ult3</th>
      <th>saldo_var2_ult1</th>
      <th>saldo_medio_var5_hace2</th>
      <th>saldo_medio_var5_hace3</th>
      <th>saldo_medio_var5_ult1</th>
      <th>saldo_medio_var5_ult3</th>
      <th>saldo_medio_var8_hace2</th>
      <th>saldo_medio_var8_hace3</th>
      <th>saldo_medio_var8_ult1</th>
      <th>saldo_medio_var8_ult3</th>
      <th>saldo_medio_var12_hace2</th>
      <th>saldo_medio_var12_hace3</th>
      <th>saldo_medio_var12_ult1</th>
      <th>saldo_medio_var12_ult3</th>
      <th>saldo_medio_var13_corto_hace2</th>
      <th>saldo_medio_var13_corto_hace3</th>
      <th>saldo_medio_var13_corto_ult1</th>
      <th>saldo_medio_var13_corto_ult3</th>
      <th>saldo_medio_var13_largo_hace2</th>
      <th>saldo_medio_var13_largo_hace3</th>
      <th>saldo_medio_var13_largo_ult1</th>
      <th>saldo_medio_var13_largo_ult3</th>
      <th>saldo_medio_var13_medio_hace2</th>
      <th>saldo_medio_var13_medio_hace3</th>
      <th>saldo_medio_var13_medio_ult1</th>
      <th>saldo_medio_var13_medio_ult3</th>
      <th>saldo_medio_var17_hace2</th>
      <th>saldo_medio_var17_hace3</th>
      <th>saldo_medio_var17_ult1</th>
      <th>saldo_medio_var17_ult3</th>
      <th>saldo_medio_var29_hace2</th>
      <th>saldo_medio_var29_hace3</th>
      <th>saldo_medio_var29_ult1</th>
      <th>saldo_medio_var29_ult3</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>2</td>
      <td>32</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6.0</td>
      <td>2.43</td>
      <td>6.00</td>
      <td>4.80</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40532.10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>2</td>
      <td>35</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>3.0</td>
      <td>2.55</td>
      <td>3.00</td>
      <td>2.85</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45486.72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>60.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>90.0</td>
      <td>57.00</td>
      <td>51.45</td>
      <td>66.15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>46993.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>2</td>
      <td>24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>99</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>187898.61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30.0</td>
      <td>3.87</td>
      <td>30.00</td>
      <td>21.30</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>73649.73</td>
    </tr>
  </tbody>
</table>
</div>



Preparando os dados do dataset de teste.|


```python
cols_test = vars_corr
cols_test.remove('TARGET')
test_pred = test[cols_test]
test_predRF = modelRF.predict(test_pred)
test_predRF[0:10]
```




    array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])




```python
#Criando a série com a resposta.
predict1 = pd.Series(test_predRF, index=test["ID"])
predict1.name = 'TARGET'
predict1.to_csv('predict1.csv', header=True)
#Mostrando os 5 primeiros elementos da série.
!head -n5 predict1.csv
```

    ID,TARGET
    2,0
    5,0
    6,0
    7,0


O resultado dado no kaggle não foi muito satisftório (score = 0.63), logo vamos tentar melhorar as features selections para obter resultados mais expressivos no ranking do kaggle.

## Feature selection utilizando SelectKBest

Nesta segunda tentativa vamos utilizar o método SelectKbest para fazer a feature selection do modelo. Este método faz a seleção também baseada na correlação entre as variáveis, mas ao contrário da primeira tentativa ele também leva em consideração a correlação entre outras variáveis e não somente o a variável target.

Antes de utilizar o método vamos também utilizar a biblioteca VarianceThreshold que remove as features com variância muito baixas, neste caso vou utilizar o treshold=0, assim removerá as variáveis com as mesmas variáveis entre elas. No primeiro modelo fora removidas apenas variáveis com valores todos iguais a 0, neste caso serão removidas features com vairáveis iguais, não importando se é apenas 0.


```python
#import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
X = df.drop(columns='TARGET')
y = df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Carregando a biblioteca
from sklearn.feature_selection import VarianceThreshold
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X_train)
# Colunas que serão removidas
constant_columns = [column for column in X_train.columns
                if column not in
X_train.columns[constant_filter.get_support()]]
X_train = constant_filter.transform(X_train)
X_test = constant_filter.transform(X_test)
print('Número de colunas excluídas: {}' .format(len(constant_columns)))
for column in constant_columns:
    print("Removed ", column)
```

    Número de colunas excluídas: 46
    Removed  ind_var2_0
    Removed  ind_var2
    Removed  ind_var18_0
    Removed  ind_var18
    Removed  ind_var27_0
    Removed  ind_var28_0
    Removed  ind_var28
    Removed  ind_var27
    Removed  ind_var41
    Removed  ind_var46_0
    Removed  ind_var46
    Removed  num_var18_0
    Removed  num_var18
    Removed  num_var27_0
    Removed  num_var28_0
    Removed  num_var28
    Removed  num_var27
    Removed  num_var41
    Removed  num_var46_0
    Removed  num_var46
    Removed  saldo_var18
    Removed  saldo_var28
    Removed  saldo_var27
    Removed  saldo_var41
    Removed  saldo_var46
    Removed  delta_imp_amort_var18_1y3
    Removed  delta_imp_reemb_var33_1y3
    Removed  delta_num_reemb_var33_1y3
    Removed  imp_amort_var18_hace3
    Removed  imp_amort_var18_ult1
    Removed  imp_amort_var34_hace3
    Removed  imp_reemb_var13_hace3
    Removed  imp_reemb_var33_hace3
    Removed  imp_reemb_var33_ult1
    Removed  imp_trasp_var17_out_hace3
    Removed  imp_trasp_var33_out_hace3
    Removed  num_var2_0_ult1
    Removed  num_var2_ult1
    Removed  num_reemb_var13_hace3
    Removed  num_reemb_var33_hace3
    Removed  num_reemb_var33_ult1
    Removed  num_trasp_var17_out_hace3
    Removed  num_trasp_var33_out_hace3
    Removed  saldo_var2_ult1
    Removed  saldo_medio_var13_medio_hace3
    Removed  saldo_medio_var29_hace3


## Balanceando o dataset


```python
X_resampled, y_resampled = SMOTE(random_state=1).fit_sample(X_train, y_train)

sns.set(style="whitegrid")
plt.figure(figsize=(10,6))

ax = sns.countplot(x=y_resampled)
ax.xaxis.set_label_text("Variável Target",fontdict= {'size':16})
ax.yaxis.set_label_text("Quantidade", fontdict={'size':16})
ax.set_title("Quantidade de repostas positivas ou negativas", fontdict={'size':18})
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1, height ,ha="center", fontdict={'size':14})
plt.show()
```


![png](output_47_0.png)


Agora que o dataset está limpo vamos fazer a feature selection utilizando o método SelectKBest. Vamos imprimir uma lista que começa com as duas variáveis mais importantes escolhidas pelo modelo até todas as variáveis do dataset e ver qual é o melhor resultado com as k variáveis escolhidas.


```python
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
```


```python
list_k = []

for k in range(2, X_train.shape[1], 10):
    selector = SelectKBest(score_func=f_classif, k=k)

    X_train2 = selector.fit_transform(X_resampled, y_resampled)
    X_test2 = selector.transform(X_test)

    modelRF = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    modelRF.fit(X_train2, y_resampled)

    predRF = modelRF.predict(X_test2)

    accuracyRF = accuracy_score(y_test, predRF)
    print('k = {} - Acuracy = {}' .format(k, accuracyRF))

    predRF_AUC= modelRF.predict_proba(X_test2)
    aucRF = roc_auc_score(y_test, predRF_AUC[:,1])
    print('k = {} - AUC = {}' .format(k, aucRF))

    list_k.append(aucRF)
```

    k = 2 - Acuracy = 0.7528720512145927
    k = 2 - AUC = 0.6341478008403783
    k = 12 - Acuracy = 0.9573796369376479
    k = 12 - AUC = 0.7118176419533141
    k = 22 - Acuracy = 0.9562395860738402
    k = 22 - AUC = 0.7476120510637279
    k = 32 - Acuracy = 0.956502674734719
    k = 32 - AUC = 0.7512333711011812
    k = 42 - Acuracy = 0.9571165482767693
    k = 42 - AUC = 0.7464504089187038
    k = 52 - Acuracy = 0.9575550293782338
    k = 52 - AUC = 0.7447076211922309
    k = 62 - Acuracy = 0.9391826712268702
    k = 62 - AUC = 0.7539238497343896
    k = 72 - Acuracy = 0.9396211523283347
    k = 72 - AUC = 0.7516210843940407
    k = 82 - Acuracy = 0.9396650004384811
    k = 82 - AUC = 0.7510102587259829
    k = 92 - Acuracy = 0.9421643427168289
    k = 92 - AUC = 0.7449100398821483
    k = 102 - Acuracy = 0.9443567482241515
    k = 102 - AUC = 0.7505960355492052
    k = 112 - Acuracy = 0.9436113303516619
    k = 112 - AUC = 0.7483234745022096
    k = 122 - Acuracy = 0.943918267122687
    k = 122 - AUC = 0.7461235535326167
    k = 132 - Acuracy = 0.9443567482241515
    k = 132 - AUC = 0.7492617296875502
    k = 142 - Acuracy = 0.9442690520038587
    k = 142 - AUC = 0.7476081819187115
    k = 152 - Acuracy = 0.9447952293256161
    k = 152 - AUC = 0.7475468746918725
    k = 162 - Acuracy = 0.943918267122687
    k = 162 - AUC = 0.7477408062249802
    k = 172 - Acuracy = 0.9438305709023941
    k = 172 - AUC = 0.7499001635774665
    k = 182 - Acuracy = 0.9447075331053232
    k = 182 - AUC = 0.7445276685056339
    k = 192 - Acuracy = 0.9442690520038587
    k = 192 - AUC = 0.7441557563017763
    k = 202 - Acuracy = 0.9438744190125405
    k = 202 - AUC = 0.7452561910688649
    k = 212 - Acuracy = 0.9449267736560554
    k = 212 - AUC = 0.7461257502084971
    k = 222 - Acuracy = 0.9451898623169341
    k = 222 - AUC = 0.7484478363117013
    k = 232 - Acuracy = 0.94536525475752
    k = 232 - AUC = 0.7508244648785226
    k = 242 - Acuracy = 0.9456283434183986
    k = 242 - AUC = 0.750194118749802
    k = 252 - Acuracy = 0.9449267736560554
    k = 252 - AUC = 0.7514154455769807
    k = 262 - Acuracy = 0.9454091028676664
    k = 262 - AUC = 0.7515700116798255
    k = 272 - Acuracy = 0.9449706217662018
    k = 272 - AUC = 0.7512742592272244
    k = 282 - Acuracy = 0.94536525475752
    k = 282 - AUC = 0.7487220463633407
    k = 292 - Acuracy = 0.9449706217662018
    k = 292 - AUC = 0.748330713547724
    k = 302 - Acuracy = 0.94536525475752
    k = 302 - AUC = 0.7506089659822274
    k = 312 - Acuracy = 0.9452337104270806
    k = 312 - AUC = 0.7466572958470594
    k = 322 - Acuracy = 0.9498377619924582
    k = 322 - AUC = 0.7675076440576297


## Imprimindo o resultados das k variáveis


```python
pd.Series(list_k, index=range(2, X_train.shape[1], 10)).plot(figsize=(10,7))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a27521b90>




![png](output_52_1.png)


Podemos ver que a partir de 20 features os resutados são bem semelhantes neste método, portanto vou treinar o modelo com 70 features para ver como o resultado fica no kaggle.


```python
k = 70
selector = SelectKBest(score_func=f_classif, k=k)

X_train2 = selector.fit_transform(X_resampled, y_resampled)
X_test2 = selector.transform(X_test)

modelRF = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelRF.fit(X_train2, y_resampled)

predRF = modelRF.predict(X_test2)

accuracyRF = accuracy_score(y_test, predRF)
print('k = {} - Acuracy = {}' .format(k, accuracyRF))

predRF_AUC= modelRF.predict_proba(X_test2)
aucRF = roc_auc_score(y_test, predRF_AUC[:,1])
print('k = {} - AUC = {}' .format(k, aucRF))

fpr2, tpr2, _ = roc_curve(y_test, predRF_AUC[:,1])
plt.plot([0, 1], [0, 1], 'blue')
plt.plot(fpr2, tpr2, marker='.',color='green', label='Random Forest AUC Curve')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
```

    k = 70 - Acuracy = 0.9391388231167237
    k = 70 - AUC = 0.7518087254462185



![png](output_54_1.png)


Podemos perceber que o resultado ficou um pouco pior do que na primeira predição, mas a diferença é bem pouco, de qualquer forma, não é um resultado satisfatório.


```python
test = pd.read_csv("test.csv")
test_pred = test.drop(columns=constant_columns)

mask = selector.get_support()
test_pred = test_pred.iloc[:, mask]
test_pred.shape
```




    (75818, 70)




```python
test_predRF = modelRF.predict(test_pred)

#Criando a série com a resposta.
predict2 = pd.Series(test_predRF, index=test["ID"])
predict2.name = 'TARGET'
predict2.to_csv('predict2.csv', header=True)
#Mostrando os 5 primeiros elementos da série.
!head -n5 predict2.csv
```

    ID,TARGET
    2,0
    5,0
    6,0
    7,0


Este método não foi satisfatório para o kaggle, conseguindo apenas 0,56 de erro na métrica AUC. Vamos tentar utilizar outros métodos para melhorar a resposta do problema.

## Método Embedded

Utilizando as mesmas features do dataset anterior vamos utilizar um método embedde para fazer a feature selection. Este método utliza modelos essemble para descobrir quais são as variáeis que mais influenciam no modelo utilizado. Para isso iremos carregar o pacote SelectFromModel, que seleciona as variáveis do modelo.


```python
from sklearn.feature_selection import SelectFromModel
```

Vamos utilizar o método random forest classifiers com 100 árvores.


```python
X_train = X_resampled
y_train = y_resampled
print('Before: {}' .format(X_train.shape))
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
clf.fit(X_train, y_train)

model = SelectFromModel(clf, prefit=True)
X_new_train = model.transform(X_train)

print('After: {}' .format(X_new_train.shape))
```

    Before: (102242, 324)
    After: (102242, 42)


O método reduziu a número de variáeis de 324 para 42. Vamos testar o modelo e ver qual é o resultado dado.


```python
X_new_test = model.transform(X_test)
modelRF = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelRF.fit(X_new_train, y_train)

predRF = modelRF.predict(X_new_test)

accuracyRF = accuracy_score(y_test, predRF)
print('Acuracy = {}' .format(accuracyRF))

predRF_AUC= modelRF.predict_proba(X_new_test)
aucRF = roc_auc_score(y_test, predRF_AUC[:,1])
print('AUC = {}' .format(aucRF))
```

    Acuracy = 0.9491361922301149
    AUC = 0.7700884885946342


Obtemos uma métrica um pouco melhor do que as vistas anteriormente. Agora vamos carregar o dataset de treino para carregar no kaggle


```python
test = pd.read_csv("test.csv")
test_pred = test.drop(columns=constant_columns)
test_pred = model.transform(test_pred)
test_pred.shape
```




    (75818, 42)




```python
test_predRF = modelRF.predict(test_pred)

#Criando a série com a resposta.
predict3 = pd.Series(test_predRF, index=test["ID"])
predict3.name = 'TARGET'
predict3.to_csv('predict3.csv', header=True)
#Mostrando os 5 primeiros elementos da série.
!head -n5 predict3.csv
```

    ID,TARGET
    2,0
    5,0
    6,0
    7,0


Este resultado também não foi satisfatório (0.53). Vamos utilizar outro método essemble e ver se o resultado fica melhor.

## Gradient Boosting


```python
from sklearn.ensemble import GradientBoostingClassifier
```


```python
X_train = X_resampled
y_train = y_resampled
print('Before: {}' .format(X_train.shape))
clf = GradientBoostingClassifier(n_estimators=20, max_features=2, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

model = SelectFromModel(clf, prefit=True)
X_new_train = model.transform(X_train)

print('After: {}' .format(X_new_train.shape))
```

    Before: (102242, 324)
    After: (102242, 24)



```python
X_new_test = model.transform(X_test)

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    modelGB = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    modelGB.fit(X_new_train, y_train)
    predGB = modelGB.predict(X_new_test)
    accuracyGB = accuracy_score(y_test, predGB)
    print('Learning rate: {} - Acuracy = {}' .format(learning_rate, accuracyGB))
    predGB_AUC= modelGB.predict_proba(X_new_test)
    aucGB = roc_auc_score(y_test, predGB_AUC[:,1])
    print('Learning rate: {} - Auc = {}' .format(learning_rate, aucGB))
```

    Learning rate: 0.05 - Acuracy = 0.7406384284837324
    Learning rate: 0.05 - Auc = 0.7588856163410719
    Learning rate: 0.075 - Acuracy = 0.7394545295097781
    Learning rate: 0.075 - Auc = 0.7565580884726189
    Learning rate: 0.1 - Acuracy = 0.853064982899237
    Learning rate: 0.1 - Auc = 0.7569982973265705
    Learning rate: 0.25 - Acuracy = 0.8524072612470402
    Learning rate: 0.25 - Auc = 0.7685814940541227
    Learning rate: 0.5 - Acuracy = 0.8697711128650355
    Learning rate: 0.5 - Auc = 0.7802506856499403
    Learning rate: 0.75 - Acuracy = 0.8784530386740331
    Learning rate: 0.75 - Auc = 0.7783572009656388
    Learning rate: 1 - Acuracy = 0.878146101903008
    Learning rate: 1 - Auc = 0.7658953588482229


Podemos ver que o melhor learning rate é de 0.5, portanto vamos rodar o modelo com essa taxa e ver o resultado no kaggle.


```python
modelGB = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
modelGB.fit(X_new_train, y_train)
predGB = modelGB.predict(X_new_test)
```


```python
test = pd.read_csv("test.csv")
test_pred = test.drop(columns=constant_columns)
test_pred = model.transform(test_pred)
test_pred.shape
```




    (75818, 24)




```python
test_predGB = modelGB.predict(test_pred)

#Criando a série com a resposta.
predict4 = pd.Series(test_predGB, index=test["ID"])
predict4.name = 'TARGET'
predict4.to_csv('predict4.csv', header=True)
#Mostrando os 5 primeiros elementos da série.
!head -n5 predict4.csv
```

    ID,TARGET
    2,0
    5,0
    6,0
    7,0


Com esse modelo conseguimos um score de 0.70 com os dados de teste do kaggle, um resultado bem satisfatório comparado aos anterires.

O ideal para fazer uma análise exploratória é conhecer bem o dados que temos, neste caso não foi possível devido à disponibização dos dados, as variáeis estão todas nomeados com código e também existem muitas variáveis. Portanto resolvi utilizar algumns métodos para computacionais para fazer a feature selection. Existem outras bibliotecas mais robustas para fazer a feature selection, como o RFE, mas como temos muitas variáveis neste dataset o tempo computacional fica inviável. Os métos utilizado até neste projeto obtiveram bons resultados com um tempo computacional viável, portanto foram satisfatórios para este projeto.

Podemos fazer algumas coisas que podem melhorar o resultado como a normalização dos dados, mas fico feliz com o resultado obtido até o momento.
