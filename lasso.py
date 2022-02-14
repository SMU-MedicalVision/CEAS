import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,LassoCV
import pandas as pd

fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=600)
plt.subplots_adjust(left=0.1, right=0.94, top=0.85, bottom=0.2, wspace=0.3, hspace=0.0)
for i, ax in enumerate(axs):
    ax.annotate('B' if i else 'A', xy=[-0.15, 1.05], xycoords='axes fraction', fontsize=16, fontweight='bold')
    exl = pd.read_excel(f'exp{i+1}.xlsx', sheet_name='Sheet1', index_col=None,
                        header=0)
    df_concat = exl[['Gender', 'Age', 'Disease Duration', 'ESR', 'CRP', 'HLA-B27', 'Label']]

    lambs = np.logspace(-4, -1, 200)
    lasso_coefs = []
    Xtrain, Ytrain = df_concat[['Gender', 'Age', 'Disease Duration', 'ESR', 'CRP', 'HLA-B27']], df_concat['Label']
    for lamb in lambs:
        lasso = Lasso(alpha=lamb, normalize=True,max_iter=10000).fit(Xtrain, Ytrain)
        lasso_coefs.append(lasso.coef_)
    lasso = LassoCV(normalize=False).fit(Xtrain, Ytrain)
    print(lasso.alpha_)
    # plt.figure(figsize=(6, 4),dpi=600)
    ax.plot(lambs, lasso_coefs)
    ax.legend(['Gender', 'Age', 'Disease Duration', 'ESR', 'CRP', 'HLA-B27'])
    ax.set_xscale('log')
    ax.set_xlabel('λ')
    ax.set_ylabel('Cofficients')
    ax.set_xlim(1e-4,1e-1)
    ax.set_ylim(-0.02,0.25)
os.makedirs('lasso')
plt.savefig('lasso/fig.svg')
plt.savefig('lasso/fig.tif')
plt.show()
