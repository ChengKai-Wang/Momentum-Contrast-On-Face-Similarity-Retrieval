import pandas as pd
import math
import os
import numpy as np
from scipy import spatial
from scipy.stats import pearsonr
from numpy.linalg import norm
import random
import time
import math
random.seed(time.time())

# load measurement file
xls = pd.ExcelFile('./MyFile.xlsx')
dicts = pd.read_excel('./MyFile.xlsx', header=0, sheet_name=xls.sheet_names)

# compute the movement of each pair (presugery, postsurgery) element in table
vectors = {}
for i in dicts:
    vector = []
    for j in dicts[i]:
        if j[0:4] == 'PRE_' :
            for s, k in enumerate(dicts[i][j]):
                if not math.isnan(k):
                    vector.append(abs(dicts[i][j][s]-dicts[i]['POST_'+j[4:]][s]))
    vectors[i]=vector

# Load embeddings (representations) from encoders
""" Type: {'img' : representation, ...} """
file = open('MoCo_featDS2.json', 'r')
# file = open('SimCLR2.json', 'r')
# file = open('HD_feat.json','r')

feats = file.read()
feats = eval(feats)

score = 0
n = 0
for img in vectors:
    # store representations similarity and measurement values similarity
    sim = []
    ans = []
    meas_query = np.array(vectors[img])
    rep_query = feats[img]
    for img2 in vectors:
        if img==img2:
            continue
        meas_comp = np.array(vectors[img2])
        ans.append(norm(meas_query-meas_comp))
        rep_comp = np.array(feats[img2])
        sim.append(1-np.dot(np.array(rep_query),np.array(rep_comp).T)/(norm(rep_query)*norm(rep_comp)))
#         sim.append(np.count_nonzero(rep_query!=rep_comp))
    score += pearsonr(np.array(sim), np.array(ans))[0]
    print(pearsonr(np.array(sim), np.array(ans))[0])
    n+=1
print('correlation coef: %.3f'%(score/n))
        

