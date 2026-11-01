from __future__ import annotations
import csv, io, json, urllib.request
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).parent
DATA=ROOT/'data'; BASE='https://raw.githubusercontent.com/petermattia/revisit-severson-et-al/main/data'
def fetch(rel):
 p=DATA/rel; p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): p.write_bytes(urllib.request.urlopen(f'{BASE}/{rel}',timeout=60).read())
 return p
def load(split,n):
 labels=np.loadtxt(fetch(f'cycle_lives/{split}_cycle_lives.csv'))
 out=[]
 for i,y in enumerate(labels,1):
  x=np.loadtxt(fetch(f'{split}/cell{i}.csv'),delimiter=',')
  # Four contiguous temporal windows summarize repeated cycle segments per cell.
  feats=x[:250].mean(0)
  out.append((i,feats,float(y)))
 return out

def windows(split):
 labels=np.loadtxt(fetch(f'cycle_lives/{split}_cycle_lives.csv')); out=[]
 for i,y in enumerate(labels,1):
  x=np.loadtxt(fetch(f'{split}/cell{i}.csv'),delimiter=',')
  for j in range(4): out.append((i,j,x[j*250:(j+1)*250].mean(0),float(y)))
 return out
train=windows('train'); test=load('test1',43)
testx=np.array([r[1] for r in test]); testy=np.array([r[2] for r in test])
results=[]
for n, reps in [(8,20),(32,20)]:
 scores=[]
 for seed in range(reps):
  rng=np.random.default_rng(seed); ids=np.sort(rng.choice(41,n,replace=False))
  # Match total training segments S=32: 4 segments per cell at N=8, 1 per cell at N=32.
  per=32//n; samples=[]; targets=[]
  for ix in ids:
   cell=ix+1; ws=[r for r in train if r[0]==cell]
   chosen=rng.choice(4,per,replace=False)
   for j in chosen: samples.append(ws[int(j)][2]); targets.append(ws[int(j)][3])
  model=RandomForestRegressor(n_estimators=200,min_samples_leaf=2,max_features=.7,random_state=seed,n_jobs=-1)
  model.fit(samples,targets); pred=model.predict(testx)
  scores.append(mean_absolute_error(testy,pred))
 results.append({'N_unique_cells':n,'S_segments':32,'repeats':reps,'MAE_cycles_median':float(np.median(scores)),'MAE_cycles_iqr':[float(np.quantile(scores,.25)),float(np.quantile(scores,.75))]})
report={'dataset':'Severson processed features; train=41 cells, test1=43 separate cells','target':'cycle life in cycles','features':'training segment vectors are means of 99 features over each consecutive 250-row window; test vectors use the first window only','model':'RandomForestRegressor(200 trees, min_samples_leaf=2)','design':'same S=32; compare N=8 (4 segments/cell) vs N=32 (1 segment/cell); 20 random training-cell subsamples; test1 fixed','results':results,'delta_MAE_32_vs_8':results[1]['MAE_cycles_median']-results[0]['MAE_cycles_median'],'caveat':'Pilot sampling scheme is a proxy for repeated segments; four file windows are temporal partitions of processed feature rows and are not asserted to be independent physical cycling protocols. Result is descriptive and does not establish that smaller N is generally preferable.'}
(ROOT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
