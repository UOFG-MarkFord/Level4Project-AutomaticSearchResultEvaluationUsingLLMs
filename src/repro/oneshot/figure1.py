import pandas as pd
import ir_measures
import ir_datasets
from autoqrels.oneshot import OneShotLabeler
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

dataset = ir_datasets.load('msmarco-document/trec-dl-2019/judged')
SYSTEMS = [
    
    ('DuoT5Baseline',   OneShotLabeler('duot5Doc.cache.json.gz'),             False, -38),
    ('Generic Summarization Model',   OneShotLabeler('duot5Abb.cache.json.gz'),  False, -38),
    
 
]


def qdid(df):
    return df['query_id'] + '\t' + df['doc_id']

# Preparing data
full_qrels = pd.DataFrame(dataset.qrels)
full_qrels['relevance'] = (full_qrels['relevance'] >= 2).astype(int) # Binarize
sparse_qrels = pd.DataFrame(ir_measures.read_trec_qrels('dl19.bm25-firstrel.qrels'))
missing_qrels = full_qrels[~qdid(full_qrels).isin(qdid(sparse_qrels))]
missing_qrels = missing_qrels[missing_qrels['query_id'].isin(sparse_qrels['query_id'])]

# Plot settings
fig, ax = plt.subplots(figsize=(5, 5*0.6))
ax.set_xlim(0, 0.7)
ax.set_xlabel('Recall')
ax.set_ylim(0.2, 0.68)  # This sets the ylim for the main plot
ax.set_ylabel('Precision')

inset_scale = 0.48
axins = ax.inset_axes([
    1-inset_scale/2-0.015, 1-inset_scale-0.035,
    inset_scale/2, inset_scale])
axins.set_xlim(0, 1)
axins.set_xticks([0, 1])
axins.set_ylim(0, 1)
axins.set_yticks([0, 1])
lines = []

for name, model, under, rot in SYSTEMS:
    inf_qrels = model.infer_qrels(missing_qrels, sparse_qrels)
    inf_qrel_map = {}
    for rec in inf_qrels.itertuples(index=False):
        inf_qrel_map[rec.query_id, rec.doc_id] = rec.relevance
    label, pred = [], []
    for qid, did, rel in zip(missing_qrels['query_id'],
                             missing_qrels['doc_id'],
                             missing_qrels['relevance']):
        label.append(rel)
        pred.append(inf_qrel_map[qid, did])
    p, r, t = precision_recall_curve(label, pred)
    f1 = 2 * p * r / (p + r + 1e-10)
    mx = f1[2:-2].argmax() + 2 # f1 score undefined at bounds? Correct this.
    f1_max = f1[mx]  # max F1 score
    ap = average_precision_score(label, pred)
    print(f"{name}: AP={ap}, Max F1={f1_max}")  # Print AP and max F1 score
    line, = ax.plot(r, p, label=name)
    axins.plot(r, p)
    plt.plot(r[mx], p[mx], c=line.get_color())



 
    

# Display legend in the top left corner with system names
ax.legend(loc='upper left', fontsize='small')

ax.spines['top'].set_linestyle((0,(4,4)))
ax.spines['left'].set_linestyle((0,(4,4)))
ax.spines['bottom'].set_linestyle((0,(4,4)))
ax.spines['right'].set_linestyle((0,(4,4)))
axins.add_patch(patches.Rectangle((0.0, 0.2), 0.6, 0.45, linewidth=1, edgecolor='k', facecolor='none', ls=':', zorder=100))
axins.annotate('detail', (0.3, 0.18), ha='center', va='top')
plt.tight_layout()
plt.savefig('figure1.pdf')