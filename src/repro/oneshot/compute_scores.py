import os
from glob import glob
import argparse
import pandas as pd
import ir_datasets
import pyterrier as pt
import autoqrels

if not pt.started():
    pt.init()

DATASETS = {
    'dl19': 'msmarco-passage/trec-dl-2019/judged',
    'dl20': 'msmarco-passage/trec-dl-2020/judged',
    'dl21': 'msmarco-passage-v2/trec-dl-2021/judged',
}

def build_cache(labeler, dataset, irds):
    qrels = pt.io.read_qrels(f'{dataset}.bm25-firstrel.qrels')
    qrels = qrels.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'})
    all_qrels_as_run = pd.DataFrame(irds.qrels).rename(columns={'relevance': 'score'})
    labeler.infer_qrels(all_qrels_as_run, qrels)
    for runfile in pt.tqdm(glob(f'{dataset}-runs/*'), unit='runfile'):
        run = pt.io.read_results(runfile)
        run = run.rename(columns={'qid': 'query_id', 'docno': 'doc_id', 'score': 'score'})
        labeler.infer_qrels(run, qrels)

def maxrep_bm25(name, cache_path):
    raise NotImplementedError('todo: coming soon')
    from pyterrier_pisa import PisaIndex
    from pyterrier_adaptive import CorpusGraph
    for dataset, dsid in DATASETS.items():
        ds = ir_datasets.load(dsid)
        corpus_id = dataset.split('/')[0]
        index = PisaIndex(f'{corpus_id}.pisa')
        if not index.built():
            index.index(pt.get_dataset(f'irds:{dataset}').get_corpus_iter())
        # TODO: build bm25 corpus graph
        corpus_graph = index.corpus_graph(k=128)
        labeler = autoqrels.oneshot.MaxRep(corpus_graph, cache_path=cache_path, verbose=True)
        build_cache(labeler, dataset, ds)
        del labeler

def maxrep_tcthnp(name, cache_path):
    raise NotImplementedError('todo: coming soon')
    from pyterrier_dr import FlexIndex, TctColBert
    for dataset, dsid in DATASETS.items():
        ds = ir_datasets.load(dsid)
        corpus_id = dataset.split('/')[0]
        index = FlexIndex(f'{corpus_id}.tcthnp.flex')
        if not index.built():
            pipeline = TctColBert('') >> index
            pipeline.index(pt.get_dataset(f'irds:{dataset}').get_corpus_iter())
        corpus_graph = index.corpus_graph(k=128)
        labeler = autoqrels.oneshot.MaxRep(corpus_graph, cache_path=cache_path, verbose=True)
        build_cache(labeler, dataset, ds)
        del labeler

def duot5(name, cache_path):
    for dataset, dsid in DATASETS.items():
        ds = ir_datasets.load(dsid)
        labeler = autoqrels.oneshot.DuoT5(ds, cache_path=cache_path, verbose=True)
        build_cache(labeler, dataset, ds)
        del labeler

def duoprompt(name, cache_path):
    for dataset, dsid in DATASETS.items():
        ds = ir_datasets.load(dsid)
        labeler = autoqrels.oneshot.DuoPrompt(ds, cache_path=cache_path, verbose=True)
        build_cache(labeler, dataset, ds)
        del labeler

SYSTEMS = {
    'maxrep.bm25-128': maxrep_bm25,
    'maxrep.tcthnp-128': maxrep_tcthnp,
    'duot5': duot5,
    'duoprompt': duoprompt,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('system', choices=SYSTEMS.keys())
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()

    # Define the output path in your Google Drive
    drive_path = '/content/drive/MyDrive/'  # Modify this to your specific folder in Google Drive
    output_path = os.path.join(drive_path, f'{args.system}.cache.json.gz')

    # Check if the file exists and the replace flag is set, then delete the existing file
    if args.replace and os.path.exists(output_path):
        os.unlink(output_path)

    # Call the selected system function with the new output path
    SYSTEMS[args.system](args.system, output_path)

if __name__ == '__main__':
    main()
