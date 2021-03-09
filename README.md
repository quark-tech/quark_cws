训练
train/predict: sh run_seq.sh train/predict
评测
evaluate: ./conlleval.pl -d '\t' < checkpoint/seq/test_results.tsv
