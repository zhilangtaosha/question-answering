### Pipeline Parameters:
* Name: BM25_1000_DPR_20_electra-base-squad2_naturalQuestions-dev_1000
* BM25 top K = 1000
* Dense top K = 20
* Reader top K = 1
* FAISS dimension = 768
* USE_GPU = True
* Number of questions: 284
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.268
* BM25 Recall @ 10: 0.349
* BM25 Recall @ 20: 0.423
* BM25 Recall @ 50: 0.507
* BM25 Recall @ 100: 0.56
### Dense Retrieval recall 
* Dense Recall @ 5: 0.454
* Dense Recall @ 10: 0.507
* Dense Recall @ 20: 0.556
* Dense Recall @ 50: 0.592
* Dense Recall @ 100: 0.606
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.102
* BM25 Precision @ 10: 0.089
* BM25 Precision @ 20: 0.078
* BM25 Precision @ 50: 0.066
* BM25 Precision @ 100: 0.057
### Dense Retrieval Precision 
* Dense Precision @ 5: 0.191
* Dense Precision @ 10: 0.165
* Dense Precision @ 20: 0.136
* Dense Precision @ 50: 0.1
* Dense Precision @ 100: 0.08
### F1 
* Mean F1 per q: 0.199
* Median F1 per q: 0.015
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.339
### Precision 
* Mean precision per q: 0.418
* Median precision per q: 0.423
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.41
### Recall 
* Mean recall per q: 0.21
* Median recall per q: 0.008
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.368
### Exact Match 
* Mean EM per q: 0.113
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.316
### Time(s) 
* Mean time per q: 43.86s
* Max time per q: 49.79s
* Min time per q: 38.22s
* Std time per q: 2.73s
