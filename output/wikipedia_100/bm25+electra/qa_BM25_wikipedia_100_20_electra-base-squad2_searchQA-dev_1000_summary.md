### Pipeline Parameters:
* Name: BM25_wikipedia_100_20_electra-base-squad2_searchQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.588
* BM25 Recall @ 10: 0.649
* BM25 Recall @ 20: 0.708
* BM25 Recall @ 50: 0.708
* BM25 Recall @ 100: 0.708
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.281
* BM25 Precision @ 10: 0.233
* BM25 Precision @ 20: 0.194
* BM25 Precision @ 50: 0.078
* BM25 Precision @ 100: 0.039
### F1 
* Mean F1 per q: 0.135
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.311
### Precision 
* Mean precision per q: 0.126
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.299
### Recall 
* Mean recall per q: 0.166
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.364
### Exact Match 
* Mean EM per q: 0.083
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.276
### Time(s) 
* Mean time per q: 4.16s
* Max time per q: 6.13s
* Min time per q: 0s
* Std time per q: 0.48s