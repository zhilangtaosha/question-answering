### Pipeline Parameters:
* Name: BM25_20_electra-base-squad2_triviaQA-dev_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 1000
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.607
* BM25 Recall @ 10: 0.667
* BM25 Recall @ 20: 0.708
* BM25 Recall @ 50: 0.708
* BM25 Recall @ 100: 0.708
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.317
* BM25 Precision @ 10: 0.264
* BM25 Precision @ 20: 0.216
* BM25 Precision @ 50: 0.087
* BM25 Precision @ 100: 0.043
### F1 
* Mean F1 per q: 0.424
* Median F1 per q: 0.0
* Max F1 per q: 1.0
* Min F1 per q: 0
* Std F1 per q: 0.462
### Precision 
* Mean precision per q: 0.432
* Median precision per q: 0.0
* Max precision per q: 1.0
* Min precision per q: 0
* Std precision per q: 0.471
### Recall 
* Mean recall per q: 0.435
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.473
### Exact Match 
* Mean EM per q: 0.345
* Median EM per q: 0.0
* Max EM per q: 1.0
* Min EM per q: 0.0
* Std EM per q: 0.475
### Time(s) 
* Mean time per q: 6.34s
* Max time per q: 12.32s
* Min time per q: 4.81s
* Std time per q: 0.61s