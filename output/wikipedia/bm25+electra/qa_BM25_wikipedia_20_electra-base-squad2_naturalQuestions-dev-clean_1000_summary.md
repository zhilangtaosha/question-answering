### Pipeline Parameters:
* Name: <BM25_wikipedia>_20_<electra-base-squad2>_naturalQuestions-dev-clean_1000
* BM25 top K = 20
* Reader top K = 1
* USE_GPU = True
* Number of questions: 17
* Seed = 42
------
### BM25 Retrieval recall 
* BM25 Recall @ 5: 0.647
* BM25 Recall @ 10: 0.647
* BM25 Recall @ 20: 0.706
* BM25 Recall @ 50: 0.706
* BM25 Recall @ 100: 0.706
### BM25 Retrieval precision 
* BM25 Precision @ 5: 0.235
* BM25 Precision @ 10: 0.176
* BM25 Precision @ 20: 0.138
* BM25 Precision @ 50: 0.055
* BM25 Precision @ 100: 0.028
### F1 
* Mean F1 per q: 0.06
* Median F1 per q: 0.0
* Max F1 per q: 0.5
* Min F1 per q: 0
* Std F1 per q: 0.14
### Precision 
* Mean precision per q: 0.061
* Median precision per q: 0.0
* Max precision per q: 0.5
* Min precision per q: 0
* Std precision per q: 0.141
### Recall 
* Mean recall per q: 0.096
* Median recall per q: 0.0
* Max recall per q: 1.0
* Min recall per q: 0
* Std recall per q: 0.256
### Exact Match 
* Mean EM per q: 0.0
* Median EM per q: 0.0
* Max EM per q: 0.0
* Min EM per q: 0.0
* Std EM per q: 0.0
### Time(s) 
* Mean time per q: 8.16s
* Max time per q: 16.33s
* Min time per q: 4.3s
* Std time per q: 3.06s
