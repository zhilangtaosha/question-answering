### Performance Summary
This folder contains evaluation scripts, which run full pipelines to 
answer open domain questions.

#### wikipedia_paragraph

* number of tokens per paragraph:
    - mean = 61.1
    - std = 52.6
    - min = 1
    - max = 5503
    - median = 48

* Retrieval

|                      seed=42, excl.unanswerable |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25para  Retrieval Recall | @5=.33 @10=.40 @20=.49  | @5=.40 @10=.48 @20=.54 | @5=.54 @10=.62 @20=.68 |**@5=.50 @10=.57 @20=.63**| @5=.61 @10=.67 @20=.71 | @5=.01 @10=.02 @20=.04    |
|      BM25para + DPR (top 1000) Retrieval Recall |**@5=.55 @10=.62 @20=.67**|**@5=.55 @10=.62 @20=.65**|**@5=.64 @10=.71 @20=.76**|@5=.47 @10=.56 @20=.65|**@5=.73 @10=.77 @20=.80**|**@5=.09 @10=.11 @20=.11**|
|                   BM25para  Retrieval Precision | @5=.12 @10=.10 @20=.09  | @5=.18 @10=.15 @20=.13 | @5=.25 @10=.21 @20=.17 | @5=.17 @10=.13 @20=.10 | @5=.32 @10=.26 @20=.22 | @5=.002 @10=.002 @20=.002 |
|  BM25para + DPR (top 1000)  Retrieval Precision |**@5=.24 @10=.19 @20=.16**|**@5=.29 @10=.26 @20=.22**|**@5=.34 @10=.29 @20=.25**|**@5=.18 @10=.16 @20=.13**|**@5=.44 @10=.38 @20=.33**|**@5=.02 @10=.01 @20=.007**|


* Reading

|                      seed=42, excl.unanswerable |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25para + Electra (top 20) F1 | mean=.192 std=.335      | mean=.236 std=.397     | mean=.142 std=.321     |**mean=.358 std=.443**  | mean=.424 std=.462     | mean=.087 std=.109        |
| BM25para + DPR (top 1000) + Electra (top 20) F1 |**mean=.236 std=.365**   |**mean=.278 std=.417**  |**mean=.188 std=.357**  | mean=.319 std=.427     |**mean=.457 std=.462**  |**mean=.095 std=.106**     |
|                  BM25para + Electra (top 20) EM | mean=.105 std=.307      | mean=.185 std=.388     | mean=.095 std=.293     |**mean=.281 std=.449**  | mean=.345 std=.475     | mean=0 std=.0             |
| BM25para + DPR (top 1000) + Electra (top 20) EM |**mean=.136 std=.343**   |**mean=.211 std=.408**  |**mean=.123 std=.328**  | mean=.243 std=.429     |**mean=.368 std=.482**  | mean=0 std=.0             |


#### wikipedia_100_stride_50
* Retrieval

|                      seed=42, excl.unanswerable |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25para  Retrieval Recall | @5=.33 @10=.40 @20=.49  | @5=.40 @10=.48 @20=.54 | @5=.54 @10=.62 @20=.68 |**@5=.50 @10=.57 @20=.63**| @5=.61 @10=.67 @20=.71 | @5=.01 @10=.02 @20=.04    |
|      BM25para + DPR (top 1000) Retrieval Recall |                         |                        |                        |                        |                        |                           |
|                   BM25para  Retrieval Precision | @5=.12 @10=.10 @20=.09  | @5=.18 @10=.15 @20=.13 | @5=.25 @10=.21 @20=.17 | @5=.17 @10=.13 @20=.10 | @5=.32 @10=.26 @20=.22 | @5=.002 @10=.002 @20=.002 |
|  BM25para + DPR (top 1000)  Retrieval Precision |                         |                        |                        |                        |                        |                           |


* Reading

|                      seed=42, excl.unanswerable |       NQ-dev (1000)     |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                  BM25para + Electra (top 20) F1 | mean=.192 std=.335      | mean=.236 std=.397     | mean=.142 std=.321     |**mean=.358 std=.443**  | mean=.424 std=.462     | mean=.087 std=.109        |
| BM25para + DPR (top 1000) + Electra (top 20) F1 |**mean=.236 std=.365**   |**mean=.278 std=.417**  |**mean=.188 std=.357**  | mean=.319 std=.427     |**mean=.457 std=.462**  |**mean=.095 std=.106**     |
|                  BM25para + Electra (top 20) EM | mean=.105 std=.307      | mean=.185 std=.388     | mean=.095 std=.293     |**mean=.281 std=.449**  | mean=.345 std=.475     | mean=0 std=.0             |
| BM25para + DPR (top 1000) + Electra (top 20) EM |**mean=.136 std=.343**   |**mean=.211 std=.408**  |**mean=.123 std=.328**  | mean=.243 std=.429     |**mean=.368 std=.482**  | mean=0 std=.0             |




### Data Summary
|  excl.unanswerables |                                NQ-dev (excl. long answers)                               |                                        QuasarT-dev                                        |                                  SearchQA-dev                                  |                                     SQuAD2-dev                                    |                                       TriviaQA-dev                                       |                                      WikiQA-dev                                      |
|--------------------:|:----------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
|          #questions |                                                                                    4,201 |                                                                                     3,000 |                                                                         21,613 |                                                                             5,928 |                                                                                   11,313 |                                                                                  126 |
| #answers p.question |                                                                                        2 |                                                                                         1 |                                                                              1 |                                                                               1.7 |                                                                  15 (variants / aliases) |                                                                                  1.1 |
|  #tokens p.question |                                                                                        9 |                                                                                        11 |                                                                             15 |                                                                                10 |                                                                                       14 |                                                                                    6 |
|    #tokens p.answer |                                                                                      4.2 |                                                                                         2 |                                                                              2 |                                                                               3.6 |                                                                                      2.3 |                                                                                   26 |
|                6W1H | who=33.2%, what=16.5%, why=0.5%, when17.1%, where=12.6%, which=2%, how=5.0% others=15.2% | who=10.7%, what=32.4%, why=0.1%, when=0.8%, where=2.1%, which=12.8%, how=1.8%, others=53% | who=0%, what=0.1%, why=0%, when=0.6%, where=0%, which=0%, how=0.1%, others=99% | who=8%, what=49%, why=1.6%, when=6.4%, where=3.8%, which=4%, how=9.7%, others=17% | who=9.7%, what=21.2%, why=0.1%, when=0.4%, where=1%, which=22.6%, how=1.4%, others=43.7% | who=11.9%, what=57.9%, why=0%, when=8.7%, where=13.5%, which=0%, how=7.9%, others=0% |

Remarks:
* NQ (NaturalQuestions): real web queries accompanied with wikipedia articles, 
from which people annotate short/long answers, or yes/no answers.
    - "who wrote the song photograph by ringo starr"
    - "when was the first hunger games book published"
    - "to aru kagaku no railgun s episode 3"
    - "dominant alleles are always the most common allele in a population"
    - "original cast of natasha pierre and the great comet of 1812"
* QuasarT: factoid/trivia questions
    - "Lockjaw is another name for which disease"
    - "Which vegetable is a Welsh emblem ?"
    - "Where did judy garland 's family have their vaudeville act"
    - "What nationality was Oddjob"
* SearchQA: Jeopardy question-answer pairs (factoid/trivia, but "none-questions")
    - "The bestselling passenger car of all time is this company's Corolla"
    - "Like a door, a Broadway show does these 2 things"
    - "One of the 2 main types of chemical bonds"
* SQuAD2: Questions are created by people after given a Wikipedia article
    - "What German ruler invited Huguenot immigration?"
    - "What percentage of electrical power in the United States is made by steam turbines?"
    - "What do nuclear power plants heat to create electricity?"
* TriviaQA: factoid/trivia questions
    - "Which Lloyd Webber musical premiered in the US on 10th December 1993?"
    - "In Greek mythology, who were Arges, Brontes and Steropes?"
    - "Who first hosted Family Feud?"
* WikiQA: questions sampled from Bing query logs (factoid, WH-questions, but with long answers)
    - "how big is bmc software in houston?"
    - "what does a plus-minus sign mean"
    - "what kind of literature did john steinbeck writing"
    
    
### Installation notes:
Install haystack locally:

```shell script
cd haystack-0.3.0
pip install --editable .
```
