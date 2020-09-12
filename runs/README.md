This folder contains evaluation scripts, which run full pipelines to 
answer open domain questions.

Install haystack locally:

```shell script
cd haystack-0.3.0
pip install --editable .
```

### Data Summary
|                     |                              NQ-dev                              |                            QuasarT-dev                           |                             SearchQA-dev                             |                                 SQuAD2-dev                                |                                TriviaQA-dev                               |                                 WikiQA-dev                                 |
|--------------------:|:----------------------------------------------------------------:|:----------------------------------------------------------------:|:--------------------------------------------------------------------:|:-------------------------------------------------------------------------:|:-------------------------------------------------------------------------:|:--------------------------------------------------------------------------:|
|          #questions |                                                            7,830 |                                                            3,000 |                                                               21,613 |                                                                     5,928 |                                                                    11,313 |                                                                        126 |
| #answers p.question |                                                                2 |                                                                1 |                                                                    1 |                                                                       1.7 |                                                   15 (variants / aliases) |                                                                        1.1 |
|  #tokens p.question |                                                                9 |                                                               11 |                                                                   15 |                                                                        10 |                                                                        14 |                                                                          6 |
|    #tokens p.answer |                                                              129 |                                                                2 |                                                                    2 |                                                                       3.6 |                                                                       2.3 |                                                                         26 |
|          5W1H dist. | who=25%, what=16%, why=1%, when13%, where=11%, how=6% others=29% | who=11%, what=31%, why=0%, when=1%, where=2%, how=2%, others=53% | who=0%, what=0.1%, why=0%, when=0.6%, where=0%, how=0.1%, others=99% | who=8%, what=46%, why=1.6%, when=6.4%, where=3.8%, how=9.7%, others=24.3% | who=9.7%, what=20.3%, why=0.1%, when=0.4%, where=1%, how=1.4%, others=67% | who=11.9%, what=57.9%, why=0%, when=8.7%, where=13.5%, how=7.9%, others=0% |


### Performance Summary

|                      seed=42, excl.unanswerable |       NQ-dev (696)      |   QuasarT-dev (1000)   |   SearchQA-dev (1000)  |    SQuAD2-dev (1000)   |   TriviaQA-dev (1000)  |      WikiQA-dev (126)     |
|------------------------------------------------:|:-----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-------------------------:|
|                      BM25para  Retrieval Recall | @5=.24 @10=.31 @20=.38  | @5=.40 @10=.48 @20=.54 | @5=.54 @10=.62 @20=.68 | @5=.50 @10=.57 @20=.63 | @5=.61 @10=.67 @20=.71 | @5=.01 @10=.02 @20=.04    |
|      BM25para + DPR (top 1000) Retrieval Recall |                         | @5=.55 @10=.62 @20=.65 |                        | @5=.42 @10=.52 @20=.61 |                        |                           |
|                   BM25para  Retrieval Precision | @5=.09 @10=.08 @20=.07  | @5=.18 @10=.15 @20=.13 | @5=.25 @10=.21 @20=.17 | @5=.17 @10=.13 @20=.10 | @5=.32 @10=.26 @20=.22 | @5=.002 @10=.002 @20=.002 |
|  BM25para + DPR (top 1000)  Retrieval Precision |                         | @5=.29 @10=.26 @20=.22 |                        | @5=.16 @10=.14 @20=.12 |                        |                           |
|                  BM25para + Electra (top 20) F1 | mean=.168 std=.311      | mean=.236 std=.397     | mean=.142 std=.321     | mean=.358 std=.443     | mean=.424 std=.462     | mean=.087 std=.109        |
| BM25para + DPR (top 1000) + Electra (top 20) F1 |                         | mean=.278 std=.417     |                        | mean=.302 std=.421     |                        |                           |
|                  BM25para + Electra (top 20) EM | mean=.093 std=.291      | mean=.185 std=.388     | mean=.095 std=.293     | mean=.281 std=.449     | mean=.345 std=.475     | mean=0 std=.0             |
| BM25para + DPR (top 1000) + Electra (top 20) EM |                         | mean=.211 std=.408     |                        | mean=.226 std=.418     |                        |                           |
