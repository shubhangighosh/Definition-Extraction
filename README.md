# Definition-Extraction
Implementation of the word class lattices approach for definition extraction and our proposed changes for improvement.  
The word class lattice approach for definition extraction was implemented. [Link to original paper](http://www.anthology.aclweb.org/P/P10/P10-1134.pdf).  
Following modifications were proposed which led to performance improvements.  
# Modifications for WCL-1
# 1. Pattern Augmentation  
In order to improve the recall of WCL-1, based on
grammatic rules, extra generalized patterns were created for each sentence in
the training set, which was then added to the WCL.  
# 2. Match Based Scoring  
we compute the match score of the test sentence with each of the stored generalized sentences in
the WCL, and classify the sentence as definitional if the maximum score obtained
crosses a threshold which is proportional to the length of the sentence.  
# 3. Coverage-Support based Scoring  
We chose to use a scoring function based on the product of Coverage and Log Support. The sentence was classified as definitional if this score was larger than a threshold.  

# Modifications for WCL-3  
# 1. Pattern Augmentation   
Similar to WCL-1.
# 2. Field-Specific Coverage-Support Score  
Instead of computing a single coverage
score for all 3 fields’ Lattices together, in this model, a coverage-support
score was computed for each field separately, and the maximum score for each
was used. This greedy approach, is more efficient in terms of computation time,
as it doesn’t involve iterating through every WCL multiple times.  

For details and analysis of results, please refer to our [Report](https://github.com/shubhangighosh/Definition-Extraction/blob/master/Report.pdf).
