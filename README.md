> **If you use the resources (algorithm, code and dataset) presented in this repository, please cite our paper.**  
*The BibTeX entry is provided at the bottom of this page. 

# Optimal time window for temporal segmentation of sensor streams in multi-activity recognition
Multi-activity recognition in the urban environment is a challenging task. This is largely attributed to the influence of urban dynamics, the variety of the label sets, and the heterogeneous nature of sensor data that arrive irregularly and at different rates. One of the first tasks in multi-activity recognition is temporal segmentation. A common temporal segmentation method is the sliding window approach with a fixed window size, which is widely used for single activity recognition. In order to recognise multiple activities from heterogeneous sensor streams, we propose a new time windowing technique that can optimally extract segments with multiple activity labels. The mixture of activity labels causes the impurity in the corresponding temporal segment. Hence, larger window size imposes higher impurity in temporal segments while increasing class separability. In addition, the combination of labels from multiple activity label sets (i.e. number of unique multi-activity) may decrease as impurity increases. Naturally, these factors will affect the performance of classification task. In our proposed technique, the optimal window size is found by gaining the balance between minimising impurity and maximising class separability in temporal segments. As a result, it accelerates the learning process for recognising multiple activities (such as higher level and atomic human activities under different environment contexts) in comparison to laborious tasks of sensitivity analysis. The evaluation was validated by experiments on a real-world dataset for recognising multiple human activities in a smart environment. 

This repository contains resources developed within the following paper:

	Liono, J., Qin, A. K., & Salim, F. D. (2016, November). Optimal time window for temporal segmentation of sensor streams in multi-activity recognition. 
	In Proceedings of the 13th International Conference on Mobile and Ubiquitous Systems: Computing, Networking and Services (pp. 10-19). ACM. 

You can find the [paper](https://github.com/cruiseresearchgroup/OPTWIN/blob/master/paper/liono2016optwin.pdf) and [presentation](https://github.com/cruiseresearchgroup/OPTWIN/blob/master/presentation/Mobiquitous2016.pdf) in this repository. 

Alternative link: https://dl.acm.org/citation.cfm?id=2994388

## Contents of the repository
This repository contains resources used and described in the paper.

The repository is structured as follows:

- `code/`: Code implementation and evaluation of OPTWIN, experimented using OPPORTUNITY dataset. 
- `data/`: Preprocessed dataset used for this paper. 
- `paper/`: Formal description of the algorithm and evaluation result. 
- `presentation/`: PDF of paper presentation from MobiQuitous 2016. 

## OPPORTUNITY Dataset
The original OPPORTUNITY dataset is sourced from [here](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition). 

## Possible Applications

## Citation
If you use the resources presented in this repository, please cite (using the following BibTeX entry):
```
@inproceedings{liono2016optwin,
 author = {Liono, Jonathan and Qin, A. K. and Salim, Flora D.},
 title = {Optimal Time Window for Temporal Segmentation of Sensor Streams in Multi-activity Recognition},
 booktitle = {Proceedings of the 13th International Conference on Mobile and Ubiquitous Systems: Computing, Networking and Services},
 series = {MOBIQUITOUS 2016},
 year = {2016},
 isbn = {978-1-4503-4750-1},
 location = {Hiroshima, Japan},
 pages = {10--19},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/2994374.2994388},
 doi = {10.1145/2994374.2994388},
 acmid = {2994388},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {data stream processing, multi-activity recognition, multi-objective function, multivariate sensor streams, optimal window size, temporal segmentation},
} 
```
