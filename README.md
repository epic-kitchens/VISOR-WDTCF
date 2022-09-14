# VISOR WDTCF

## Overview
*Where Did This Come From (WDTCF)?* is a taster benchmark for long-term perception with scene segmentation. Given a frame from an untrimmed video with a mask indicating a query object, the task aims to trace the mask back through time to identify WDTCF, e.g., milk from the fridge, and a plate from a particular cupboard.


For example, where did this (**onion**) come from?

<div align="center">
  <img src='figures/example.png' width='80%' />
</div>

Answer: **Cupboard**

## Statistics
In total, 224 WDTCF examples are obtained from 92 untrimmed videos in train and val sets of VISOR, with 78 unique query objects. The gap between the query and evidence frames is 5.4 mins (19K frames) on average, but it varies widely with a standard deviation of 8 mins (min=1s, max=52 mins). 

<!-- ![image](https://github.com/AndrewZhuZJU/WDTCF/blob/main/figures/rsz_query_distribution.jpg) -->

<div align="center">
<img src='figures/rsz_query_distribution.jpg' width='80%' />
  <img src='figures/distance.png' width='70%' />
</div>

## Annotation Download
The annotations include the following:
* `Query frame ID`: frame to ask the question 'where did this come from?'.
* `Query object`: object of interest in the query frame and 'this' in the question. Note that we assume object name is unknown and only mask of the object is provided as input.
* `Source object`: a finite list of 15 sources from which objects emerge in the kitchen, i.e., [fridge, freezer, cupboard, drawer, oven, dishwasher, bag, bottle, box, package, jar, can, pan, tub, cup].
* `Evidence frame ID`: evidence to infer the answer for the question, i.e, the query object emerging from the source object.

[Download WDTCF_GT.json](https://www.dropbox.com/s/dc15fh9qkkffnhm/WDTCF_GT.json?dl=0)

## Annotation Visulization 


[Download all annotation visulization](https://www.dropbox.com/s/06yk26c2f4zszzg/WDTCF_Vis.zip?dl=0)


## Evaluation Code