# TypeNet
This is the official repo for TypeNet, a hierarchical type system for the task of fine grained entity typing. It contains 1081 freebase types, and 860 Wordnet types organised in a deep hierarchy with an average depth of 7.8. 

![GitHub Logo](typenet_image.png)

# Instructions for Using the Dataset

To generate a joblib dump that creates a transitive closure of the TypeNet dag, run ``` python process_taxonomy.py ```. This generates 2 joblibs **TypeNet_transitive_closure.joblib** and **TypeNet_type2idx.joblib**. The first joblib is an adjacency matrix of size 1941 x 1941, and the second joblib maps TypeNet types to their ids. 

The accompanying paper may be found [here!](http://arxiv.org)


