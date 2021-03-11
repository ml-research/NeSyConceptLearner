## Set Prediction with Slot Attention on CLEVR

Code for pretraining the set prediction slot attention architecture on the original CLEVR data set. To run this please 
download the original [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) data set and run 
```scripts/clevr-slot-attention.sh``` as:

```./scripts/clevr-slot-attention.sh 0 13 /path/to/CLEVR/```

For running on GPU device 0 with run number 13 (will be stored as slot-attention-clevr-state-13) with your local path 
to the CLEVR directory.