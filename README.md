Poor Man's GPT-3

This is what happens when you don't have $10 million but you want to do machine learning!

How to run:
1. open Poor_Man's_GPT_3_v2.ipynb (should have a GPU instance and pytorch, etc. installed)
2. Pick hyperparameters, datasets, etc. in notebook cells
3. Pick model (all should be imported, so just change when model is loaded)
4. Run! (and see validation ppl get stuck or explode, as per our experience)

Files:
* Poor_Man's_GPT-3.ipynb: original notebook
* Poor_Man's_GPT_3_v2.ipynb: notebook to use for running code
* README.md
* data_utils.py: utils for 3rd dataloader
* dataloader.py: 3rd dataloader used (from Tim Dettmer's repository)
* dataset.py: original dataset (pytorch construct). This was most based
* datasetxl.py: attempt at rewriting 3rd dataloader (prior to copying it in)
* transformer.py: vanilla transformer code
* transformerxl.py: transformerxl + word dropout + relative positional encoding
* transformerxl3.py: vanilla transformer + word dropout
