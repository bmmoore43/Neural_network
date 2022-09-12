# Neural_network
python scripts to building a neural network with text data

## word embedding download
Download word embeddings:

    !wget http://nlp.stanford.edu/data/glove.6B.zip
    
    !unzip -q glove.6B.zip
    
## get data in correct format

  NEED: a .csv file where first column is text data (ie. abstract, or other text to predict) and second column is T.F.UC where data is designated as either true (T) or false (F)

## run neural net script

**NOTE: glove.6B should be in your working directory**

    python neuralnet.py <text_data.csv>
    
    
