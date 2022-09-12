# Neural_network
python scripts to building a neural network with text data

## Install Tensorflow and Keras

1. Install Tensorflow (neural network software from google) with a virtual environment:

    create virtual environment:
    
        python3 -m venv --system-site-packages ./venv
        
    activate
    
        source ./venv/bin/activate
        
    install pip and tensorflow
    
        pip install --upgrade pip
        
        pip install --upgrade tensorflow

    verify
    
        python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

2. Install Keras (deep learning and neural network software (python)- runs on top of tensorflow- so need tensorflow in order to run)

        pip install keras
        

## Word embedding download
These are word associations of semantically similar words. So one word can lead to prediction of a different word: i.e. queen / royalty

Download word embeddings:

    !wget http://nlp.stanford.edu/data/glove.6B.zip
    
    !unzip -q glove.6B.zip
    
## Get data in correct format

  NEED: a .csv file where first column is text data (ie. abstract, or other text to predict) and second column is T.F.UC where data is designated as either true (T) or false (F)

## Run neural net script

**NOTE: glove.6B should be in your working directory**

    python neuralnet.py <text_data.csv>
    
    
