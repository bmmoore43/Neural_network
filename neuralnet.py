## neural network models
# import packages
### increase epochs for nn word embed
from keras.models import Sequential
from keras import layers
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import operator
import sys, os

def list2str(list1):
	res=[]
	for ele in list1:
		ele= "".join(str(ele))
		ele= ele.replace("[","")
		ele= ele.replace("]","")
		res.append(ele)
		
	return(res)

#import data
filepath_dict = {'papers':sys.argv[1]}
df_list=[]
for source, filepath in filepath_dict.items():
	#df = pd.read_csv(filepath, names=['entry','abstract', 'label'], sep='\t')
	df = pd.read_csv(filepath, sep=',')
	df['source'] = source
	df_list.append(df)
# add to dataframe
df = pd.concat(df_list)
print(df.iloc[0])
df_pap = df[df['source'] == 'papers']
if 'T.F.UC' in df_pap.columns:
	df_pap['T.F.UC']= df_pap['T.F.UC'].replace('T', 1)
	df_pap['T.F.UC']= df_pap['T.F.UC'].replace('F', 0)
elif "status" in df_pap.columns:
	df_pap['status']= df_pap['status'].replace('TRUE', 1)
	df_pap['status']= df_pap['status'].replace('FALSE', 0)
else:
	pass
print(df_pap.dtypes)
#sentences = df_pap['abstract'].values
sentences = df_pap.iloc[:,0].values
#y = df_pap['label'].values
y = df_pap.iloc[:,1].values

# vectorize to get unique words in sentence
vectorizer = CountVectorizer()
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences)
vectorizer.vocabulary_
vectorizer.transform(sentences).toarray()
# split into train/test
from sklearn.model_selection import train_test_split
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
input_dim = X_train.shape[1]
# logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score1= classifier.score(X_train, y_train)
score = classifier.score(X_test, y_test)
print("LogReg Accuracy (train,test):", score1, score)
mlD= {'classifier': 'LogReg', 'train_acc': score1, 'test_acc': score, 'train_loss': 'NA', "test_loss": "NA"}
df_MLresult = pd.DataFrame(mlD, index=[0])
# NN model set up
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print('first NN',model.summary())
# run model and print accuracy and loss 
## add validation_split=0.1 to keep out 10% of training data each epoch to avoid overfit
history = model.fit(X_train, y_train,epochs=50,verbose=False, validation_data=(X_test, y_test),batch_size=10)
loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
newdata= {'classifier':'NN', 'train_acc':accuracy1, 'test_acc':accuracy2, 'train_loss':loss1, "test_loss":loss2}
df_MLresult = df_MLresult.append(newdata, ignore_index=True)
D_acc_mod= {'NN':accuracy2}
#predictions
predicted= model.predict(X_test)
predicted_c= model.predict_classes(X_test)
predicted= predicted.tolist()
predicted_c= predicted_c.tolist()
predicted= list2str(predicted)
predicted_c= list2str(predicted_c)
print(predicted,predicted_c)
df_nn= pd.DataFrame({'abstract':sentences_test, 'class':y_test, 'prediction_score':predicted, 'predicted_class':predicted_c})
print(df_nn.head())
D_df_mod= {'NN':df_nn}
# make accuracy plots
import matplotlib.pyplot as plt
# plot function
def plot_history(history):
     acc = history.history['accuracy']
     val_acc = history.history['val_accuracy']
     val_loss = history.history['val_loss']
     loss = history.history['loss']
     x = range(1, len(acc) + 1)
     plt.figure(figsize=(12, 5))
     plt.subplot(1, 2, 1)
     plt.plot(x, acc, 'b', label='Training acc')
     plt.plot(x, val_acc, 'r', label='Validation acc')
     plt.title('Training and validation accuracy')
     plt.legend()
     plt.subplot(1, 2, 2)
     plt.plot(x, loss, 'b', label='Training loss')
     plt.plot(x, val_loss, 'r', label='Validation loss')
     plt.title('Training and validation loss')
     plt.legend()
# write to pdf     
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(sys.argv[1]+'_NN.pdf') as pdf:
     mp = plot_history(history)
     pdf.savefig(mp)


# get vocab layers
#use one hot encode
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
#word embeddings
#tokenize the data into a format that can be used by the word embeddings
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)
X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

## One problem that we have is that each text sequence 
## has in most cases different length of words. To counter this, you can use pad_sequence()
# pad sequences
from keras.preprocessing.sequence import pad_sequences
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(X_train[0, :])

## add embedding layers to model
from keras.models import Sequential
from keras import layers
embedding_dim = 50

model = Sequential()

# layers: input_dim: the size of the vocabulary,output_dim: the size of the dense vector,input_length: the length of the sequence
model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# summarize model
print("NN layers",model.summary())

# retrain model
history = model.fit(X_train, y_train,epochs=50,verbose=False,validation_data=(X_test, y_test),batch_size=10)
# get accuracy
loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
D_acc_mod["NN_layers"]= accuracy2
newdata= {'classifier':'NN layers', 'train_acc':accuracy1, 'test_acc':accuracy2, 'train_loss':loss1, "test_loss":loss2}
df_MLresult = df_MLresult.append(newdata, ignore_index=True)

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(sys.argv[1]+'_NN_layers.pdf') as pdf:
     mp = plot_history(history)
     pdf.savefig(mp)
#predictions
predicted= model.predict(X_test)
predicted_c= model.predict_classes(X_test)
predicted= predicted.tolist()
predicted_c= predicted_c.tolist()
predicted= list2str(predicted)
predicted_c= list2str(predicted_c)

df_nn_l= pd.DataFrame({'abstract':sentences_test, 'class':y_test, 'prediction_score':predicted, 'predicted_class':predicted_c})
print(df_nn_l.head())
D_df_mod["NN_layers"]= df_nn_l



## Global max/average pooling     
# pooling layers as a way to downsample incomoing features
# get embedding layers

embedding_dim = 50

model = Sequential()
# add layers to model
model.add(layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=maxlen))
# add max pooling
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# make model and summary
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
print("NN maxpool",model.summary())

# retrain model
history = model.fit(X_train, y_train,epochs=50,verbose=False,validation_data=(X_test, y_test),batch_size=10)
loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
D_acc_mod["NN_maxpool"]= accuracy2

# replot new model
with PdfPages(sys.argv[1]+'_NN_maxpool.pdf') as pdf:
     mp = plot_history(history)
     pdf.savefig(mp)
#
predicted= model.predict(X_test)
predicted_c= model.predict_classes(X_test)
predicted= predicted.tolist()
predicted_c= predicted_c.tolist()
predicted= list2str(predicted)
predicted_c= list2str(predicted_c)

df_nn_pool= pd.DataFrame({'abstract':sentences_test, 'class':y_test, 'prediction_score':predicted, 'predicted_class':predicted_c})
print(df_nn_pool.head())
D_df_mod["NN_maxpool"]= df_nn_pool

newdata= {'classifier':'NN maxpool', 'train_acc':accuracy1, 'test_acc':accuracy2, 'train_loss':loss1, "test_loss":loss2}
df_MLresult = df_MLresult.append(newdata, ignore_index=True)
# pre train word embeddings

# get pre trained matrix
import numpy as np

# function to create embedding matrix
def create_embedding_matrix(filepath, word_index, embedding_dim):
	vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
	embedding_matrix = np.zeros((vocab_size, embedding_dim))
	with open(filepath) as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word] 
				embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
	return embedding_matrix

# read in pre trained matrix
embedding_dim = 50
embedding_matrix = create_embedding_matrix('glove.6B/glove.6B.50d.txt',tokenizer.word_index, embedding_dim)

# check out matrix- how many nonzero elements are there?
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / vocab_size

# build model with pre-trained embedding matrix
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
print("NN maxpool wordemb",model.summary())
# add in data to train your model
history = model.fit(X_train, y_train,epochs=50,verbose=False,validation_data=(X_test, y_test),batch_size=10)
loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
D_acc_mod["NN_maxpool_wb"]= accuracy2
# replot new model
with PdfPages(sys.argv[1]+'_NN_maxpool_wordem.pdf') as pdf:
     mp = plot_history(history)
     pdf.savefig(mp)

predicted= model.predict(X_test)
predicted_c= model.predict_classes(X_test)
predicted= predicted.tolist()
predicted_c= predicted_c.tolist()
predicted= list2str(predicted)
predicted_c= list2str(predicted_c)

df_nn_pool_wb= pd.DataFrame({'abstract':sentences_test, 'class':y_test, 'prediction_score':predicted, 'predicted_class':predicted_c})
D_df_mod["NN_maxpool_wb"]= df_nn_pool_wb
print(df_nn_pool_wb.head())

newdata= {'classifier':'NN max pool word embed', 'train_acc':accuracy1, 'test_acc':accuracy2, 'train_loss':loss1, "test_loss":loss2}
df_MLresult = df_MLresult.append(newdata, ignore_index=True)
# build model with pre-trained word embedding and additional embed training while building model
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print("NN wordemb trained",model.summary())
# train new model
history = model.fit(X_train, y_train, epochs=50, verbose=False, validation_split=0.1, validation_data=(X_test, y_test),batch_size=10)
loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
D_acc_mod["NN_maxpool_wb_tr"]= accuracy2
# graph
with PdfPages(sys.argv[1]+'_NN_maxpool_wordem_trained.pdf') as pdf:
     mp = plot_history(history)
     pdf.savefig(mp)

predicted= model.predict(X_test)
predicted_c= model.predict_classes(X_test)
predicted= predicted.tolist()
predicted_c= predicted_c.tolist()
predicted= list2str(predicted)
predicted_c= list2str(predicted_c)

df_nn_pool_wbtr= pd.DataFrame({'abstract':sentences_test, 'class':y_test, 'prediction_score':predicted, 'predicted_class':predicted_c})
print(df_nn_pool_wbtr.head())
D_df_mod["NN_maxpool_wb_tr"]= df_nn_pool_wbtr
newdata= {'classifier':'NN max pool word embed tr', 'train_acc':accuracy1, 'test_acc':accuracy2, 'train_loss':loss1, "test_loss":loss2}
df_MLresult = df_MLresult.append(newdata, ignore_index=True)
# convolutional neural networks (cnn)
# cnn model
embedding_dim = 100

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
print("CNN max pool",model.summary())
# train cnn model
history = model.fit(X_train, y_train,epochs=50,verbose=False,validation_data=(X_test, y_test),batch_size=10)

loss1, accuracy1 = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss2, accuracy2 = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
D_acc_mod["CNN"]= accuracy2
# plot
with PdfPages(sys.argv[1]+'_CNN_maxpool.pdf') as pdf:
     mp = plot_history(history)
     pdf.savefig(mp)

predicted= model.predict(X_test)
predicted_c= model.predict_classes(X_test)
predicted= predicted.tolist()
predicted_c= predicted_c.tolist()
predicted= list2str(predicted)
predicted_c= list2str(predicted_c)

df_cnn = pd.DataFrame({'abstract':sentences_test, 'class':y_test, 'prediction_score':predicted, 'predicted_class':predicted_c})
print(df_cnn.head())
D_df_mod["CNN"]= df_cnn

newdata= {'classifier':'CNN max pool', 'train_acc':accuracy1, 'test_acc':accuracy2, 'train_loss':loss1, "test_loss":loss2}
df_MLresult = df_MLresult.append(newdata, ignore_index=True)

# build model with k-fold cross-validation
# create model function with cross-validation
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
# define the parameter grid 
param_grid = dict(num_filters=[32, 64, 128],kernel_size=[3, 5, 7],vocab_size=[5000], embedding_dim=[50],maxlen=[100])

# add data in to model and run model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# Main settings
epochs = 50
embedding_dim = 50
maxlen = 100
output_file = 'data/'+ sys.argv[1]+'_cnn-output.txt'

# Run grid search for each source with 4-fold CV
for source, frame in df.groupby('source'):
    print('Running grid search for data set :', source)
    df_pap = df[df['source'] == 'papers']
    if 'T.F.UC' in df_pap.columns:
         df_pap['T.F.UC']= df_pap['T.F.UC'].replace('T', 1)
         df_pap['T.F.UC']= df_pap['T.F.UC'].replace('F', 0)
    elif "status" in df_pap.columns:
         df_pap['status']= df_pap['status'].replace('TRUE', 1)
         df_pap['status']= df_pap['status'].replace('FALSE', 0)
    else:
         pass
    #sentences = df_pap['abstract'].values
    sentences = df_pap.iloc[:,0].values
    #y = df_pap['label'].values
    y = df_pap.iloc[:,1].values
    # Train-test split
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
    # Tokenize words
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)
    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1
    # Pad sequences with zeros
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    # Parameter grid for grid search
    param_grid = dict(num_filters=[32, 64, 128],kernel_size=[3, 5, 7],vocab_size=[vocab_size],embedding_dim=[embedding_dim],maxlen=[maxlen])
    model = KerasClassifier(build_fn=create_model,epochs=epochs, batch_size=10,verbose=False)
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=4, verbose=1, n_iter=5) # cv=4 is 4 cross validations
    grid_result = grid.fit(X_train, y_train)
    # Evaluate testing set
    test_accuracy = grid.score(X_test, y_test)
    # Save and evaluate results
#     prompt = input(f'finished {source}; write to file and proceed? [y/n]')
#     if prompt.lower() not in {'y', 'true', 'yes'}:
#         break
    with open(output_file, 'a') as f:
        s = ('Running {} data set\nBest Accuracy : {:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
        output_string = s.format(source,grid_result.best_score_,grid_result.best_params_,test_accuracy)
        print(output_string)
        f.write(output_string)

#
# get model based on best grid search with embedding matrix
def create_model2(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, embedding_matrix):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen, weights=[embedding_matrix],trainable=True))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    return model   
    
print(grid_result.best_params_['num_filters'])
modelx= create_model2(grid_result.best_params_['num_filters'], grid_result.best_params_['kernel_size'], grid_result.best_params_['vocab_size'], grid_result.best_params_['embedding_dim'], grid_result.best_params_['maxlen'], embedding_matrix)
print("CNN with GS and wordemb",modelx.summary())
#run model on training data
history = modelx.fit(X_train, y_train, epochs=epochs, verbose=False,validation_data=(X_test, y_test),batch_size=10)
loss1, accuracy1 = modelx.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy1))
loss2, accuracy2 = modelx.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy2))
D_acc_mod["CNN_GS_wb"]= accuracy2

with PdfPages(sys.argv[1]+'_CNN_maxpool_GS_wordembed.pdf') as pdf:
     mp = plot_history(history)
     pdf.savefig(mp)

predicted= modelx.predict(X_test)
predicted_c= modelx.predict_classes(X_test)
predicted= predicted.tolist()
predicted_c= predicted_c.tolist()
predicted= list2str(predicted)
predicted_c= list2str(predicted_c)

df_cnn_gs = pd.DataFrame({'abstract':sentences_test, 'class':y_test, 'prediction_score':predicted, 'predicted_class':predicted_c})
#df_cnn_gs.to_csv(sys.argv[1]+'_CNN_maxpool_randomGS_wordembed.csv', sep=",")
D_df_mod["CNN_GS_wb"]= df_cnn_gs

newdata= {'classifier':'CNN with GS and wordemb', 'train_acc':accuracy1, 'test_acc':accuracy2, 'train_loss':loss1, "test_loss":loss2}
df_MLresult = df_MLresult.append(newdata, ignore_index=True)

df_MLresult.to_csv(sys.argv[1]+"_ML_results.txt", sep="\t")

##find model with highest test accuracy and output sample predictions
key1=max(D_acc_mod.items(), key=operator.itemgetter(1))[0] # model with max accuracy
print(key1)
df_predict= D_df_mod[key1] # sample predictions from that model
df_predict.to_csv(sys.argv[1]+"_"+str(key1)+"_predictions.txt", sep="\t")
# vocab_size=2990 
# num_filters=32 
# maxlen=100 
# kernel_size=3 
# embedding_dim=50
# modelx= create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen)
# print("CNN gridsearch",modelx.summary())
# run model on training data
# history= modelx.fit(X_train, y_train, epochs=epochs,verbose=False,validation_data=(X_test, y_test),batch_size=10)
# 
# loss, accuracy = modelx.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = modelx.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# with PdfPages('1400entries_50_50_ML_ML_CNN_maxpool_randomGS.pdf') as pdf:
#      mp = plot_history(history)
#      pdf.savefig(mp)
# 

#      
# try improving modelby increasing epochs for embedded word layers
# 
# using trained embedded layets
# def create_model4(vocab_size, embedding_dim, maxlen, embedding_matrix):
#     model = Sequential()
#     model.add(layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],input_length=maxlen,trainable=True))
#     model.add(layers.GlobalMaxPooling1D())
#     model.add(layers.Dense(10, activation='relu'))
#     model.add(layers.Dense(1, activation='sigmoid'))
#     model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
#     return model
# 
# modelx= create_model4(vocab_size, embedding_dim, maxlen, embedding_matrix)
# print("increase epochs to 100",modelx.summary())
# run model on training data
# history = modelx.fit(X_train, y_train,
#                     epochs=100,
#                     verbose=False,
#                     validation_data=(X_test, y_test),
#                     batch_size=10)
# loss, accuracy = modelx.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = modelx.evaluate(X_test, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# with PdfPages('1400entries_50_50_ML_NN_maxpool_wordembed_100epochs_tr.pdf') as pdf:
#      mp = plot_history(history)
#      pdf.savefig(mp)
     
# retrieve history
#for i in range(readline.get_current_history_length()):
#	print (readline.get_history_item(i + 1))
#plt.close('all')