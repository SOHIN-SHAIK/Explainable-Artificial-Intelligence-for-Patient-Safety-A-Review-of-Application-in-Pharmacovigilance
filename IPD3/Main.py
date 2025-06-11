from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
import shap #loading SHAP tool for XAI based explanation
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from keras_dgl.layers import GraphCNN #loading GNN class
import keras.backend as K
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
import os
from keras.callbacks import ModelCheckpoint
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics 
import seaborn as sns

main = tkinter.Tk()
main.title("Explainable Artificial Intelligence for Patient Safety: A Review of Application in Pharmacovigilance") #designing main screen
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca,GraphCNN,y_train1,y_test1
global accuracy, precision, recall, fscore, values,cnn_model,label_encoder,cnn_model,xgc
precision = []
recall = []
fscore = []
accuracy = []

def uploadDataset():
    global filename, dataset, labels, values,GraphCNN, y_train1,y_test1,label_encoder,cnn_model,xgc
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    labels = np.unique(dataset['Drug'])
    label = dataset.groupby('Drug').size()
    label.plot(kind="bar", figsize=(4,3))
    plt.xlabel('Drug Type')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=90)
    plt.title("Drug Graph")
    plt.show()
def preprocessing():
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca,GraphCNN,y_train1,y_test1,label_encoder,cnn_model,xgc
    text.delete('1.0', END)
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
            label_encoder.append([columns[i], le])
    dataset.fillna(0, inplace = True)
    dataset
    Y = dataset['Drug'].ravel()
    dataset.drop(['Drug'], axis = 1,inplace=True)
    X = dataset.values
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)#shuffling dataset values
    X = X[indices]
    Y = Y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)#features normalization
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4)
    text.insert(END,"Total records found in dataset = "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset= "+str(X.shape[1])+"\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n")
    cm = metrics.confusion_matrix(testY,predict)
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(cm,xticklabels=labels,yticklabels=labels,annot=True,cmap="viridis",fmt="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

def xgboost():
    global X_train, y_train, X_test, y_test,GraphCNN,y_train1,y_test1
    global accuracy, precision, recall, fscore,label_encoder,cnn_model,xgc
    text.delete('1.0', END)
    #training XGBOOST algorithm on 80% training data and then predicting on 20% test data
    xgc = XGBClassifier(n_estimators=2)
    xgc.fit(X_train, y_train)
    #performing prediction on test data
    predict = xgc.predict(X_test)
    #calling function to calculate accuracy on predicted data
    calculateMetrics("XGBoost", y_test, predict)

def graphcnn():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    global GraphCNN,cnn_model,xgc
    text.delete('1.0', END)
    #training graphCNN algorithm
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #Create GNN model to detect fault from all services
    graph_conv_filters = np.eye(1)
    graph_conv_filters = K.constant(graph_conv_filters)
    graph_model = Sequential()
    graph_model.add(GraphCNN(128, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    graph_model.add(GraphCNN(64, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    graph_model.add(GraphCNN(1, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    graph_model.add(Dense(units = 256, activation = 'elu'))
    graph_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    graph_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/gcnn_weights.h5") == False:
        hist = graph_model.fit(X_train, y_train1, batch_size=1, epochs=50, validation_data = (X_test, y_test1), verbose=1)
        graph_model.save_weights("model/gcnn_weights.h5")
    else:
        graph_model.load_weights("model/gcnn_weights.h5")
    #perform prediction on test data of all services and calculate accuracy and other metrics
    pred = []
    for i in range(len(X_test)):
        temp = []
        temp.append(X_test[i])
        temp = np.asarray(temp)
        predict = graph_model.predict(temp, batch_size=1)
        predict = np.argmax(predict)
        pred.append(predict)
    y_tested = np.argmax(y_test1, axis=1)    
    predict = np.asarray(pred)
    #calling function to calculate accuracy on predicted data
    calculateMetrics("Graph Model", y_tested, predict)

def MLP():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    global y_train1,y_test1,label_encoder,GraphCNN,cnn_model,xgc
    text.delete('1.0', END)
    #training neural network algorithm
    nn = MLPClassifier(max_iter=800)
    nn.fit(X_train, y_train)
    predict = nn.predict(X_test)
    #calling function to calculate accuracy on predicted data
    calculateMetrics("Neural Network", y_test, predict)

def CNN():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    global y_train1, y_test1,label_encoder,GraphCNN,cnn_model,xgc
    text.delete('1.0', END)
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    #training extension Convolution Neural Network 2D algortihm as extension
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    cnn_model = Sequential()
    #creating cnn2d layer with 32 neurons of 1 X 1 matrix to filter dataset 32 times
    cnn_model.add(Convolution2D(32, (1, 1), input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    #max layer tio collect relevant filtered features from CNN layer
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    cnn_model.add(Convolution2D(32, (1, 1), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (1, 1)))
    #defining output layer
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train1.shape[1], activation = 'softmax'))
    #compiling the model
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #training and loading model
    if os.path.exists("model/cnn_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train1, y_train1, batch_size = 8, epochs = 50, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model.load_weights("model/cnn_weights.hdf5")
    #performing prediction on test data   
    predict = cnn_model.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    #calling function to calculate accuracy on predicted data
    calculateMetrics("Extension CNN2D", y_test1, predict)

def graph():
#comparison graph between all algorithms
    df = pd.DataFrame([['XGBoost','Accuracy',accuracy[0]],['XGBoost','Precision',precision[0]],['XGBoost','Recall',recall[0]],['XGBoost','FSCORE',fscore[0]],
                   ['Graph Model','Accuracy',accuracy[1]],['Graph Model','Precision',precision[1]],['Graph Model','Recall',recall[1]],['Graph Model','FSCORE',fscore[1]],
                   ['Neural Network','Accuracy',accuracy[2]],['Neural Network','Precision',precision[2]],['Neural Network','Recall',recall[2]],['Neural Network','FSCORE',fscore[2]],
                   ['Extension CNN2D','Accuracy',accuracy[3]],['Extension CNN2D','Precision',precision[3]],['Extension CNN2D','Recall',recall[3]],['Extension CNN2D','FSCORE',fscore[3]],
                  ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()


def predict():
    global labels
    global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
    global accuracy, precision, recall, fscore, values, text,label_encoder,GraphCNN,cnn_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    testData = pd.read_csv("Dataset/testData.csv")#load test data
    temps = testData.values
    label_encoder = []
    for i in range(len(label_encoder)-1):
        temp = label_encoder[i]
        name = temp[0]
        le = temp[1]
        testData[name] = pd.Series(le.transform(testData[name].astype(str)))#encode all str columns to numeric
    testData.fillna(0, inplace = True)#replace missing values
    testData = testData.values
    X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))#reshape test data as cnn2d format
    predict = cnn_model.predict(X)#predict on test data and then display predicted drug
    for i in range(len(predict)):
        text.insert(END, "Test Data = " + str(temps[i]) + " Predicted Drug = " + str(labels[np.argmax(predict[i])]) + "\n")

       
        
font = ('times', 16, 'bold')
title = Label(main, text='Explainable Artificial Intelligence for Patient Safety: A Review of Application in Pharmacovigilance')
title.config(bg='#FFF8DC', fg='#D2691E')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Data Preprocessing", command=preprocessing)
preprocessButton.place(x=250,y=100)
preprocessButton.config(font=font1)

xgboostButton = Button(main, text="Run XGBOOST Algorithm", command=xgboost)
xgboostButton.place(x=500,y=100)
xgboostButton.config(font=font1)

graphcnnButton = Button(main, text="Run Graph CNN Algorithm", command=graphcnn)
graphcnnButton.place(x=750,y=100)
graphcnnButton.config(font=font1)

mlpButton = Button(main, text="Run Neural Network Algorithm", command=MLP)
mlpButton.place(x=1000,y=100)
mlpButton.config(font=font1)

cnnButton = Button(main, text="Run Extension CNN2D", command=CNN)
cnnButton.place(x=10,y=150)
cnnButton.config(font=font1)

graphButton = Button(main, text="Comparision Graph", command=graph)
graphButton.place(x=250,y=150)
graphButton.config(font=font1)



predictButton = Button(main, text="Upload Test Data", command=predict)
predictButton.place(x=500,y=150)
predictButton.config(font=font1)



main.config(bg='#808000')
main.mainloop()
