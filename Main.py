
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
import soundfile
import librosa
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

main = tkinter.Tk()
main.title("Depression Detection from Text, Image & Speech using Deep Learning Algorithm") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y, text_X, text_Y, text_X_train, text_X_test, text_y_train, text_y_test, rf_model, tfidf_vectorizer
global face_classifier
global speech_X, speech_Y
global speech_classifier
global accuracy, precision, recall, fscore
global speech_X_train, speech_X_test, speech_y_train, speech_y_test
global image_X_train, image_X_test, image_y_train, image_y_test
stop_words = set(stopwords.words('english'))

def getID(name):
    index = 0
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index        
    

def uploadDataset():
    global filename, tfidf_vectorizer
    filename = filedialog.askdirectory(initialdir=".")
    f = open('model/tfidf.pckl', 'rb')
    tfidf_vectorizer = pickle.load(f)
    f.close()  
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
    
def processDataset():
    text.delete('1.0', END)
    global X, Y, text_X, text_Y
    global speech_X, speech_Y
    global speech_X_train, speech_X_test, speech_y_train, speech_y_test
    global image_X_train, image_X_test, image_y_train, image_y_test
    global text_X_train, text_X_test, text_y_train, text_y_test
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
        speech_X = np.load('model/speechX.txt.npy')
        speech_Y = np.load('model/speechY.txt.npy')
        text_X = np.load("model/textX.txt.npy")
        text_Y = np.load("model/textY.txt.npy")
        indices = np.arange(text_X.shape[0])
        np.random.shuffle(indices)
        text_X = text_X[indices]
        text_Y = text_Y[indices]
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(name+" "+root+"/"+directory[j])
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32,32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr)
                    Y.append(getID(name))        
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    image_X_train, image_X_test, image_y_train, image_y_test = train_test_split(X, Y, test_size=0.2)
    speech_X_train, speech_X_test, speech_y_train, speech_y_test = train_test_split(speech_X, speech_Y, test_size=0.2)
    text_X_train, text_X_test, text_y_train, text_y_test = train_test_split(text_X, text_Y, test_size=0.2)
    text.insert(END,"Total number of Depression images found in dataset is  : "+str(len(X))+"\n")
    text.insert(END,"Total number of Depression speech audio files found in dataset is  : "+str(speech_X.shape[0])+"\n\n")
    text.insert(END,"Total number of Depression Text Comments found in dataset is  : "+str(text_X.shape[0])+"\n\n")
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"80% images used to train Deep Learning Algorithm : "+str(image_X_train.shape[0])+"\n")
    text.insert(END,"20% images used to test Deep Learning Algorithm : "+str(image_X_test.shape[0])+"\n")
    text.insert(END,"80% Speech Audio used to train Deep Learning Algorithm : "+str(speech_X_train.shape[0])+"\n")
    text.insert(END,"20% Speech Audio used to test Deep Learning Algorithm : "+str(speech_X_test.shape[0])+"\n")
    text.insert(END,"80% Text Comment used to train Deep Learning Algorithm : "+str(text_X_train.shape[0])+"\n")
    text.insert(END,"20% Text Comment used to test Deep Learning Algorithm : "+str(text_X_test.shape[0])+"\n")
    text_X_train, text_X_test1, text_y_train, text_y_test1 = train_test_split(text_X, text_Y, test_size=0.1)

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

def trainTextRF():
    global text_X_train, text_X_test, text_y_train, text_y_test, rf_model
    rf_model = RandomForestClassifier(max_depth=30)
    rf_model.fit(text_X_train, text_y_train)
    predict = rf_model.predict(text_X_test)
    calculateMetrics("Text Random Forest Algorithm", predict, text_y_test)

def predictTextDepression():
    text.delete('1.0', END)
    global rf_model, tfidf_vectorizer, stop_words
    test_file = filedialog.askopenfilename(initialdir="testText")
    test = pd.read_csv(test_file, encoding='iso-8859-1')#read test data
    test = test.values
    for i in range(len(test)):
        comments = test[i,0]#loop all comments from test dataset
        print(comments)
        arr = comments.split(" ")
        msg = ''
        for k in range(len(arr)):#remove stop words
            word = arr[k].strip()
            if len(word) > 2 and word not in stop_words:
                msg+=arr[k]+" "
        text_data = msg.strip()
        text_data = [text_data]
        text_data = tfidf_vectorizer.transform(text_data).toarray()#convert text to numeric vector
        predict = rf_model.predict(text_data)[0]# predict sentiment from test comments
        print(predict)
        if predict == 0:
            text.insert(END,"User Text Comment = "+comments+" Predicted as ----> NON-Depressed\n\n")
        elif predict == 1:
            text.insert(END,"User Text Comment = "+comments+" Predicted as ----> Depressed\n\n")
    

def trainSpeechCNN():
    global speech_classifier
    global speech_X_train, speech_X_test, speech_y_train, speech_y_test
    if os.path.exists('model/speechmodel.json'):
        with open('model/speechmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            speech_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        speech_classifier.load_weights("model/speech_weights.h5")
        speech_classifier._make_predict_function()                  
    else:
        speech_classifier = Sequential()
        speech_classifier.add(Convolution2D(32, 1, 1, input_shape = (speech_X.shape[1], speech_X.shape[2], speech_X.shape[3]), activation = 'relu'))
        speech_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        speech_classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
        speech_classifier.add(MaxPooling2D(pool_size = (1, 1)))
        speech_classifier.add(Flatten())
        speech_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        speech_classifier.add(Dense(output_dim = speech_Y.shape[1], activation = 'softmax'))
        print(speech_classifier.summary())
        speech_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = speech_classifier.fit(speech_X_train, speech_y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        speech_classifier.save_weights('model/speech_weights.h5')            
        model_json = speech_classifier.to_json()
        with open("model/speechmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/speechhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = speech_classifier.predict(speech_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(speech_y_test, axis=1)
    calculateMetrics("CNN Speech Algorithm", predict, y_test1)


def trainFaceCNN():
    global face_classifier, accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    global image_X_train, image_X_test, image_y_train, image_y_test
    text.delete('1.0', END)
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            face_classifier = model_from_json(loaded_model_json)
        json_file.close()    
        face_classifier.load_weights("model/cnnmodel_weights.h5")
        face_classifier._make_predict_function()                  
    else:
        face_classifier = Sequential()
        face_classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Flatten())
        face_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        face_classifier.add(Dense(output_dim = 7, activation = 'softmax'))
        print(face_classifier.summary())
        face_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = face_classifier.fit(image_X_train, image_y_train, batch_size=16, epochs=10, shuffle=True, verbose=2)
        face_classifier.save_weights('model/cnnmodel_weights.h5')            
        model_json = face_classifier.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = face_classifier.predict(image_X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(image_y_test, axis=1)
    face_classifier = load_model("model1/cnn_weights.hdf5")
    calculateMetrics("CNN Image Algorithm", predict, y_test1)   

def predictFaceDepression():
    global face_classifier
    text.delete('1.0', END)
    global face_classifier
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = face_classifier.predict(img)
    predict = np.argmax(preds)
    output = "Depressed"
    if predict == 0:
        output = "Happy"
    if predict == 1:
        output = "Neutral"
    if predict == 2:
        output = "Sad"
    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Facial Output : '+output, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Facial Output : '+output, img)
    cv2.waitKey(0)



def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    sound_file.close()        
    return result

def predictSpeechDepression():
    text.delete('1.0', END)
    global speech_classifier
    filename = filedialog.askopenfilename(initialdir="testSpeech")
    fname = os.path.basename(filename)
    test = []
    mfcc = extract_feature(filename, mfcc=True, chroma=True, mel=True)
    test.append(mfcc)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255

    test = test.reshape((test.shape[0],test.shape[1],1,1))
    predict = speech_classifier.predict(test)
    predict = np.argmax(predict)
    print(predict)
    predict = predict - 1
    output = "Depressed"
    if predict == 0 or predict == 1 or predict == 2:
        output = "Non Depressed"  
    text.delete('1.0', END)
    text.insert(END,"Upload speech file : "+fname+" Output : "+output+"\n")
    
def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['Image CNN','Accuracy',accuracy[0]],['Image CNN','Precision',precision[0]],['Image CNN','Recall',recall[0]],['Image CNN','FSCORE',fscore[0]],
                       ['Speech CNN','Accuracy',accuracy[1]],['Speech CNN','Precision',precision[1]],['Speech CNN','Recall',recall[1]],['Speech CNN','FSCORE',fscore[1]],
                       ['Text Random Forest','Accuracy',accuracy[2]],['Text Random Forest','Precision',precision[2]],['Text Random Forest','Recall',recall[2]],['Text Random Forest','FSCORE',fscore[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()


font = ('times', 13, 'bold')
title = Label(main, text='Depression Detection from Text, Image & Speech using Deep Learning Algorithm')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=420,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Depression Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train Facial Depression CNN Algorithm", command=trainFaceCNN)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 

rnnButton = Button(main, text="Train Speech Depression CNN Algorithm", command=trainSpeechCNN)
rnnButton.place(x=50,y=250)
rnnButton.config(font=font1)

rnnButton = Button(main, text="Train Text Depression Random Forest Algorithm", command=trainTextRF)
rnnButton.place(x=50,y=300)
rnnButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

predictfaceButton = Button(main, text="Predict Facial Depression", command=predictFaceDepression)
predictfaceButton.place(x=50,y=400)
predictfaceButton.config(font=font1)

predictspeechButton = Button(main, text="Predict Speech Depression", command=predictSpeechDepression)
predictspeechButton.place(x=50,y=450)
predictspeechButton.config(font=font1)

predictspeechButton = Button(main, text="Predict Text Depression", command=predictTextDepression)
predictspeechButton.place(x=50,y=500)
predictspeechButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
