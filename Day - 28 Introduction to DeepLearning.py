#Diabeties report for deeplearning

from numpy import loadtxt #handle/load datset
from keras.models import Sequential # Empty working area
from keras.layers import Dense #Dense layer

dataset = loadtxt('pima-indians-diabeties.csv',delimiter = ',')
x = dataset[:,0:8]
y = dataset[:,8]
print(x)

model = Sequential()
model.add(Dense(12,input_dim=8,activateion = 'relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics=['accuracy'])
model.fit(x,y,epochs=5,batch_size=10)


_, accuracy = model.evaluate(x,y)
print('Accuracy : %.2f'%(accuracy*100))

model_json = model.to_json()
with open("model.json",'w') as json_file :
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")



#For testing > Do this in separate slide

from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabeties.csv',delimieter = ',')
x= dataset[:,0:8]
y = datset[:,8]

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load+weights('model.h5')
print('loaded model from disk')

predictions = mdoel.predict+classes(x)

for i in range(5,10):
    print('%x => %d (ORighinal Class : %d' % (x[i].tolist(), predictions[i],


































