import tensorflow as tf
import matplotlib.pyplot as plt
size=(100,100) #指定圖片大小
batchSize=64 #小->收斂快，大->穩定
trainFileCount=2500 #訓練檔案數量500x5=2500
testFileCount=2500 #測試檔案數量200x5=2500

# 簡單的二層卷積加上ReLU激勵函式，再接一個max-pooling層
model = tf.keras.models.Sequential()
#64個3x3卷積核(3or5or7)
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

#32個3x3卷積核
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

#轉平面層
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(5, activation='softmax')) #五類:C,D,L,N,P
model.summary() # 顯示類神經架構
model.compile(optimizer= "adam", #使用adam最佳化
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

#訓練的圖片生成器
train_ImgDataGen = tf.keras.preprocessing.image.ImageDataGenerator(   
    rescale= 1.0 /255,
    )

#指定訓練圖片路徑參數
train_generator = train_ImgDataGen.flow_from_directory(
    'train',#訓練樣本路徑
    target_size=size,
    batch_size=batchSize,
    class_mode='categorical' #多分類
    )

#驗證的圖片生成器
test_ImgDataGen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale= 1.0 /255,
    )

#指定驗證圖片路徑參數
validation_generator = test_ImgDataGen.flow_from_directory(
    'test',#驗證樣本路徑
    target_size=size,
    batch_size=batchSize,
    class_mode='categorical', #多分類
    )
  
# 然後我們可以用這個生成器來訓練網路了。
train_history=model.fit_generator(    
    train_generator, #指定訓練圖片生成器
    steps_per_epoch = trainFileCount//batchSize, #一個世代幾批次=訓練檔案總量/批次量
    epochs=20,
    verbose=1, 
    validation_steps =testFileCount//batchSize ,#一個世代幾批次=測試檔案總量/批次量
    validation_data=validation_generator, #指定驗證圖片生成器
    )

model.save('tensorflow\\Toy\\CNN_Toy.h5')

def show_train_history(train_history, train, validation):  
    plt.plot(train_history.history[train])  
    plt.plot(train_history.history[validation])  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.show() 

show_train_history(train_history, 'accuracy', 'val_accuracy')  
