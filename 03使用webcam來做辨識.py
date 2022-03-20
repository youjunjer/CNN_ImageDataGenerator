import time
import os
import tensorflow as tf 
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image,ImageOps
#顯示用宣告 請複製c:\windows\fonts\的微軟正黑體到根目錄
font1 = ImageFont.truetype('msjhl.ttc', 50)
font2 = ImageFont.truetype('msjhl.ttc', 30)
fillColor = (255,0,0)
#商品及價格列表
productDic={"C":"小貓吊飾",  "D":"狗狗吊飾", "L":"鱷魚吊飾","P":"布丁狗吊飾","N":"無商品"}
priceDic={"C":16,  "D":20, "L":35,"P":40,"N":0}
#載入模型
model=tf.keras.models.load_model('CNN_Toy.h5') #選擇正確的模型檔(h5)
size=(200,200)
data = np.ndarray(shape=(1, 200,200, 3), dtype=np.float32)
#取得類別標籤(用測試資料夾)
dirList = sorted(os.listdir("train\\"))
cap = cv2.VideoCapture(1)#自行選擇攝影機編號
while cap.isOpened():
    startTime = time.time()
    ret, img = cap.read()
    frame=cv2.resize(img,(640,480))
    #處理陣列
    img = cv2.resize(img,size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)    
    image = ImageOps.fit(image, size)
    #image.show()
    data[0]=np.asarray(image)/255
    #預測
    pr=model.predict(data)
    maxIndex = np.argmax(pr)#找出1,2,...0，機率最大的輸出
    maxPr=round(pr[0][maxIndex],3)
    #print(label) #印出所有機率
    if pr[0][maxIndex]>0.6:        
        dirName=dirList[maxIndex]
        result=productDic[dirName] + ",金額=" + str(priceDic[dirName])
        print(result)
        #print(result,"(", maxPr ,")")
    else:
        result="無法辨識"
        print(result)   
    endTime = time.time()    
    fps = round(1 / ( endTime - startTime),1)
    #print("fps:" + str(fps))
    #在圖片中加入結果
    img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    draw.text((0,0), "FPS:" + str(fps), font=font2, fill=fillColor)
    draw.text((0,30), result, font=font1, fill=fillColor)    
    frame = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)    
    cv2.imshow("Result", frame)
    key=cv2.waitKey(1)
    # 按q離開
    if key & 0xFF == ord('q'):
        break
# 釋放攝影機
cap.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
