import cv2 # pip install opencv-contrib-python

cap = cv2.VideoCapture(0) # 選擇第1隻攝影機
i=0 #檔案名稱序號
while cap.isOpened():
  # 從攝影機擷取一張影像
  ret, frame = cap.read() #ret=retval,frame=image    
  # 顯示圖片
  cv2.imshow('frame', frame)
  frame=cv2.resize(frame,(100,100))
  i=i+1
  cv2.imwrite("train/C" + '/' + str(i) + ".jpg", frame) #記得改路徑
  print("Save photo:" + str(i) + ".jpg")
  if i>=500: #存500張照片
    break
  key=cv2.waitKey(1)
  # 按q離開
  if key & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()


