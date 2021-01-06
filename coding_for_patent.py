import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(r'C:\Users\MIM-1\Desktop\gps')
os.getcwd()
import glob
'''
   file1 == gps man's pixel coordinates
   file2 == gps man's map coordinates
   file3 == machine learning people,cars coordinates
   
'''

file = '' #gps man picture coordinates csvfile
file2 = r'C:\Users\MIM-1\Documents\xy.csv' #gps man's location coordinates csvfile
file3 = 'suwon_output.txt' #machine learning every people & cars picture coordinates csvfile


#gps man on the picture--machine learning 으로 찾은 사람 좌표
def standard_in_picture(file):
 ###pixel = pd.read_csv(file)
    four_pixel = [[104, 332], [219, 916], [745, 844], [335, 365]]
    return np.float32(four_pixel)
    #[[226, 1103], [745, 980], [74, 237], [268, 261]] 식의 return1
    

#gps man on map: gps coordinate file
def getgps(file2):
    gps = pd.read_csv(file2)
    x = gps['x']
    y = gps['y']
    gps = [[x[0],y[0]],[x[1],y[1]],[x[2],y[2]],[x[3],y[3]]]
    return np.float32(gps)
    #[[18.850, 979.620], [0.0, 611.907], [2308.900, 496.375], [2281.149, 0.0]]식의 return2


#perspective transform (parameter은 standard_in_picure, getgps의 리턴값 넣기)
def pers_trans(gps_picture,gps_map):
 
    M = cv2.getPerspectiveTransform(gps_picture,gps_map)
    return M #변환 매트릭스
    '''
    dst = cv2.warpPerspective(img,M,(2400,2400))
    print(M)
    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    '''
    
#matrix를 구했으므로 other people, or cars
#datetime,object_found,confidence,X,Y 5개의 columns 만 남기고 다 삭제
#confidence에 이상 조건 넣음
def moving_map(file3,confidence=0):
    header_list = ['datetime','camera_location','object','confidence','X','Y']
    df = pd.read_csv(file3,names=header_list)
    df['datetime'] = df['datetime'].apply(lambda x: x[10:])
    df['confidence'] = df['confidence'].apply(lambda x: x[12:])
    df['X'] = df['X'].apply(lambda x: x[12:])
    df['object'] = df['object'].apply(lambda x: x[15:])
    df =df.drop(columns = ['camera_location'])
    df['datetime'] = pd.to_datetime(df['datetime'],format = '%Y-%m-%d %H:%M:%S')
    df[['confidence','X','Y']] = df[['confidence','X','Y']].astype(float)
    df2 = df['confidence']>confidence
    df = df[df2] # confidence 이하는 전부 삭제
    df = df.drop(columns = 'confidence')
    df = df.set_index(['datetime','object'])
    return df           #confidence 이상만 남기고 x,y,datetime, object 만 존재


#matrix multiply & convert coord to (real or birdeye coordinates)
def map_to_picture(df,M):
    x = [x for x in df.X]
    y = [y for y in df.Y]
    
    listx = []
    listy = []
    for i in range(len(x)):
       coor = np.asmatrix(np.array([[x[i]],[y[i]],[1]]))
       t=np.matmul(M,coor)
       t=t/t[2]
       result = np.array(t).flatten()
       listx.append(result[0])
       listy.append(result[1])
    df['X'] = listx
    df['Y'] = listy
    return df





#_______________________________________________________________________________________



gps = glob.glob('g*.csv')
pixel = standard_in_picture(file)
#pixel = glob.glob('p*.csv')
for i in range(len(gps)):
    gpsc = getgps(gps[i])
    pixelc = standard_in_picture(pixel[i])
    matrix = pers_trans(pixelc,gpsc)
    df = moving_map(file3,confidence = 0.5) #---put wanting confidence
    result = map_to_picture(df,matrix)
    result.to_csv("result{}".format(gps[i][3:]))
    






















    
    
    