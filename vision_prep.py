import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
from time import time

for whole_vidio_id in glob.glob('./videos/*'):

    try:

        with open('output_log', 'a') as a:
            a.write(whole_vidio_id.split('/')[-1])
            a.write('\n')

        video_id = whole_vidio_id.split('/')[-1]

        # if os.path.exists(f'./split_frame_train/{video_id[:-4]}') or os.path.exists(f'./split_frame_test/{video_id[:-4]}'):
        #     continue

        # os.mkdir(f'./split_frame/{video_id[:-4]}')

        cap = cv2.VideoCapture(os.path.join('./videos', video_id))

        if os.path.exists(f'./train/seg/{video_id[:-4]}_seg.csv'):
            idS = []
            startS = []
            endS = []
            tfS = []
            with open(f'./train/seg/{video_id[:-4]}_seg.csv') as dur_file:
                infos = dur_file.readlines()
                for info in range(len(infos)-1):
                    person_id, start_id, end_id, tf = infos[info+1].split(',')
                    idS.append(int(person_id))
                    startS.append(int(start_id))
                    endS.append(int(end_id))
                    tfS.append(int(tf))

            os.mkdir(f'./split_frame_train/{video_id[:-4]}')
            for cnt in range( len(idS) ):
                os.mkdir(f'./split_frame_train/{video_id[:-4]}/{idS[cnt]}_{startS[cnt]}_{endS[cnt]}_{int(tfS[cnt])}')

        elif os.path.exists(f'./test/seg/{video_id[:-4]}_seg.csv'):
            idS = []
            startS = []
            endS = []
            with open(f'./test/seg/{video_id[:-4]}_seg.csv') as dur_file:
                infos = dur_file.readlines()
                for info in range(len(infos)-1):
                    person_id, start_id, end_id = infos[info+1].split(',')
                    idS.append(int(person_id))
                    startS.append(int(start_id))
                    endS.append(int(end_id))
                pin = 0

            os.mkdir(f'./split_frame_test/{video_id[:-4]}')
            for cnt in range( len(idS) ):
                os.mkdir(f'./split_frame_test/{video_id[:-4]}/{idS[cnt]}_{startS[cnt]}_{endS[cnt]}')

        else: raise('wrong file name !')

        # ======================================================================================================================================================
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # with open('output_log', 'a') as f:
        #     print(f'{video_id[:-4]}, {num_frames}', file=f)

        for idx in tqdm( range(num_frames) ):
            ret, frame = cap.read()

            # === train partial ===

            if os.path.exists(f'./train/bbox/{video_id[:-4]}_bbox.csv'):
                with open(f'./train/bbox/{video_id[:-4]}_bbox.csv') as xy_file:
                    infos = xy_file.readlines()
                    for info in range(len(infos)-1):
                        person_id, frame_id, x1, y1, x2, y2 = infos[info+1].split(',')
                        if int(frame_id) == idx and x1 != '-1':
                            
                            for id_range in glob.glob(f'./split_frame_train/{video_id[:-4]}/{int(person_id)}_*'):
                                id, start, end, _tf = id_range.split('/')[-1].split('_')
                                # print(id, start, end)
                                if int(frame_id) >= int(start) and int(frame_id) <= int(end):
                                    # print('save train !')
                                    cv2.imwrite(os.path.join(f'./split_frame_train/{video_id[:-4]}/{id}_{start}_{end}_{_tf}', f'{id}_{frame_id}.jpg'), frame[round(float((y1))):round(float((y2))), round(float((x1))):round(float((x2)))])
                                    # np.save(os.path.join(f'./split_frame_train/{video_id[:-4]}', f'{person_id}_{frame_id}'), frame[round(float((y1))):round(float((y2))), round(float((x1))):round(float((x2)))])

            # === test partial ===

            elif os.path.exists(f'./test/bbox/{video_id[:-4]}_bbox.csv'):
                with open(f'./test/bbox/{video_id[:-4]}_bbox.csv') as xy_file:
                    infos = xy_file.readlines()
                    for info in range(len(infos)-1):
                        person_id, frame_id, x1, y1, x2, y2 = infos[info+1].split(',')
                        if int(frame_id) == idx and x1 != '-1':
                            
                            for id_range in glob.glob(f'./split_frame_test/{video_id[:-4]}/{int(person_id)}_*'):
                                id, start, end = id_range.split('/')[-1].split('_')
                                # print(id, start, end)
                                if int(frame_id) >= int(start) and int(frame_id) <= int(end):
                                    # print('save test !')
                                    cv2.imwrite(os.path.join(f'./split_frame_test/{video_id[:-4]}/{id}_{start}_{end}', f'{id}_{frame_id}.jpg'), frame[round(float((y1))):round(float((y2))), round(float((x1))):round(float((x2)))])
                                    # np.save(os.path.join(f'./split_frame_test/{video_id[:-4]}', f'{person_id}_{frame_id}'), frame[round(float((y1))):round(float((y2))), round(float((x1))):round(float((x2)))])

            else: raise('wrong file name !')

            # === all ===

            # cv2.imwrite(os.path.join(f'./split_frame/{video_id[:-4]}', f'frame_{idx}.jpg'), frame)
            # np.save(os.path.join(f'./split_frame/{video_id[:-4]}', f'frame_{idx}'), frame)

    except:
        
        with open('output_log', 'a') as a:
            a.write('try unsuccess :')
            a.write(whole_vidio_id.split('/')[-1])
            a.write('\n')