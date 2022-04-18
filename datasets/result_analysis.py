import numpy as np
import cv2

def mark_TP_FP_TN_FN(predict: np.ndarray, label: np.ndarray):
    shape = (predict.shape[0], predict.shape[1], 3)
    mark = np.zeros(shape)
    count_FN = 0
    count_FP = 0
    count_TN = 0
    count_TP = 0
    for row in range(shape[0]):
        for col in range(shape[1]):
            # cv2输出按照BGR
            if predict[row, col] == 0 and label[row, col] == 0:
                mark[row, col, :] = [0, 0, 0]
                count_TN += 1
            elif predict[row, col] == 255 and label[row, col] == 255:
                mark[row, col, :] = [255, 255, 255]
                count_TP += 1
            elif predict[row, col] == 255 and label[row, col] == 0:
                mark[row, col, :] = [255, 0, 0]
                count_FN += 1
            else:
                mark[row, col, :] = [0, 0, 255]
                count_FP += 1
    return mark, count_FN, count_FP, count_TN, count_TP

if __name__ == '__main__':
    path = '/archive/hot8/cd_data/BIT/vis/auxilary_model_kmcuda_lam0.5/'
    img = cv2.imread(path + 'eval_1.jpg')
    print(img.shape)
    crop = int(img.shape[1] / 4)
    total = 256 * 4096
    label = img[:, 256*2:256*3, 0]
    pred = img[:, 256*3:256*4, 0]
    label[label < 128] = 0
    label[label > 128] = 255
    pred[pred < 128] = 0
    pred[pred > 128] = 255
    print(label.shape)
    print(pred.shape)
    mark, count_FN, count_FP, count_TN, count_TP = mark_TP_FP_TN_FN(pred, label)
    print("FN: ", count_FN, count_FN/total)
    print("FP: ", count_FP, count_FP/total)
    print("TN: ", count_TN, count_TN/total)
    print("TP: ", count_TP, count_TP/total)
    img = np.concatenate((img, mark), axis=1)
    cv2.imwrite(path+'eval_TP_FP_TN_FN.png', img)
