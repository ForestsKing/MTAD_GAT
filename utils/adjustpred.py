def adjust_predicts(label, pred):
    anomaly_state = False

    for i in range(len(pred)):
        if label[i] and pred[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if not label[j]:
                    break
                else:
                    if not pred[j]:
                        pred[j] = 1

        elif not label[i]:
            anomaly_state = False

        if anomaly_state:
            pred[i] = 1
    return pred