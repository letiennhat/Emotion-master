def feture_extraction(gray_face):
    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    print(gray_face.shape)
    emotion_prediction = emotion_classifier.predict(gray_face)
    print(emotion_prediction)
    emotion_prediction[0][0:3]=[0,0,0]
    emotion_prediction[0][5]=0
    emotion_probability = np.max(emotion_prediction)
    
    emotion_label_arg = np.argmax(emotion_prediction)
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_window.append(emotion_text)
    return emotion_text