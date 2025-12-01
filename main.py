import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        print("Result:", result)

        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]

            dominant_emotion = first_result['dominant_emotion']
            emotion_scores = first_result['emotion']

            filtered_emotions = {key: emotion_scores[key] for key in ['happy', 'neutral', 'surprise'] if key in emotion_scores}

            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            for i, (emotion, score) in enumerate(filtered_emotions.items()):
                text = f"{emotion}: {score:.2f}%"
                cv2.putText(frame, text, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        else:
            print("No faces detected.")

    except Exception as e:
        print(f"Error in DeepFace analysis: {e}")

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
