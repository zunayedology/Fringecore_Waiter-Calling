import cv2
import numpy as np

regions = [
    ((562, 515), (675, 515), "TANVIR"),
    ((710, 515), (847, 515), "FAISAL"),
    ((848, 515), (940, 515), "TOUFIQ"),
    ((1000, 515), (1090, 515), "MUFRAD"),
    ((1091, 515), (1158, 515), "ANIK"),
    ((1160, 515), (1268, 515), "IMRAN"),
    ((1270, 515), (1370, 515), "EMON")
]

def is_darker(current_pixel, previous_pixel, threshold=10):
    return np.linalg.norm(current_pixel) < np.linalg.norm(previous_pixel) - threshold

def main():
    cap = cv2.VideoCapture('desk_video.mp4')

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, previous_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        display_text = ""

        for (start, end, label) in regions:
            x_start, y_start = start
            x_end, y_end = end

            for x in range(x_start, x_end + 1):
                current_pixel = gray_frame[y_start, x]
                previous_pixel = previous_frame[y_start, x]

                if is_darker(current_pixel, previous_pixel):
                    display_text = label
                    break
            if display_text:
                break

        if display_text:
            cv2.putText(
                frame,
                display_text,
                (frame_width - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow('Video', frame)

        previous_frame = gray_frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
