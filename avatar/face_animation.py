import cv2
import mediapipe as mp
import time

class AvatarAnimator:
    def __init__(self, avatar_image_path):
        self.avatar = cv2.imread(avatar_image_path)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False)
        self.drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.drawing.DrawingSpec(thickness=1, circle_radius=1)

    def animate_avatar(self):
        cap = cv2.VideoCapture(0)  # Use webcam
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip and process frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            # Draw landmarks on avatar
            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    self.drawing.draw_landmarks(
                        image=self.avatar,
                        landmark_list=landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawing_spec
                    )

            # Display the avatar with landmarks
            cv2.imshow("Avatar Animation", self.avatar)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    animator = AvatarAnimator("avatar/babak.png")
    animator.animate_avatar()
