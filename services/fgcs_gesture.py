import cv2
import mediapipe as mp
import numpy as np
import torch
from .cnn_lstm import CNN_LSTM


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PyQt5')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GestureRecogniton:
    def __init__(self, model_path='./hmi_FGCS.pt', seq_length=30, threshold_frames=20,device=None):
       
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils        
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                         model_complexity=1,
                                         min_detection_confidence=0.75,
                                         min_tracking_confidence=0.75)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = CNN_LSTM(input_size=99, output_size=128, hidden_size=64)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self.seq_length = seq_length
        self.threshold_frames = threshold_frames
        self.seq = []
        self.action_seq = []
        self.check_eventId = ''
        self.eventId_confidence = ''
        self.json_data = ''
        self.prev_action = '?'
        self.result_action = ''
        self.result_print = ''
        self.result_text = ''
        self.text_status = ''
        self.text = ''
        self.running = False

        self.actions = ['Select Drone', 'Select Group', 'Select Mode', 'ARM', 'DISARM', 'TAKEOFF', 'LAND', 'RTL', 
                        'Change Altitude', 'Change Speed', 'Move Up', 'Move Down', 'Rotate CW', 'Move Forward', 
                        'Move Backward', 'Move Right', 'Move Left', 'Rotate CCW', 'Cancel', 'Check', 
                        'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']

    def process_frame(self, img):
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.hands.process(img)
        
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 가까운 손만 detection
        if result.multi_hand_landmarks:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21,4))
            
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                v = v2 - v1 

                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) 

                gesture_joint = np.array([angle], dtype=np.float32)
                gesture_joint = np.concatenate([joint.flatten(), angle])

                self.seq.append(gesture_joint)

                self.mp_drawing.draw_landmarks(
                            img, 
                            res, 
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(), 
                            self.mp_drawing_styles.get_default_hand_connections_style())
                
        return img
    
    def predict_action(self):
        if len(self.seq) < self.seq_length:
            return None

        input_data = np.expand_dims(np.array(self.seq[-self.seq_length:], dtype=np.float32), axis=0)
        input_data = torch.FloatTensor(input_data).to(self.device)


        with torch.no_grad():
            y_pred = self.model(input_data)
        values, indices = torch.max(y_pred.data, dim=1,keepdim=True)
        model_confidence = values.item()

        if model_confidence < 0.98:
            return None

        action = self.actions[indices.item()]
        self.action_seq.append(action) 

        if len(self.action_seq) >= self.threshold_frames and all(action == self.action_seq[-1] for action in self.action_seq[-self.threshold_frames:]):
            self.result_action = self.action_seq[-1]
            self.action_seq = []

            if self.result_action in ['TAKEOFF', 'LAND', 'RTL', 'Ten', 'Check', 'Cancel']:

                if self.result_action == 'TAKEOFF':
                        self.prev_action = self.result_action
                        self.result_text = '이륙'
                        self.result_print = "takeoff"

                elif self.result_action == 'LAND':
                        self.prev_action = self.result_action
                        self.result_text = '착륙'
                        self.result_print = "land"    

                elif self.result_action == 'RTL':
                        self.prev_action = self.result_action
                        self.result_text = '귀환'
                        self.result_print = 'return'

                elif self.result_action == 'Ten':
                        self.prev_action = self.result_action
                        self.result_text = '정지'
                        self.result_print = 'stop'

                elif self.result_action == 'Check':
                        self.prev_action = self.result_action
                        self.result_text = '확인'
                        self.result_print = 'confirm'   

                elif self.result_action == 'Cancel':
                        self.prev_action = self.result_action
                        self.result_text = '취소'
                        self.result_print = "cancel"

                else:
                    self.result_action = '?'
                    self.prev_action = self.result_action

                return self.result_print
                
                    
              
                    
                    
            
       
    
    


