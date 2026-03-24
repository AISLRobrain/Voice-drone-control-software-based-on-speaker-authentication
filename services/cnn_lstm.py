import torch
import torch.nn as nn  


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(CNN_LSTM, self).__init__()
        self.input_size = input_size    # 입력 크기(99)로, 훈련 데이터셋의 칼럼 개수
        self.output_size = output_size          # CNN 출력 크기
        self.hidden_size = hidden_size  # 은닉층 개수 
        # self.num_layers = num_layers    # LSTM 층 개수
        # self.num_classes = num_classes  # class 개수

        self.cnn = nn.Conv1d(in_channels=input_size,
                             out_channels=output_size, 
                             kernel_size=3, 
                             stride=1, 
                             padding=0)
        
        self.lstm = nn.LSTM(input_size=output_size,
                            hidden_size=hidden_size, 
                            num_layers=1, 
                            dropout=0.2,
                            batch_first=True)
        
        self.Relu = nn.ReLU()

        self.fc = nn.Linear(in_features=hidden_size, 
                            out_features=30)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
    
        # print('1 : ', x.shape)    # torch.Size([64, 30, 99])
        x=x.permute(0,2,1)
        # print('2 : ', x.shape)    # torch.Size([64, 99, 30])
        x = self.cnn(x)
        # print('3 : ', x.shape)    # torch.Size([64, 64, 28])
        x = self.Relu(x)
        # print('4 : ', x.shape)    # torch.Size([64, 64, 28])
        x = x.permute(0, 2, 1)
        h_n, c_n = self.lstm(x)
        # print('5 : ', h_n.shape)    # torch.Size([64, 28, 32])
        x = self.fc(h_n[:, -1, :])  #[:,:,0]
        # print('6 : ', x.shape)  # torch.Size([64, 21])
        x = self.softmax(x)
        # print('7 : ', x.shape)  
        return x