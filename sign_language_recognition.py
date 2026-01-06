#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手语识别系统 - 主训练脚本
使用 MediaPipe 提取关键点，LSTM 模型进行训练和识别
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import mediapipe as mp

# 配置参数
CONFIG = {
    'data_dir': '/root/autodl-tmp/data',
    'output_dir': '/root/autodl-nus/sign_language_output',
    'train_dataset_dir': '/root/autodl-nus/train_dataset',
    'model_save_path': 'sign_language_model.pth',
    'sequence_length': 120,  # 每个手势的帧数
    'input_size': 258,  # MediaPipe 关键点维度 (33*4 + 21*3 + 21*3)
    'hidden_size': 64,
    'num_epochs': 200,
    'batch_size': 16,
    'learning_rate': 0.001,
    'test_size': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# MediaPipe 初始化
mp_holistic = mp.solutions.holistic

def process_single_video_worker(args):
    """处理单个视频的辅助函数（用于多进程）"""
    video_path, max_frames, label = args
    
    try:
        # 每个进程创建自己的 MediaPipe 处理器
        holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            holistic.close()
            return None, label, video_path
        
        keypoints_sequence = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            # 只保存检测到手部的帧
            if results.left_hand_landmarks or results.right_hand_landmarks:
                # 提取关键点
                pose = np.array([[res.x, res.y, res.z, res.visibility] 
                                for res in results.pose_landmarks.landmark]).flatten() \
                    if results.pose_landmarks else np.zeros(33 * 4)
                lh = np.array([[res.x, res.y, res.z] 
                              for res in results.left_hand_landmarks.landmark]).flatten() \
                    if results.left_hand_landmarks else np.zeros(21 * 3)
                rh = np.array([[res.x, res.y, res.z] 
                              for res in results.right_hand_landmarks.landmark]).flatten() \
                    if results.right_hand_landmarks else np.zeros(21 * 3)
                keypoints = np.concatenate([pose, lh, rh])
                
                keypoints_sequence.append(keypoints)
                frame_count += 1
        
        cap.release()
        holistic.close()
        
        if len(keypoints_sequence) == 0:
            return None, label, video_path
        
        # 填充或截断到固定长度
        if len(keypoints_sequence) < max_frames:
            last_frame = keypoints_sequence[-1]
            keypoints_sequence.extend([last_frame] * (max_frames - len(keypoints_sequence)))
        else:
            keypoints_sequence = keypoints_sequence[:max_frames]
        
        return np.array(keypoints_sequence), label, video_path
        
    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {e}")
        return None, label, video_path


class MediaPipeProcessor:
    """MediaPipe 关键点提取器"""
    
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, results):
        """从 MediaPipe 结果中提取关键点"""
        # Pose landmarks (33 points, 4 values: x, y, z, visibility)
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
            if results.pose_landmarks else np.zeros(33 * 4)
        
        # Left hand landmarks (21 points, 3 values: x, y, z)
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
            if results.left_hand_landmarks else np.zeros(21 * 3)
        
        # Right hand landmarks (21 points, 3 values: x, y, z)
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
            if results.right_hand_landmarks else np.zeros(21 * 3)
        
        return np.concatenate([pose, lh, rh])
    
    def process_frame(self, frame):
        """处理单帧图像"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results
    
    def process_video(self, video_path, max_frames=30):
        """处理视频，提取关键点序列"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        keypoints_sequence = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.process_frame(frame)
            
            # 只保存检测到手部的帧
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = self.extract_keypoints(results)
                keypoints_sequence.append(keypoints)
                frame_count += 1
        
        cap.release()
        
        if len(keypoints_sequence) == 0:
            return None
        
        # 填充或截断到固定长度
        if len(keypoints_sequence) < max_frames:
            # 用最后一帧填充
            last_frame = keypoints_sequence[-1]
            keypoints_sequence.extend([last_frame] * (max_frames - len(keypoints_sequence)))
        else:
            keypoints_sequence = keypoints_sequence[:max_frames]
        
        return np.array(keypoints_sequence)
    
    def __del__(self):
        if hasattr(self, 'holistic'):
            self.holistic.close()


class SignLanguageLSTM(nn.Module):
    """手语识别 LSTM 模型"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLanguageLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.output_layer(x)
        
        return x


class SignLanguageDataset(Dataset):
    """手语数据集"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
