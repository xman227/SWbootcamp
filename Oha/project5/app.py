import streamlit as st

import os #데이터 경로 라이브러리

import pandas as pd
import numpy as np # 수치 다루는 라이브러리


import re # 텍스트 핸들링 라이브러리

import tensorflow as tf # 텐서플로우
import chatbot_model as chat # 내가 만든 모델
import tensorflow_datasets as tfds # Subword 토크나이저 라이브러리




st.title('🙂 위로를 건네는 챗봇')

st.write("힘든 일이 있으시면, 챗봇에게 말을 걸어보아요..")


title = st.text_input('챗봇에게 : ', value="나 오늘 너무 우울해... 😭")
st.write("챗봇의 한마디 :")

#모델
tf.keras.backend.clear_session()

# 하이퍼파라미터
NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원
NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기, 노드
DROPOUT = 0.1 # 드롭아웃의 비율
VOCAB_SIZE = 6360

model = chat.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

model.load_weights('./my_model_weights')




corpus = []
with open('tokenize.subwords', 'r', encoding='utf-8') as f:
   for inx, line in enumerate(f):
       if inx > 1:
          sent = re.sub(r"$[_]+", " ", line)
          sent = line.replace('\n', '')
          corpus.append(sent)

def process(item):
    if item[-2] == '_':
        return (item[1:-2] + ' ')
    else:
        return item[1:-1]
corpus2 = list(map(process, corpus))

tokenizer = tfds.deprecated.text.SubwordTextEncoder(vocab_list = corpus2)


def decoder_inference(sentence):
    sentence = chat.preprocess_sentence(sentence)

    START_TOKEN, END_TOKEN = [6358], [6359]
    MAX_LENGTH = 16

  # 입력된 문장을 정수 인코딩 후, 시작 토큰과 종료 토큰을 앞뒤로 추가.
  # ex) Where have you been? → [[8331   86   30    5 1059    7 8332]]
    sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  # 디코더의 현재까지의 예측한 출력 시퀀스가 지속적으로 저장되는 변수.
  # 처음에는 예측한 내용이 없음으로 시작 토큰만 별도 저장. ex) 8331
    output_sequence = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 인퍼런스 단계
    for i in range(MAX_LENGTH):
    # 디코더는 최대 MAX_LENGTH의 길이만큼 다음 단어 예측을 반복합니다.
        predictions = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]

    # 현재 예측한 단어의 정수
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 현재 예측한 단어가 종료 토큰이라면 for문을 종료
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

    # 예측한 단어들은 지속적으로 output_sequence에 추가됩니다.
    # 이 output_sequence는 다시 디코더의 입력이 됩니다.
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)

    return tf.squeeze(output_sequence, axis=0)

def sentence_generation(sentence):
      # 입력 문장에 대해서 디코더를 동작 시켜 예측된 정수 시퀀스를 리턴받습니다.
    prediction = decoder_inference(sentence)

  # 정수 시퀀스를 다시 텍스트 시퀀스로 변환합니다.
    predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

    print('입력 : {}'.format(sentence))
    print('출력 : {}'.format(predicted_sentence))

    return predicted_sentence 




st.write(sentence_generation(title))
