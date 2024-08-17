Speech Emotion Recognition
This project classifies emotions from speech using deep learning.

Steps
1. Import Libraries
python
Copy code
import pandas as pd, numpy as np, os, seaborn as sns, matplotlib.pyplot as plt
import librosa, librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')
2. Load Dataset
python
Copy code
paths, labels = [], []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        labels.append(filename.split('_')[2].split('.')[0].lower())
df = pd.DataFrame({'speech': paths, 'label': labels})
3. Data Analysis
python
Copy code
sns.countplot(x='label', data=df)
4. Feature Extraction
python
Copy code
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
X = np.array([extract_mfcc(x) for x in df['speech']])
X = np.expand_dims(X, -1)
5. One-Hot Encode Labels
python
Copy code
from sklearn.preprocessing import OneHotEncoder
y = OneHotEncoder().fit_transform(df[['label']]).toarray()
6. Build and Train Model
python
Copy code
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential([
    LSTM(128, input_shape=(40,1)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512, shuffle=True)
7. Plot Results
python
Copy code
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
