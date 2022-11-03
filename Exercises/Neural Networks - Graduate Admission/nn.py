import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

path = 'https://storage.googleapis.com/kagglesdsdata/datasets/14872/228180/Admission_Predict_Ver1.1.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220731%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220731T104429Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=3485323b2c85d4a0c7b01d0c9760b7f2c65c2cf26656678914eaf5f9c3eb561002c6bad96e0d933bdb1318d573e68cfb080cab13f64fb619f5d9bd5c23b0ee25ff666f38ab81a345d7f08d6592e5efca65cd3cb9fb77507f4f214ad76fd53e8d67667598e20b7a66dc7d740f68bb07f6f35bed4d2a81313e95fc80c834b47e4a1ab34dfb00216b308d267defe4bd9282a3e3f5f70199a93dc1959eec484a0073455e768afd32b53f4a04463e2828ee4f818dc1f54cda9336fc35e4a03e999d544483d14de2c083aeba3c8a6688e29ef54a598ec303a36e266abf5d8a574828958c1c9ff644e91d52a6ccd272895d0a5ecc35e8a6a10b3a0266e3a12b87a53231'

df = pd.read_csv(path)


X = df.iloc[:,1:-1]
y = df.iloc[:,-1]




X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)




def model_fn():
    model = Sequential( \
            [layers.InputLayer(input_shape=(X.shape[1])),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1,activation='sigmoid')
            ])
    model.compile(loss='mse', metrics='mae', optimizer=Adam(learning_rate=0.01))
    return model

model = model_fn()

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0)
hist = keras.callbacks.History()


scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train,\
                epochs = 150,\
                batch_size = 32,\
                callbacks = [es, hist],\
                validation_split=0.2,
                verbose=0
)




print(f'R2 score: {r2_score(y_test, model.predict(X_test_scaled))}')

# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below

fig, ax = plt.subplots(1,1)
ax.plot(hist.history['loss'], label='train')
ax.plot(hist.history['val_loss'], label='test')
ax.legend()
# fig.savefig('static/images/my_plots.png')
plt.show()
