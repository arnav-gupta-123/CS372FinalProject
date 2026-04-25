import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, AveragePooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

# P300 Neural Network Classifier 

def p300modelarch(nb_classes, chans, samples, dropout_rate, kern_length, F1, D, F2, norm_rate):
    inp = Input(shape=(chans, samples, 1))

    # Step 1
    x = Conv2D(F1, (1, kern_length), padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((chans, 1), use_bias=False, depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropout_rate)(x)

    # Step 2
    x = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropout_rate)(x)

    # Step 3
    x = SeparableConv2D(F2, (1, 8), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(x)
    out = Activation('softmax', dtype='float32')(x)

    return tf.keras.Model(inputs=inp, outputs=out)


class P300NNClassifierNCB:
    def __init__(self, nb_classes=2, chans=8, samples=206, dropout_rate=0.3, kern_length=16, F1=16, D=2, F2=32, norm_rate=0.25, learning_rate=0.01, patience=5, optim_type = 'adam'):

        self.model = p300modelarch(
            nb_classes=nb_classes, chans=chans, samples=samples,
            dropout_rate=dropout_rate, kern_length=kern_length,
            F1=F1, D=D, F2=F2, norm_rate=norm_rate
        )
        if optim_type == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optim_type == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=0.9)
        elif optim_type == "adamw":
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,weight_decay=1e-4)
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.patience = patience
        self.history = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # handle extreme class imbalance
        #weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        #class_weights = dict(enumerate(weights))

        early_stop = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.patience,
            restore_best_weights=True
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=3000,
            callbacks=[early_stop],
            #class_weight=class_weights,
            shuffle=True,
            verbose=2
        )

    def test(self, X_test, y_test):
        probs = self.model.predict(X_test)
        scores = probs[:, 1] - probs[:, 0]
        auc = roc_auc_score(y_test, scores)
        print(f"AUC: {auc:.4f}")
        return scores