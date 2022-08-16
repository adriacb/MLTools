from utils import *


class Autoencoder:
    def __init__(self, input_dim: int, 
                       hidden_dim_enc: int, 
                       hidden_dim_dec: int,
                       output_dim: int, activation: str='relu', 
                       optimizer: str='adam', loss: str='mse', metrics: list=['mse'],
                       kernel_initializer: str='glorot_uniform'):

        self.input_df = keras.Input( shape = (input_dim, ))
        self.input_dim = input_dim
        # It can be a list of ints or a single int
        self.hidden_dim_enc = hidden_dim_enc
        self.hidden_dim_dec = hidden_dim_dec

        self.output_dim = output_dim
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.k_initializer = kernel_initializer
        self.model, self.encoder = self.build_model()

    
    def build_model(self):
        if isinstance(self.hidden_dim_enc, list) and isinstance(self.hidden_dim_dec, list):
            first=True
            for i in self.hidden_dim_enc:
                if first:
                    x = Dense(i, activation=self.activation)(self.input_df)
                    first = False
                else:
                    x = Dense(i, activation=self.activation, kernel_initializer=self.k_initializer)(x)
            
            encoded = Dense(self.output_dim, activation=self.activation, kernel_initializer=self.k_initializer)(x)

            first=True
            for j in self.hidden_dim_dec:
                if first:
                    x = Dense(j, activation=self.activation, kernel_initializer=self.k_initializer)(encoded)
                    first = False
                else:
                    x = Dense(j, activation=self.activation, kernel_initializer=self.k_initializer)(x)
            
            decoded = Dense(self.input_dim, kernel_initializer=self.k_initializer)(x)
            
            autoencoder = keras.Model(self.input_df, decoded)
            encoder = keras.Model(self.input_df, encoded)

        else:
            x = Dense(self.hidden_dim_enc, activation=self.activation, kernel_initializer=self.k_initializer)(self.input_df)
            encoded = Dense(self.output_dim, activation='relu')(x)
            x = Dense(self.hidden_dim_dec, activation=self.activation, kernel_initializer=self.k_initializer)(encoded)
            decoded = Dense(self.input_dim, activation='relu')(x)
            
            autoencoder = keras.Model(self.input_df, decoded)
            encoder = keras.Model(self.input_df, encoded)
        
        autoencoder.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return autoencoder, encoder

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int=32, epochs: int=10, verbose: int=1, callbacks: list=None, validation_split: float=0.2, validation_data: tuple=None, shuffle: bool=True, class_weight: dict=None, sample_weight: dict=None):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split, validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int=32, verbose: int=1):
        return self.model.evaluate(X, y, batch_size=batch_size, verbose=verbose)
    
    def predict(self, X: np.ndarray, batch_size: int=32, verbose: int=0):
        return self.model.predict(X, batch_size=batch_size, verbose=verbose)
    
    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = load_model(path)

    def plot_history(self, history: dict, title: str='', save: bool=False, path: str=None):
        plt.figure(figsize=(12,8))
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(history['loss'], label='Train')
        plt.plot(history['val_loss'], label='Test')
        plt.legend()
        if save:
            plt.savefig(path)
        plt.show()
        plt.close()
        return history['loss'], history['val_loss']
    
    def plot_model(self, path: str=None):
        keras.utils.plot_model(self.model, to_file=path, show_shapes=True, show_layer_names=True)

    def accuracy(self, X: np.ndarray, y: np.ndarray):
        return self.model.evaluate(X, y)
    

