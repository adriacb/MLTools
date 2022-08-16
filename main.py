
from utils import *
from ann import Autoencoder

def read_data():
    df = pd.read_csv('/home/adria/Downloads/BBBP.csv')
    #df['mol'] = df['smiles'].progress_apply(lambda x: Chem.MolFromSmiles(x, sanitize=True))
    from sklearn.utils import resample
    # Separate majority and minority classes
    df_majority = df[df.p_np==1]
    df_minority = df[df.p_np==0]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                    replace=True,     # sample with replacement
                                    n_samples=1567,    # to match majority class
                                    random_state=123) # reproducible results
    
    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    dfs = list()

    for index, row in df.iterrows():

        try:
            curr_mol_df = pd.DataFrame([row.to_dict()])
            mol = Chem.MolFromSmiles(row['smiles'], sanitize=True)
            curr_mol_df['mol'] = Chem.AddHs(mol)
            to_add = pd.concat([pd.DataFrame([row.to_dict()]), curr_mol_df], axis=1)
            dfs.append(to_add)
        except:
            print("Something wrong")
    df = pd.concat(dfs)


    df['MolWt'] = df['mol'].apply(lambda x: Descriptors.MolWt(x))
    df['LogP'] = df['mol'].apply(lambda x: Descriptors.MolLogP(x))
    df['NumHDonors'] = df['mol'].apply(lambda x: Chem.Lipinski.NumHDonors(x))
    df['NumAromaticRings'] = df['mol'].apply(lambda x: Chem.Lipinski.NumAromaticRings(x))
    df['PFI'] = df['LogP'] + df['NumAromaticRings']
    df['QueryProperties'] = (df['MolWt'] < 450)\
                        & (df['LogP'] < 3)\
                        & (df['NumHDonors'] <= 2)\
                        & (df['NumAromaticRings'] <= 3)\
                        & (df['PFI'] <= 5)


    features = ['MolWt', 'LogP', 'NumHDonors', 'NumAromaticRings', 'PFI', 'p_np']
    selected = df[features]
    selected = selected.loc[:,~selected.columns.duplicated()].copy()

    return selected



def main():

    selected = read_data()
    items_features_fitted, p_items = prepareInput(selected)
    

    ac = Autoencoder(
                        input_dim=items_features_fitted.shape.as_list()[1], 
                        hidden_dim_enc=[7, 500, 500, 2000],
                        hidden_dim_dec=[2000, 500], 
                        output_dim=3
                    )

    autoencoder, encoder = ac.build_model()
    

    # split into training and testing sets (80/20 split)
    train_data, test_data, train_labels, test_labels = train_test_split(items_features_fitted.numpy(), 
                                                                        selected, train_size=0.8, 
                                                                        random_state=5)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)


    # fit the model using the training data
    history = autoencoder.fit(train_data, train_data, 
                                    epochs=500, batch_size=256, shuffle=True, 
                                    validation_data=(test_data, test_data),
                                    verbose=1,
                                    callbacks=[callback])
                        



    ac.plot_history(history.history)
    ac.plot_model(path = 'model_1.png')


    # evaluate the model using the test data
    test_loss, test_acc = autoencoder.evaluate(test_data, test_data)
    print('Test loss:', test_loss)



if __name__ == '__main__':
    main()