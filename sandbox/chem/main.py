
from utils import *
from sandbox.ann import Autoencoder

def read_data():
    df = pd.read_csv('/home/acabello/Downloads/BBBP.csv')

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
    df = pd.concat([df_majority, df_minority_upsampled]) #df_unsampled = pd.concat([df_majority, df_minority_upsampled])
    

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
    # df['QueryProperties'] = (df['MolWt'] < 450)\
    #                     & (df['LogP'] < 3)\
    #                     & (df['NumHDonors'] <= 2)\
    #                     & (df['NumAromaticRings'] <= 3)\
    #                     & (df['PFI'] <= 5)
    
    df['NumHAcceptors'] = df['mol'].apply(lambda x: Chem.Lipinski.NumHAcceptors(x))
    df['NumRadicalElectrons'] = df['mol'].apply(lambda x: Descriptors.NumRadicalElectrons(x))
    df['NumValenceElectrons'] = df['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    df['HeavyAtomMolWt'] = df['mol'].apply(lambda x:Descriptors.HeavyAtomMolWt(x))
    df['MaxAbsPartialCharge'] = df['mol'].apply(lambda x: Descriptors.MaxAbsPartialCharge(x))
    df['MinAbsPartialCharge'] = df['mol'].apply(lambda x: Descriptors.MinAbsPartialCharge(x))
    df['fraCSP3'] = df['mol'].apply(lambda x: Chem.Lipinski.FractionCSP3(x))

    df['NHOHCount'] = df['mol'].apply(lambda x: Chem.Lipinski.NHOHCount(x))
    df['NOCount'] = df['mol'].apply(lambda x: Chem.Lipinski.NOCount(x))
    df['NumAliphaticCarbocycles'] = df['mol'].apply(lambda x: Chem.Lipinski.NumAliphaticCarbocycles(x))
    df['NumAliphaticHeterocycles'] = df['mol'].apply(lambda x: Chem.Lipinski.NumAliphaticHeterocycles(x))
    df['NumAliphaticRings'] = df['mol'].apply(lambda x: Chem.Lipinski.NumAliphaticRings(x))
    df['NumAromaticCarbocycles'] = df['mol'].apply(lambda x: Chem.Lipinski.NumAromaticCarbocycles(x))
    df['NumAromaticHeterocycles'] = df['mol'].apply(lambda x: Chem.Lipinski.NumAromaticHeterocycles(x))
    df['NumHeteroatoms'] = df['mol'].apply(lambda x: Chem.Lipinski.NumHeteroatoms(x))
    df['NumSaturatedCarbocycles'] = df['mol'].apply(lambda x: Chem.Lipinski.NumSaturatedCarbocycles(x))
    df['NumSaturatedHeterocycles'] = df['mol'].apply(lambda x: Chem.Lipinski.NumSaturatedHeterocycles(x))
    df['NumSaturatedRings'] = df['mol'].apply(lambda x: Chem.Lipinski.NumSaturatedRings(x))
    df['RingCount'] = df['mol'].apply(lambda x: Chem.Lipinski.RingCount(x))
            

    features = ['MolWt', 'LogP', 'NumHDonors', 'NumAromaticRings', 'PFI', 
    'NumHAcceptors', 'NumRadicalElectrons', 'NumValenceElectrons',
    'HeavyAtomMolWt', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'fraCSP3', 'NHOHCount',
    'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 
    'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumHeteroatoms', 'NumSaturatedCarbocycles', 
    'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount',
    'p_np']
    selected = df[features]
    selected = selected.loc[:,~selected.columns.duplicated()].copy()

    return selected

def read_data2():
    df = pd.read_csv('/home/acabello/Downloads/BBBP.csv')

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
    df_unsampled = pd.concat([df_majority, df_minority_upsampled]) #df_unsampled = pd.concat([df_majority, df_minority_upsampled])
   

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


    labels = df['p_np']
    labels = labels.loc[:,~labels.columns.duplicated()].copy()
            
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in df['mol']]

    
    return fps, labels['p_np'].to_numpy()

def main():
    """Autoencoder and descriptors"""
    selected = read_data()

    items_features_fitted = prepareInput(selected)
    print(items_features_fitted)
    print(items_features_fitted.shape)

    length = items_features_fitted.shape.as_list()[1]    
    

    ac = Autoencoder(
                        input_dim=length, 
                        hidden_dim_enc=[20, 10, 5],
                        hidden_dim_dec=[5, 10, 20], 
                        output_dim=3
                    )

    autoencoder, encoder = ac.build_model()
    


    # split into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(   
                            items_features_fitted.numpy(), 
                            selected['p_np'].to_numpy(), 
                            test_size=0.2, random_state=42)

    
    

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)


    # fit the model using the training data
    history = autoencoder.fit(X_train, X_train, 
                                    epochs=500, batch_size=32, shuffle=True, 
                                    validation_data=(X_test, X_test),
                                    verbose=1,
                                    callbacks=[callback])
    





    ac.plot_history(history.history)
    ac.plot_model(path = 'model_1.png')


    # evaluate the model using the test data
    test_loss, test_acc = autoencoder.evaluate(X_test, X_test)
    print('Test loss:', test_loss)

    predauto = encoder.predict(X_test)
    print(predauto)
    

    # apply K-Means to the autoencoder output
    kmeans = KMeans(n_clusters=2, random_state=0).fit(predauto)
    y_pred = kmeans.predict(predauto)
    y_pred = [1 if not i else 0 for i in y_pred]

    
    # calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)


def main2():
    """Morgan Fingerprints and K-Means"""

    fps, labels = read_data2()


    df_morgan_fps = pd.DataFrame (np.array (fps))
    print(df_morgan_fps)

    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(df_morgan_fps)
    
    # Dimension reduction with PCA 
     
    pca = PCA (n_components = 8) 
    decomp = pca.fit_transform(df_morgan_fps) 
    x = decomp [:, 0 ] 
    y = decomp [:, 1] 
    
    #Visualization 
    plt.figure (figsize = (15,5)) 
    #Color-coded clusters obtained by kmeans 
    plt.subplot(1,2,1)
    plt.scatter (x, y, c = kmeans.labels_, alpha = 0.7)
    plt.title ("PCA: morgan_fps, cluster")
    plt.colorbar () 
    plt.show ()

    # plot the points with colors corresponding to their cluster labels
    plt.scatter(x, y, c=labels, cmap='rainbow')
    plt.title("PCA: morgan_fps, labels")
    plt.show()

    # calculate the confusion matrix
    cm = confusion_matrix(labels, kmeans.labels_)
    print(cm)

    # calculate the accuracy
    accuracy = accuracy_score(labels, [1 if not i else 0 for i in kmeans.labels_])
    print(accuracy)



def main3():
    """Use all the descriptors and Dimensionality red with UMAP and K-Means"""

    selected = read_data()
    print(selected)

    

    # split into training and testing sets (80/20 split), train selected except 'p_np'
    X_train, X_test, y_train, y_test = train_test_split(
                            selected.drop(['p_np'], axis=1).to_numpy(),
                            selected['p_np'].to_numpy(),
                            test_size=0.2, random_state=42
    )

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # perform K-Means on the descriptors
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
    
    
    y_pred = [1 if not i else 0 for i in kmeans.predict(X_test)]

    # perform dimensionality reduction with UMAP
    reducer = umap.UMAP(n_components=2)
    reducer.fit(X_train)
    x_reduced = reducer.transform(X_test)
    
    # plot the points with colors corresponding to their cluster labels
    plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_pred, cmap='rainbow')
    plt.title("UMAP: descriptors, labels")
    plt.show()


    # calculate the confusion matrix
    cm = confusion_matrix(y_pred, y_test)
    print(cm)

    # calculate the accuracy
    accuracy = accuracy_score(y_pred, y_test)
    print(accuracy)



if __name__ == '__main__':
    main3()