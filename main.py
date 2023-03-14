from fastapi import FastAPI, File, UploadFile
import shutil
import os
from pydub import AudioSegment
from scipy.io.wavfile import read
from FeatureExtraction import extract_features
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import _pickle as cPickle

app = FastAPI()


@app.post("/api/register/{username}")
async def register(username: str, file: UploadFile):
    filepath = f"uploadedFiles/{username}"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    with open(f"{filepath}/{username}.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    t1 = 0
    t2 = 5
    seg=0
    for t2 in range(5,30,5):
        seg=seg+1
        newAudio = AudioSegment.from_wav(f"{filepath}/{username}.wav")
        newAudio = newAudio[t1*1000:t2*1000]
        newAudio.export(f"{filepath}/{seg}.wav", format="wav")
        t1=t2
    os.remove(f"{filepath}/{username}.wav")

    source = f"{filepath}/"
    features = np.asarray(())
    count=1

    for f in os.listdir(source):
        sr,audio = read(source+f)
        vector   = extract_features(audio,sr)
        
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:    
            gmm = GMM(n_components = 5, covariance_type='diag',n_init = 3)
            gmm.fit(features)
            # dumping the trained gaussian model
            picklefile = f"{username}.gmm"
            cPickle.dump(gmm,open("trainedModels/" + picklefile,'wb'))
            print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)   
            features = np.asarray(())
            count = 0
        count = count + 1
    shutil.rmtree(filepath)

    return {"message": f"GMM Created"}


@app.post("/api/recognize")
async def register(file: UploadFile):
    with open("recognize/test.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    modelpath = "trainedModels/"

    gmm_files = [os.path.join(modelpath,fname) for fname in 
                os.listdir(modelpath) if fname.endswith('.gmm')]

    models    = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname 
                in gmm_files]

    error = 0
    total_sample = 0.0

    sr,audio = read("recognize/test.wav")
    vector   = extract_features(audio,sr)

    log_likelihood = np.zeros(len(models)) 

    for i in range(len(models)):
        gmm    = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    os.remove("recognize/test.wav")

    return {"message": f"Speaker Recognized as {speakers[winner]}"}
    