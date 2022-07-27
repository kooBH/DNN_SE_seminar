import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt    
from matplotlib import cm
import numpy as np
import librosa

def get_output_wav(estim_mask,noisy_mag,noisy_phase,n_fft=512):
    estim_mag = estim_mask * noisy_mag
    estim_spec = estim_mag * (noisy_phase*1j)

    estim_wav = torch.istft(estim_spec, n_fft=n_fft)

    return estim_wav

def run(data,model,criterion): 

    # convert to mag
    noisy_mag = data["noisy_mag"]
    noisy_phase = data["noisy_phase"]

    mask = model(noisy_mag)

    # masking
    estim_mag = noisy_mag*mask
    estim_spec = estim_mag * (noisy_phase*1j)

    # inverse
    estim_wav = torch.istft(estim_spec[:,0,:,:],n_fft = 512)

    loss = criterion(estim_wav,data["noisy_wav"],data["clean_wav"])


    return loss

def infer(data,model,device="cuda:0") : 
    # convert to mag
    noisy_mag = torch.unsqueeze(data["noisy_mag"],0).to(device)
    noisy_phase = torch.unsqueeze(data["noisy_phase"],0).to(device)

    mask = model(noisy_mag)

    # masking
    estim_mag = noisy_mag*mask
    estim_spec = estim_mag * (noisy_phase*1j)

    # inverse
    estim_wav = torch.istft(estim_spec[:,0,:,:],n_fft = 512)

    estim_wav = estim_wav.detach().cpu().numpy()
    estim_wav = estim_wav/np.max(np.abs(estim_wav))

    return data["clean_wav"],data["noisy_wav"],estim_wav[0]


def train(model,dataset_train,dataset_dev,criterion,device="cuda:0"):
    # params
    batch_size = 10
    num_workers = 4
    num_epochs=30

    # init
    best_loss = 1e7

    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    dataloader_dev = torch.utils.data.DataLoader(dataset=dataset_dev,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5)

    model = model.to(device)

    history_train=[]
    history_test=[]

    test_loss =0.
    for epoch in (pbar := tqdm(range(num_epochs),desc="epoch")):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, data in enumerate(dataloader_train):
            for j in data : 
                data[j]=data[j].to(device)
            loss = run(data,model,criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           
        train_loss = train_loss/len(dataloader_train)

        pbar.set_postfix({"run":"train","train": train_loss,"dev":test_loss})

        history_train.append(train_loss)


        #torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (data) in enumerate(dataloader_dev):
                for j in data : 
                    data[j]=data[j].to(device)
                loss = run(data,model,criterion)
                test_loss += loss.item()

            test_loss = test_loss/len(dataloader_dev)
            scheduler.step(test_loss)
            
            if best_loss > test_loss:
                torch.save(model.state_dict(),"bestmodel.pt")
                best_loss = test_loss
            pbar.set_postfix({"run":"dev","train": train_loss,"dev":test_loss})

            history_test.append(test_loss)
    
    plt.figure()
    plt.title("dev loss")
    plt.plot(history_test)

def plot_spec(wav,title=""):

    if type(wav) != np.ndarray : 
        wav = wav.numpy()

    spec = librosa.stft(wav,n_fft=512)
    mag = np.abs(spec)
    mag = 10*np.log(mag)
    fig, ax = plt.subplots()
    im = plt.imshow(mag, cmap=cm.jet, aspect='auto',origin='lower')
    plt.colorbar(im)
    plt.clim(-80,20)
    plt.title(title)
    
    plt.xlabel('Time')
    plt.ylabel('Freq')