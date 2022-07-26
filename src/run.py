import os
import torch
import torch.nn as nn

def run(hp,data,model,criterion,ret_output=False,device="cuda:0"): 

    # convert to mag
    noisy_mag = data["noisy_mag"].to(device)
    noisy_phase = data["noisy_phase"].to(device)

    mask = model(noisy_mag)

    # masking
    estim_mag = noisy_mag*mask
    estim_spec = estim_mag * (noisy_phase*1j).to(device)

    # inverse
    estim_wav = torch.istft(estim_spec[:,0,:,:],n_fft = 512)

    loss = criterion(estim_wav,data["noisy_wav"].to(device),data["clean_wav"].to(device), alpha=hp.loss.wSDR.alpha).to(device)

    

def train(version,model,criterion, dataset_train,datset_test,epoch=10):
    modelsave_path = os.path.join("output",version,"chkpt")
    log_path = os.path.join("output",version,"log")

    os.makedirs(modelsave_path)
    os.makedirs(log_path)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=2,
        min_lr=1e-5)

    for epoch in range(num_epochs):
        model.train()
        train_loss=0
        for i, data in enumerate(train_loader):
            step +=data["noisy_mag"].shape[0]
            
            ## TODO
            loss = run(hp,data,model,criterion,device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           
            print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (data) in enumerate(test_loader):
                ## TODO
                estim, loss = run(hp,data,model,criterion,ret_output=True,device=device)
                test_loss += loss.item()

                print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), loss.item()))
                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            scheduler.step(test_loss)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

def eval(version, model, criterion, dir_eval) : 
    