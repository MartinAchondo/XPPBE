import numpy as np
import torch
import sys

import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float64)

np.random.seed(78)
torch.manual_seed(-42)


def is_inside_mol_dom(x):
    #only for born ion right now
    a = 2 #born ion radius
    r = np.linalg.norm(x)
    if (r<a):
        return True
    else:
        return False

def born_ion_reg_true_soln(x_a):

    eps_m = 1.
    eps_s = 78.
    kappa_s = 0.918168
    kappa2 = kappa_s**2
    e_c = 4.803242384e-10
    k_B = 1.380662000e-16
    Temperature  = 300.0
      
    
    r = np.linalg.norm(x_a)
    a = 2.;
    C = e_c*e_c*1e8/(k_B*Temperature)
    alpha = np.sqrt(kappa2/eps_s)

    if (r < a):
        val = (C/a)*(-1.0/eps_m + 1.0/(eps_s*(1 + alpha*a)))
    else:
        val = (C/r)*(-1.0/eps_m + (np.exp(alpha*(a-r)))/(eps_s*(1 + alpha*a))  )
    
    return val

def sample_ball(npoints, ndim=3, rad = 10):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec = vec.T

    r = np.random.rand(npoints,1)
    for i in range(npoints):
      vec[i] = rad*pow( r[i],1/2)*vec[i,:]
    
    return vec



def init_model_from_true_soln(model,pts,u_true,npoints):
    

    input_dim=3
    train_data = torch.cat((pts, u_true), -1)
    

    loss_fn = torch.nn.MSELoss()  # mean square error

    optimizer = torch.optim.Adam((model.parameters()), lr=0.001)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    n_epochs = 500   # number of epochs to run
    batch_size = 10  # size of each batch
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    Tr_loss = []
    for epoch in range(n_epochs):
    
        tr_loss = 0.
        for _, train_data_batch in enumerate(train_loader):
            # take a batch
            
            X_batch = train_data_batch[:,:input_dim]
            y_batch = train_data_batch[:,input_dim:]
            # forward pass
            y_pred = model(X_batch)
            
            loss = loss_fn(y_pred, y_batch)
            tr_loss = tr_loss + float(loss) * train_data_batch.shape[0]

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()
            


        StepLR.step()
        tr_loss = tr_loss / len(train_loader.dataset)
        if epoch%50 ==0:
            print('Loss at epoch {:d} is {:.4e}'.format(epoch,tr_loss))
            torch.save(model.state_dict(), './data/net_PDE' + '_' + str(epoch) + 'epoch.pth')

        Tr_loss.append(tr_loss)
        
    return Tr_loss
   


boundary_radius = 10 #This is the boundary enclosing the entire system. Not the interface!

# Set number of data points
npoints_sol_dummy = 4000


#Draw points from inside the domain
pts_sol_dom_dummy = sample_ball(npoints_sol_dummy,ndim=3, rad = boundary_radius)
indices_s_dummy = []

#Remove points which are in mol dom
for i,pt in enumerate(pts_sol_dom_dummy):
    if is_inside_mol_dom(pt):
        pass
    else:
        indices_s_dummy.append(i)
                    
       
npoints_solvent = len(indices_s_dummy)

pts_s_dom = pts_sol_dom_dummy[indices_s_dummy,:]



u_true = np.zeros((npoints_solvent,1))
for i in range(npoints_solvent):
    pt = pts_s_dom[i,:]
    u_true[i] = born_ion_reg_true_soln(pt)

pts_s_dom = torch.tensor(pts_s_dom)
u_true = torch.tensor(u_true)


nwidth=100
model_s = torch.nn.Sequential(
    torch.nn.Linear(3, nwidth), torch.nn.Tanh(),
    torch.nn.Linear(nwidth, nwidth),torch.nn.Tanh(),
    torch.nn.Linear(nwidth, nwidth),torch.nn.Tanh(),
    torch.nn.Linear(nwidth, nwidth),torch.nn.Tanh(),
    torch.nn.Linear(nwidth, nwidth),torch.nn.Tanh(),
    torch.nn.Linear(nwidth, 1)
)


Tr_loss = init_model_from_true_soln(model = model_s,pts=pts_s_dom,u_true=u_true,npoints=npoints_solvent)

torch.save(model_s.state_dict(), 'pbe_s_dom_model.pth')

plt.plot(Tr_loss)
plt.savefig('pbe_loss.pdf')
plt.close()

#Set mode to eval
model_s.eval()


xpts = np.linspace(2,10,200)
pts = np.zeros((len(xpts),3))
pts[:,0] = xpts
pts =  torch.tensor(pts)


soln_NN = model_s(pts)

soln_true = np.zeros(200,)
for i,pt in enumerate(pts):
   soln_true[i] = born_ion_reg_true_soln(pt)



pl1 = plt.plot(xpts,soln_NN.detach().numpy(),label='NN')
pl2 = plt.plot(xpts,soln_true, label='True')
plt.legend(loc="upper left")

plt.savefig('pbe_nn_vs_true.pdf')