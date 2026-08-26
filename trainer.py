import os
import pickle
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as npl
import data_loader
import plots
import utils
import datetime
from model import PGTNet


class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False, data_print=False):
        super(Train_test, self).__init__()
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.print = data_print
        self.save_dir = "results_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)
        if dataset == 'samson':
            self.P, self.L, self.col = 3, 156, 95
            self.patch, self.dim = 1, 24
            self.LR, self.EPOCH = 3e-4, 900
            self.para_re, self.para_sad = 0.1, 4.9
            self.weight_decay_param = 2e-3
            self.batch = 1
            self.data = data_loader.Data(dataset)
            self.loader = self.data.get_loader(batch_size=(self.col ** 2))
            self.bundle_mean = self.data.get("bundle_mean").float() 
            self.init_bundle_list = self.data.get("bundle_list")
            self.bundle_sizes = self.data.get("bundle_sizes")
        elif dataset == 'jasper':
            self.P, self.L, self.col = 4, 198, 100
            self.patch, self.dim = 1, 32
            self.LR, self.EPOCH = 5e-3, 700
            self.para_re, self.para_sad = 0.8, 0.2            
            self.weight_decay_param = 4e-5 
            self.batch = 1 
            self.data = data_loader.Data(dataset)
            self.loader = self.data.get_loader(batch_size=(self.col ** 2))
            self.bundle_mean = self.data.get("bundle_mean").float()
            self.init_bundle_list = self.data.get("bundle_list")
            self.bundle_sizes = self.data.get("bundle_sizes")
        elif dataset == 'urban':
            self.P, self.L, self.col = 5, 162, 307
            self.patch, self.dim = 1, 40
            self.LR, self.EPOCH = 3e-3, 200     
            self.para_re, self.para_sad = 0.8, 0.2
            self.weight_decay_param = 4e-5
            self.batch = 1
            self.data = data_loader.Data(dataset)
            self.loader = self.data.get_loader(batch_size=(self.col ** 2))
            self.bundle_mean = self.data.get("bundle_mean").float()
            self.init_bundle_list = self.data.get("bundle_list")
            self.bundle_sizes = self.data.get("bundle_sizes")
        else:
            raise ValueError("Unknown dataset")

    def run(self):
        bundles_list = [b.to(self.device) for b in self.init_bundle_list]
        bundle_sizes = self.bundle_sizes
        
        net = PGTNet(P=self.P, L=self.L, size=self.col,
                          patch=self.patch, dim=self.dim, 
                          init_bundles=bundles_list, bundle_sizes=bundle_sizes).to(self.device)
        net.apply(net.weights_init)
        loss_func = nn.MSELoss(reduction='mean')  
        loss_func2 = utils.SAD(self.L) 
        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        if not self.skip_train:
            time_start = time.time()
            net.train()
            for epoch in range(self.EPOCH):

                for i, (x, _) in enumerate(self.loader):
                    x = x.transpose(1, 0).view(1, -1, self.col, self.col).to(self.device)
                    bundle_mean = self.bundle_mean.to(self.device)
                    abu_est, re_result, edm_est, _, _ = net(x, bundle_mean)
                    loss_re = self.para_re * loss_func(re_result, x.view(1, self.L, -1).transpose(1, 2)) # RE Loss
                    loss_sad = loss_func2(re_result, x.view(1, self.L, -1).transpose(1, 2)) # SAD Loss
                    loss_sad = self.para_sad * torch.mean(loss_sad).float()
                    total_loss = loss_re + loss_sad
                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()
                    if epoch % 10 == 0 and self.print:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)
                scheduler.step()
            time_end = time.time()
            if self.print:
                print('Total computational cost:', time_end - time_start)

        else:
            with open(self.save_dir + 'weights_new.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))
        
        # Testing ================
        net.eval() 
        with torch.no_grad():
            x = self.data.get("hs_img")
            x = x.transpose(1, 0).view(1, -1, self.col, self.col).to(self.device)
            bundle_mean = self.bundle_mean.to(self.device)
            abu_est, re_result, edm_est, intra_weights, _ = net(x, bundle_mean)

            intra_weights_np = [w.cpu().numpy() for w in intra_weights]

        abu_est = abu_est.view(self.col, -1, self.P).detach().cpu().numpy()
        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()
        true_endmem = self.data.get("end_mem").numpy()
        edm_est = edm_est.detach().cpu().numpy()
        est_endmem = edm_est
        est_endmem = np.mean(est_endmem, axis=0)
        set_edm = true_endmem.transpose()
        est_edm = est_endmem.transpose()
        ref_list = np.arange(0, set_edm.shape[0])
        est_list = np.arange(0, est_edm.shape[0])
        ref2est_table = np.zeros(set_edm.shape[0])

        for i in range(set_edm.shape[0]):
            best_dis = np.ones((len(ref_list), len(est_list))) * 100.
            ref = set_edm[ref_list].copy()
            est = est_edm[est_list].copy()
            for j in range(len(ref_list)):
                dis = np.arccos(np.dot(ref[j], est.T) / (npl.norm(ref[j]) * npl.norm(est, axis=1) + 1e-6))
                best_dis[j][dis < best_dis[j]] = dis[dis < best_dis[j]]
            best_match = np.argwhere(best_dis == best_dis.min())[0]
            ref_absolute_index = ref_list[best_match[0]]
            est_absolute_index = est_list[best_match[1]]
            ref2est_table[ref_absolute_index] = est_absolute_index
            ref_list = np.delete(ref_list, best_match[0])  # modify the list
            est_list = np.delete(est_list, best_match[1])

        abu_est = abu_est[:, :, ref2est_table.astype(int)]
        est_endmem = est_endmem[:, ref2est_table.astype(int)]
        edm_est = edm_est[:, :, ref2est_table.astype(int)]

        x = x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        re_result = re_result.view(self.col, self.col, -1).detach().cpu().numpy()
        re = utils.compute_re(x, re_result)
        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)

        if self.print:
            print("RE:", re)

            print("Class-wise RMSE value:")
            for i in range(self.P):
                print("Class", i + 1, ":", rmse_cls[i])
            print("Mean RMSE:", mean_rmse)
            print("Class-wise SAD value:")
            for i in range(self.P):
                print("Class", i + 1, ":", sad_cls[i])
            print("Mean SAD:", mean_sad)
            plots.plot_abundance(target, abu_est, self.P, self.save_dir, rmse_cls)
            plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir, sad_cls)

        with open(self.save_dir + "log.csv", 'a') as file:
            file.write(f"LR: {self.LR}, ")
            file.write(f"EPOCH: {self.EPOCH}, ")
            file.write(f"Batch: {self.batch}, ")
            file.write(f"para_re: {self.para_re}, ")
            file.write(f"para_sad: {self.para_sad}, ")
            file.write(f"RE: {re:.4f}, ")
            file.write(f"SAD: {mean_sad:.4f}, ")
            for i in range(self.P):
                file.write(f"Class{i}_sad: {sad_cls[i]:.4f}, ")
            file.write(f"RMSE: {mean_rmse:.4f}, ")
            for i in range(self.P):
                file.write(f"Class{i}_mse: {rmse_cls[i]:.4f}, ")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"TIME:{current_time}\n")
        plots.plot_intra_weights(intra_weights_np, self.save_dir)
# =================================================================

if __name__ == '__main__':
    pass
