from typing import Any
import torch
import matplotlib.pyplot as plt
from ..sionna_torch import SionnaScenario
import math


class ChannelGenerator:
    def __init__(
        self,
        map_resolution = 1,
        n_rx = 1,
        n_tx = 1,
        frame_size = 1024,
        f_c = .92e9,
        bw = 30e3,
        noise_power_dB = None,
        direction = "uplink",
        target_total_p = False,
        minDist = 1,
        batch_size = 128,
        max_iters = 10,
        max_error = 0.1,
        seed = 42,
        lowEnd_Tolerance_dB = -2,
        dtype = torch.float32,
        device = None,
        debug = False,
        *args: Any,
        **kwds: Any
    ) -> None:
        self.config = locals()
        self.device = device

        self.sionna_dev = device
        self.sionna = SionnaScenario(n_bs=n_rx, n_ut=n_tx, batch_size=batch_size, n_time_samples=frame_size, f_c=f_c, bw=bw, noise_power_dB=noise_power_dB, seed=seed, dtype=dtype.to_complex(), device=self.sionna_dev)
        self.avg_filter = torch.nn.Sequential(
            torch.nn.ReplicationPad1d(27//2),
            torch.nn.AvgPool1d(kernel_size=27, stride=1)
        )

    def __call__(
        self, 
        scenario_map,
        tx_xyz,
        rx_xyz,
        target_pow_db,
        los_requested = False,
        *args: Any, 
        **kwds: Any
    ) -> Any:
        assert tx_xyz.shape[0] == 1 and tx_xyz.shape[1] == self.config['n_tx'], "Tx shape incorrect"
        assert rx_xyz.shape[0] == 1 and rx_xyz.shape[1] == self.config['n_rx'], "Rx shape incorrect"
        assert rx_xyz.shape[1] == len(target_pow_db) or len(target_pow_db) == 1, "Length of target_pow_db must be either 1 or number of receivers sent"

        # Convert target to required forms
        target_pow_linear = 10**(target_pow_db/10) #SNR in dB to PowerGoal Linear Space

        # Estimate the channel gain along each path
        power_est_db, rx_xyz_sprayed = self._estimate_path_gain(tx_xyz, rx_xyz, scenario_map, los_requested)

        # Get a best initial guess for receiver locations that could meet the requirement
        rx_near_optimal = self._get_initial_guess(target_pow_linear, power_est_db, rx_xyz, rx_xyz_sprayed)

        # Loop until channel meeting requirements is found
        h_T, rx_xyz = self._gen_matching_channels(rx_near_optimal, tx_xyz, scenario_map, target_pow_db, los_requested)
        
        return h_T, rx_xyz
        
    def get_noise_power(self):
        return self.sionna.noise_power_db

    def _estimate_path_gain(self, tx_xyz, rx_xyz, scen_map, los_requested):
        map_size = scen_map.shape[0]
        
        #1. First find the minimum location of the receiver
        distance_path = torch.cdist(tx_xyz[...,:2], rx_xyz[...,:2], p=2)
        distance_ratio = (self.config['minDist']/distance_path).reshape(1,-1,1)
        min_points = torch.cat((((1 - distance_ratio) * tx_xyz[...,:2] + distance_ratio * rx_xyz[...,:2]), 
                                (rx_xyz[...,2:])), dim = 2)
        
        #2. Find the Outpost point by extending the line between the Tx and Rx beyond its original length
        extended_distance = math.sqrt(2*(map_size**2)) + distance_path
        distance_ratio = (extended_distance/distance_path).reshape(1,-1,1)
        max_points = torch.cat((((1.0 - distance_ratio) * tx_xyz[...,:2] + distance_ratio * rx_xyz[...,:2]), 
                                (rx_xyz[...,2:])), dim = 2)
        
        #3. Now verify if those new endpoints lie within the boundary of the map
        map_boundaryA = torch.tensor([[[0,0],[map_size-1,0],[map_size-1,map_size-1],[0,map_size-1]]], device=self.device)
        map_boundaryB = torch.cat((map_boundaryA[:, 1:], map_boundaryA[:, :1]), dim=1)
        max_points = self._get_poi(map_boundaryA,map_boundaryB,tx_xyz,max_points)
        
        #4. Spray the points between the min and Max Rx locations based on resolutions
        tx_xyz_replicated = tx_xyz.repeat(self.config['batch_size'],1,1) 
        spray = torch.linspace(0, 1, self.config['batch_size'], device=self.device)[:,None]
        x = (max_points[:,:,0] - min_points[:,:,0]) * spray + min_points[:,:,0]
        y = (min_points[:,:,1] - max_points[:,:,1])/(min_points[:,:,0]-max_points[:,:,0])*(x - min_points[:,:,0]) + min_points[:,:,1]
        y_vertical = (max_points[:,:,1] - min_points[:,:,1]) * spray + min_points[:,:,1]
        y = torch.where(torch.isnan(y), y_vertical, y)
        
        z = max_points[:,:,2].repeat((self.config['batch_size'],1))
        sprayed_tensor = torch.stack((x,y,z),dim=2)
        
        #5. Call the channel gain function and obtain the Power and the Channel_Z values
        self.sionna.update_topology(tx_xyz_replicated, sprayed_tensor, scen_map, map_resolution=self.config['map_resolution'], direction=self.config['direction'], los_requested=los_requested)
        h_T, rx_pow = self.sionna.generate_channels()
        rx_pow_db = 10*torch.log10(rx_pow).to(self.device)
        h_T = h_T.to(self.device)
        rx_pow_db = rx_pow_db.to(self.device).squeeze((2,4)) #batch_size,n_rx,x_tx 
        
        # Take the PowerDB and filter it, and a form a smooth curve
        filtered_power_db = self.avg_filter(rx_pow_db.permute((2,1,0))).permute((2,1,0))
        
        # Convert the PowerDB and FilteredPowerDB back to SNR for plotting puposes
        filtered_snr_db = filtered_power_db - self.sionna.noise_power_db
        rx_snr_db = rx_pow_db - self.sionna.noise_power_db
        
        #Plot the data if the flag is set
        if self.config['debug']:
            clipped_dist = torch.cdist(tx_xyz_replicated[:,:,0:2], sprayed_tensor[:,:,0:2], p=2)
            self.plotSNRvsDist(filtered_snr_db, clipped_dist,rx_snr_db,clipped_dist)
        
        return filtered_power_db, sprayed_tensor

    def _get_initial_guess(self, target_pow_linear, power_est_db, rx_xyz, rx_xyz_sprayed):
        #Convert the FilterPower in DB to Linear Power i.e., Linear Space
        power_est_linear = 10**((power_est_db)/10) 
        tolerance_linear = 10**((self.config['lowEnd_Tolerance_dB'])/10)
        
        #The new check flag tells us whether we want to replicate the Power across the RxTower or distribute it
        if self.config['target_total_p']:
            assert len(target_pow_linear) == 1, "target_total_p=True only accepts a single target value"
            assert target_pow_linear > (tolerance_linear * torch.sum(power_est_linear.min(0)[0])), "The TargetPower is not in the feasible SNR region of the Rx Towers"
            target_pow_linear_perchan = self._simple_power_solver(power_est_linear, target_pow_linear, rx_xyz, rx_xyz_sprayed)
        else: 
            assert len(target_pow_linear) == self.config['n_rx'] , "Target SNR wrong shape"
            assert (target_pow_linear[None,:,None] > (tolerance_linear * power_est_linear.min(0)[0])).all(), "The TargetPower is not in the feasible SNR region of the Rx Towers"
            target_pow_linear_perchan = target_pow_linear[None,:,None]
            
        rms_errors = torch.sqrt((target_pow_linear_perchan - power_est_linear)**2)
        index = rms_errors.argmin(0, keepdim=True)
        rx_xyz = torch.gather(rx_xyz_sprayed, 0, index.repeat((1,1,3)))

        return rx_xyz

    def _gen_matching_channels(self, rx_xyz, tx_xyz, scen_map, target_pow_db, los_requested):
        # Batch the locations for multi-test
        tx_xyz_replicated = tx_xyz.repeat(self.config['batch_size'],1,1) 
        rx_xyz_replicated = rx_xyz.repeat(self.config['batch_size'],1,1) 

        
        # Try max_iters times to find solution
        iteration_val = 0
        while(iteration_val < self.config['max_iters']):
            # Call the channel gain function
            self.sionna.update_topology(tx_xyz_replicated, rx_xyz_replicated, scen_map, map_resolution=self.config['map_resolution'], direction=self.config['direction'], los_requested=los_requested)
            h_T, rx_pow_linear = self.sionna.generate_channels()
            h_T = h_T.to(self.device)
            rx_pow_linear = rx_pow_linear.to(self.device).squeeze((2,4)) #128,n_rx,x_tx  
            
            # Find the index of the closest value of the Power to the targetPowerLinear
            if self.config['target_total_p']:
                total_pow_db = 10*torch.log10(torch.sum(rx_pow_linear, (1,2)))
                rms_errors = torch.sqrt((target_pow_db - total_pow_db)**2)
                fitting, passing_idx = torch.max((rms_errors < self.config['max_error']).int(), 0)
                fitting = fitting.bool()
            else:
                # Take first BS for each channel that meets requirements. We can do this since BSs are uncorrelated.
                rms_errors = torch.sqrt((target_pow_db[None,:,None] - 10*torch.log10(rx_pow_linear))**2)
                fitting, passing_idx = torch.max((rms_errors < self.config['max_error']).int(), 0)
                passing_idx = passing_idx.flatten()
                fitting = fitting.bool()
            
            if torch.all(fitting):
                if self.config['debug']: 
                    print('The Target is found!!')
                    print("The near Optimal Rx locations are: ", rx_xyz)
                    print("The near Optimal SNRs are: ", 10*torch.log10(rx_pow_linear[passing_idx, torch.arange(h_T.shape[1])]))
                return h_T[passing_idx, torch.arange(h_T.shape[1])][None], rx_xyz
            else:
                if self.config['debug']: 
                    print("Current iter: ", iteration_val)
                    print("Current small value: ", rms_errors.min(0).values)
                iteration_val += 1

        assert False, "max_iters reached without solution"            

    ###############      (x1,y1)        (x2,y2)    (x3,y3)  (x4,y4) ####################
    def  _get_poi(self, mapBoundaryA, mapBoundaryB, txLoc, outPostLoc):

        x1y1_x2y2 = mapBoundaryA - mapBoundaryB
        x1y1_x3y3 = mapBoundaryA - txLoc[...,:2]
        x3y3_x4y4 = txLoc[...,:2] - outPostLoc[...,:2]

        x1_x2 = x1y1_x2y2[...,0].flatten()
        y1_y2 = x1y1_x2y2[...,1].flatten()
        
        x1_x3 = x1y1_x3y3[...,0].flatten()
        y1_y3 = x1y1_x3y3[...,1].flatten()

        x3_x4 = x3y3_x4y4[...,0].flatten()
        y3_y4 = x3y3_x4y4[...,1].flatten()

        Det = torch.outer(x1_x2,y3_y4) - torch.outer(y1_y2,x3_x4)
        Det += torch.finfo(torch.float64).eps
        
        t = (torch.outer(x1_x3,y3_y4) - torch.outer(y1_y3,x3_x4))/Det
        u = ((y1_y2 * x1_x3) - (x1_x2 * y1_y3))[:,None]/Det

        completeIndex = (((t >= 0) & (t <= 1)) & ((u >= 0) & (u <= 1)))
        
        x = mapBoundaryA[...,0].T + t * (-1.0 * x1_x2)[:,None]
        y = mapBoundaryA[...,1].T + t * (-1.0 * y1_y2)[:,None]
        z = outPostLoc[...,2]
        newPoints = torch.stack((x,y),dim=2)
        
        #this is for an edge case where the line may intersect with multiple edges, 
        #for example, a diagonal line
        _, passing_idx = torch.max(completeIndex, 0)

        return torch.cat((newPoints[passing_idx, torch.arange(outPostLoc.shape[1])], z.T), dim=1)[None,:]

    def _simple_power_solver(self, powerLinear, targetPower, rxLoc, sprayedRx):
        # first get initial state power and delta
        index = torch.cdist(sprayedRx[...,None,:2],rxLoc[...,None,:2],p=2).argmin(0)
        currentRxPow = torch.gather(powerLinear,0,index.reshape(1,-1,1))
        delta = targetPower - torch.sum(currentRxPow)
        
        # Set direction to move in
        if delta < 0: #Here the delta will be negative
            #move away from Tx
            boundaryTensor = powerLinear.min(dim=0, keepdim = True)[0]
        elif delta > 0: #Here the delta will be positive
            #Bring closer to Tx
            boundaryTensor = powerLinear.max(dim=0, keepdim = True)[0]
        
        powerCapacity = torch.sum(torch.abs(currentRxPow - boundaryTensor))
        if torch.abs(delta)/powerCapacity > 1.0:
            return boundaryTensor
        else:
            shareRate = torch.abs(currentRxPow - boundaryTensor)/powerCapacity
            currentRxPow = currentRxPow + (shareRate * delta)
            return currentRxPow    

    def plotSNRvsDist(self, filteredSNR, dist1, unfilteredSNR, dist2):
        for i in range(dist1.shape[2]):
            fig = plt.figure(figsize=(8,8))
            fSNRindividual = filteredSNR[:,i,:].squeeze().tolist()
            unfSNRindividual = unfilteredSNR[:,i,:].squeeze().tolist()
            d_clipped = dist1[:,:,i].tolist()
            d_unclipped = dist2[:,:,i].tolist()
            
            plt.plot(d_unclipped,unfSNRindividual, label='Unfiltered SNR')
            plt.plot(d_clipped,fSNRindividual, label='Filtered SNR')
            plt.legend()
            plt.xlabel("Distance (meters)", fontsize = 12)
            plt.ylabel("SNR [dB]", fontsize = 12)
            plt.savefig(f"DistVsSNRplt1_{i}.jpg")