TLDR:
'*_compare.png':
    - Z500 vs Time
        - left: ground truth
        - right prediction made by graph ODE
    - Each figure contains 50 trajectories, each trajectory corresponds to one grid cell over a short time range of `[t, t+72]` 
     
'*_compare_scatter.png':
    - The comparison between prediction and ground truth at three time steps. 
    - Each dot corresponds to the Z500 value of one grid cell at the time step in the subfigure title. 

---

**Explanation of Shapes**
The input trajectories to new_dataLoader.py's load_data() are of shape
    `[#graph, #node, #time, #feat]`

In `new_dataLoader.py`, the `split_data()` produces an output named `series_list` which will be the ground truth trajectories used to compute the loss (given mode `interpolation`). This true trajectory is of shape, `[#traj, #time, #feat]`, where `#traj = #graph * #node`. Later on, this `#traj` will be chunked into different batches. But the shape of the true trajectories will remain three dimensions.

In the output, the trajectories have shape `[#sample, #traj, #time, #feat]`. This is because the variational encoder part computes a distribution for the hidden initial condition and the graph ODE would sample `#sample` initial conditions for each trajectory.  


