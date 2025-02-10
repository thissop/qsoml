# QSOML

- Goal is to implement the redshift downsampling portion of the spender autoencoder 

### Tasks
- [x] Create small data batch. 
- [ ] Add weights (based on stuff like water lines, etc.)
- [ ] Add normalizing flow 
- [ ] Add downsampling/interpolation layer 
- [ ] Custom Loss Function

### Dated Notes

**2/7/2025**
- Made mini data batch 
- Tried to do downsampling layer, waiting
- Friday to Do: 
    - [ ] Process data batch (csv into zip)
    - [ ] Add weights
    - [ ] Make sure downsampling is working 
    - [ ] Extra time? Other tasks (normalizing flow and custom loss function)

**2/10/2025**
- Got lots of progress, but having issue with ptxas, likely due to missing CUDA toolkit or incorrect GPU configuration.