# CTESN_PSCC
- Objective:
  - Create Continuous Time Echo State Networks (CTESN) that can be used for times series prediction. These models will be used as surrogates for physical system simulations. The models' ability to predict over time is what allows them to be used as surrogtates for physcial simulations.
- Project Focus:
  - This specific aspect of the project is focused on improving CTESN performance through dynamic weights.
  - Dynamic weighting is the process of training the weights within the reservior of the model as the outputs weights are being trained.
  - Principle Neuron Reinforcement (PNR) is the dynamic weight algorithm investigated here
- Repository Work:
  - In this repository a sample ESN is created to predict the Mackey-Glass chaotic time series
  - PNR is then impleneted on top of the simple ESN and the performance is compared to the original performance
# CTESN_GBS
- Objective:
  - Using knowledge of gradients select the initial training sampling points to help more effectively train the CTESN
- Background:
  - Gradient provides information on the trend of the function at a point
  - Regions with large gradients have large changes and therefore are regions of interest and require more initial training samples
- Repository Work:
  - The work in the GradientBasedSampling folder contains the code which outlines and executes the gradient based sampling approach along with training a CTESN model implementing it
  - There is also a sample file which runs on the Rosenbrock 2D function to evaluate the effectiveness of this improvement approach.

