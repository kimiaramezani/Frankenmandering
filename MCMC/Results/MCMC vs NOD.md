
# Description
In this report, we compared the results of MCM with natural opinion dynamics. Moreover, we tested two experiments, with and without normalization for two different DRFs: f1 and withsco. The setup for NOD and MCMC is as follows: 

common setup : 
DRF1 = f1  
DRF2 = DRF with social network
Graph size = 12 
steps: 50
generate 20 opinions for each graph

specific setup:

NOD: number of districts  = number of nodes
MCMC: number of districts = number of voter / 3
# Summary of the results

|DRF| Normalized | No normalized|
|------|------------|------------|
| DR1| <img width="567" height="455" alt="normal f1" src="https://github.com/user-attachments/assets/3b4cee05-643d-4ba4-b6bc-58defd5bb810" />|<img width="567" height="455" alt="no normal _f1" src="https://github.com/user-attachments/assets/9cb0df4f-03c3-4949-a9e5-4a4d296c1509" />|
|DRF2| <img width="567" height="455" alt="normal-with sco" src="https://github.com/user-attachments/assets/3d2e7a6d-6640-4797-92b3-47113614f6ff" /> |<img width="567" height="455" alt="no-normal_ withsco" src="https://github.com/user-attachments/assets/8bc7d03c-f659-4c57-b369-3231573c92a2" />|

