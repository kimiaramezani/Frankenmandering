# Natural Opinion Dynamic 
In this baseline, the natural behavior of opinion dynamics without the gerrymandering effect has been investigated. Without the gerrymandering effect, each voter (node) is in its own district, and there is no representative. Thus, the opinions’ voters are affected just with the other connected voters in social graph. Therefore, the number of districts is equal to number of voters. In this test, we analyzed how much opinion distributions have been changed over several steps. For this reason, 70 graphs with different dimensions and 20 opinion distributions for each graph are generated. Then, two metrics are calculated: 
•	Metric one: The opinions change over 100 steps for each graph 
•	Metric two: The distance between the initial opinion and the final opinion after 100 steps
The following chart shows the process of this test: 

<img width="702" height="345" alt="image" src="https://github.com/user-attachments/assets/7f623d45-8c89-419e-8e22-f2fa4031a538" />

# “total_sum” Function:
### Metric 1: Average Opinion Change

This metric quantifies how opinions evolve over time within a graph.

**Computation**
1. For each graph  
2. For each opinion  
3. Compute the change between the new and old opinion values  
4. Sum the changes across all nodes  
5. Normalize by the number of nodes  
6. Normalize by the total number of steps  

**Formula**  
distance = Σ (new_opinion − old_opinion) / number_of_nodes  
metric_1 = distance / total_steps

# “global_dist” Function:
### Metric 2: Initial–Final Opinion Distance

This metric captures how much opinions change from the beginning to the end of the simulation.

**Computation**
1. For each graph  
2. For each opinion  
3. Compute the difference between the initial opinion and the final (100th-step) opinion  
4. Sum the differences across all opinions  
5. Normalize by the total number of steps  

**Formula**  
distance = Σ (initial_opinion − last_opinion)  
metric_2 = distance / total_steps


