# Natural Opinion Dynamic 
In this baseline, the natural behavior of opinion dynamics without the gerrymandering effect has been investigated. Without the gerrymandering effect, each voter (node) is in its own district, and there is no representative. Thus, the opinions’ voters are affected just with the other connected voters in social graph. Therefore, the number of districts is equal to number of voters. In this test, we analyzed how much opinion distributions have been changed over several steps. For this reason, 70 graphs with different dimensions and 20 opinion distributions for each graph are generated. Then, two metrics are calculated: 
•	Metric one: The opinions change over 100 steps for each graph 
•	Metric two: The distance between the initial opinion and the final opinion after 100 steps
The following chart shows the process of this test: 

<img width="702" height="345" alt="image" src="https://github.com/user-attachments/assets/7f623d45-8c89-419e-8e22-f2fa4031a538" />

# “total_sum” Function:
This function computes the metric one. With the following formulation: 
For each graph:
	For each opinion:  
Sum ([New Opinion – old opinion] for 100 times) 

# “global_dist” Function:
This function computes the distance between the 100th opinion and the initial opinion for each graph with the following formulation: 
For each graph: 
	For each opinion: 
		Sum (initial opinion – 100th opinion)

