# Natural_OD_stat.ipynb
This part shows the final statistical results for investigation natural behavior of opinion dynamic without gerrymandering effect for 70 graphs and 20 opinion distributions for each one. 
# 1-	Metric one: Total opinion changes
Firstly, we show the frequency of different opinion distances across 1400 (70 graphs and 20 total distances) data. Following histogram shows distribution is right skewed with most distances clustered near very small values. Moreover, smooth KDE shows a single dominant peak around ~0.001–0.0015.


<img width="643" height="512" alt="image" src="https://github.com/user-attachments/assets/d28e95ca-a5f9-4b5b-828f-6268ad49869b" />

Following box plot illustrates the frequency of total distances for each graph. It can be seen that distances across 70 graphs vary moderately, with most medians between 0.001–0.002. Also, some graphs show wider spread and a few outliers.

<img width="975" height="325" alt="image" src="https://github.com/user-attachments/assets/f90daacf-3b29-4bbe-bb81-eda83b7df978" />

The line plot below shows the mean and variance of 20 total distances for each graph. Means remain low with mild fluctuations across graphs and variances follow the same pattern, showing generally small variability.

<img width="975" height="496" alt="image" src="https://github.com/user-attachments/assets/002a5aad-87ee-4e59-ad4d-15653bd192d7" />


# 2- Metric two: Global distance between initial opinion and last opinion for each graph
Firstly, we show the frequency of different global distances across 1400 (70 graphs and 20 global distances) data. The histogram depicts right-skewed distribution with a peak around ~0.8–1.0 and long tail suggests occasional much larger distances.

<img width="593" height="472" alt="image" src="https://github.com/user-attachments/assets/cec7c759-5280-4488-b8e3-1f11f6f6cf4d" />

Following box plot illustrates the frequency of global distances for each graph. It can be seen there is a high variation across graphs, with medians ranging roughly 0.5–2.5. Also, many graphs show large spreads and numerous outliers.

<img width="975" height="329" alt="image" src="https://github.com/user-attachments/assets/3867a4ff-b8c3-435b-b82b-c9bd63f7b50c" />

The line plot below shows the mean and variance of 20 global distances for each graph. Means fluctuate significantly across graphs, which reflects high variability. Furthermore, variances track the means which indicates consistent differences in spread.

<img width="897" height="470" alt="image" src="https://github.com/user-attachments/assets/9eac16f4-b3e9-4996-a238-4ef0299946b7" />


