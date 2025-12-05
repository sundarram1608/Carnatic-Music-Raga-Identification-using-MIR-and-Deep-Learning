# Carnatic Music Raga Identification using MIR + Deep Learning

**Project Motivation & Description**<br>
Carnatic music is one of the oldest and most historic music systems in the world. Raga is the single most crucial melodic framework of this music art form that provides the rules and notes for a Carnatic composition.Learning to identify a song’s raga is a core competency developed during an Indian Carnatic music education. Therefore, Raga recognition is an important step in computational musicology & MIR as far as Indian Carnatic music is considered. <br> <br> This Project aims to leverage MIR techniques to extract and analyze Carnatic audio for 8 melakartha ragas, followed by building a fusion deep learning model (CNN+LSTM) for Raga classification.

## Dataset:
The audio dataset is currently not included in this repository due to heavy file size.<br> 
Link to dataset:  <br>
[Indian Art Music Raga Recognition datasets](https://compmusic.upf.edu/datasets) [1]<br>

Link to the sample dataset: <br>
[Sample dataset](https://drive.google.com/drive/folders/14oVxOAg2Mu-I-rml4iA3Bmp-ABV8P8Nu?usp=share_link)<br>

The Sample dataset consisits 8 Melakarta Ragas, namely hanumathodi, harikaambhoji, kaamavardhini, kalyani, kharaharapriya, maayamaalavagowlai, shankaraabharanam & shanmugapriya. Minimum of 5 songs from each of these 8 ragas are selected based on their audio size and curated in the sample dataset.


## Project Structure:<br>

![Methodology]()



The project is organised into several key components:<br>
- README.md: The current file you are reading giving an overview of the project.<br>
- German_bank.csv: The loan default dataset that is provided. <br>
- code_file.ipynb: This is the python code notebook (Jupyter notebook) that detail the following:<br> 
	1. Library Import: <br>
		-> Setting up your Python environment with necessary libraries.<br>
	2. Data Ingestion: <br>
		-> Loading the dataset into your workspace.<br>
	3. Exploratory Data Analysis (EDA): Assessing data quality and characteristics through: <br>
		-> Overview and statistical summaries.<br>
   		-> Null value checks.<br>
   		-> Analysis of class balance.<br>
   		-> Correlation between features through visualisation plots like heatmap & pair plot for multicollinearity & correlation insights.<br>
   		-> Histograms for distribution insights.<br>
   		-> Outlier Analysis.<br>
	4. Data Preprocessing:<br>
   		-> Encoding categorical variables.<br>
   		-> Scaling numerical data to standardise features.<br>
	5. ML Model Building and Optimization:<br>
   		-> Splitting data into training and testing sets.<br>
   		-> Initial model fitting with basic parametric and tree-based models, including cross-validation.<br>
   		-> Hyper-parameter tuning of the most promising models.<br>
   		-> Feature importance analysis to refine model inputs.<br>
   		-> Re-evaluation using only the important features (a reduced feature set).<br>
   		-> Combining best individual models into an ensemble for improved performance.<br>
	6. ML Model Recommendation: <br>
		-> Analysing and comparing model performances to choose the best model for deployment.<br>
- final_report.pdf: Final Report in PDF format with Introduction, Methods and Materials, Results, Discussion & Conclusion.<br>


## References:<br>
[1] Serrà, J., Ganguli, K. K., Sentürk, S., Serra, X., & Gulati, S. (2016). Indian Art Music Raga Recognition Dataset (audio) (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7278511
