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


## Methodology:<br>
Below methoddology has been followed in this project.<br>
![Methodology](methodology.jpg)
<br>

There are 3 major pipelines in this project:<br>
![Architecture](architecture.jpg)
<br>


## Details on files & folders<br>
**This code repository is organised into following key components:**<br>
- README.md: The current file you are reading giving an overview of the project.<br>
- requirements.txt: Contains all the necessary libraries that should be installed in the local environment of Microsoft Visual Studio code. <br>
- helpers.py: Contains the 
- pipelines.py:
- feature_extraction_code.ipynb:
- eda_code.ipynb:
- model_building.ipynb:
- Reports:


## How to use this repository? <br>

For this project, I used both Google Collab & Microsoft Visual Studio for coding. Feature extraction was performed in Visual Studio code while EDA & Model Building was performed in Collab. My observation and recommendation is to build the entire project in Collab to leverage the free student compute it provides for one year.However, this can be built in local IDEs for small datasets for ease of usage and experimentations.<br>

In the following section, I have given the details as per the way I have organized and experimented with my codes. <br>

- Fork the repository <br>
- Clone your forked repo to your local <br>
```bash
git clone https://github.com/sundarram1608/carnatic_music_raga_identification_mir.git
```
- Open terminal and follow the below CLI prompts one by one<br>
```bash
cd “path to directory“
```

## References:<br>
[1] Serrà, J., Ganguli, K. K., Sentürk, S., Serra, X., & Gulati, S. (2016). Indian Art Music Raga Recognition Dataset (audio) (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7278511
