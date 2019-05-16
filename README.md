# Capstone Network Opitimizer Project


This project aims to help Highmark with optimizing the providers network, so that patients can pay the least cost, while getting a certain level of service. At the same time, other constraints, such as the legal constraints, will also be satisfied.

## File Structure:
Raw Data Sets<br />
-Programming File<br /> 
&ensp; 	- networkoptimizer (Django Application)<br /> 
&ensp; &ensp; 		- optimizer (Optimizer Application)<br /> 
&ensp; &ensp; - templates (Frontend Files)<br />
&ensp; &ensp; - urls.py (RESTful Routing File)<br />
&ensp; &ensp; - views.py (Backend File)<br />


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Python 3.7
Django 2.1.3
numpy 1.16.2
pandas 0.23.4
cvxopt 1.2.3
cvxpy 1.0.21
```
You can also run the command below to install all dependencies.

```
pip install -r requirements.txt
```


### Deploy to Local Server

A step by step series of examples that tell you how to get a development env running

cd to the directory networkoptimizer, which contains manage.py

```
python manage.py runserver
```


## Authors

* **Askari Shah** 
* **Jiayu Yao** 
* **Yucheng Huang** 
* **Zhenhao Ye** 



