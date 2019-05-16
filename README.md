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

Things that you need to install the software.

```
Python 3.7
Django 2.1.3
numpy 1.16.2
pandas 0.23.4
cvxopt 1.2.3
cvxpy 1.0.21
```

### Deploy to Local Server

A step by step instruction on how to run a server.

1. This system is written in Python 3.7, please make sure you have the same Python version.
2. Install all dependencies.

```
cd 95-720-Capstone-Project-master
pip install -r requirements.txt
```
2. Install Django framework, please run the following code in terminal.
```
pip install django
```
3. Install cvxopt and cvxpy package, please run the following code in terminal.
```
pip install cvxopt
pip install cvxpy
```
4. Go to the category of networkoptimizer.
```
cd 95-720-Capstone-Project-master/Programming Files/networkoptimizer
```
5. Run the server.
```
python manage.py runserver
```


## Authors

* **Askari Shah** 
* **Jiayu Yao** 
* **Yucheng Huang** 
* **Zhenhao Ye** 



