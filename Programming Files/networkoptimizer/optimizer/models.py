from django.db import models


# Create your models here.

class Search(models.Model):
    capacity = models.CharField(max_length=200)
    quality = models.CharField(max_length=200)


class Provider(models.Model):
    provider_id = models.CharField(max_length=200)
    provider_name = models.CharField(max_length=200)
    cost = models.CharField(max_length=200)
    quality = models.CharField(max_length=200)
    search = models.ForeignKey(Search, related_name="search", on_delete=models.CASCADE)


class CustomList(models.Model):
    must_exist = models.BooleanField()
    must_remove = models.BooleanField()






